"""Fixed-budget, read-only replay ablation. No production writes or automatic promotion."""
import argparse
import gzip
import hashlib
import heapq
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/training"))
from board import DarkChessBoardPy, NONE
from train import (initialize_challenger, extract_features, value_target,
                   augment_features, MODEL_VERSION, CURRENT_INPUT_SIZE)
from export_nnue import export_model
from validate_replay import validate_record

SEED=612422
POSITIONS=32
GAMES=192
UPDATES=96
BATCH=128
DATASET="hub-google/DarkChess-NNUE-Data"
def digest(x):
    return hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()
def deal(rec):
    return digest(rec["hid"])
def key(rec):
    return digest([SEED,deal(rec),rec["id"]])
def records(path):
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
def sample_old(path,forbidden):
    # Bounded-memory deterministic deal-group sample, validated before use.
    pool=[]
    for i,r in enumerate(records(path)):
        d=deal(r)
        if d in forbidden: continue
        item=(-int(key(r),16),i,r)
        if len(pool)<512: heapq.heappush(pool,item)
        elif item>pool[0]: heapq.heapreplace(pool,item)
    out=[]; used=set(); invalid=0
    for _,_,r in sorted(pool,reverse=True):
        d=deal(r)
        if d in used: continue
        try: validate_record(r)
        except (ValueError,RuntimeError,TypeError,KeyError,AssertionError,IndexError):
            invalid+=1
            continue
        out.append(r);used.add(d)
        if len(out)==GAMES//2: break
    if len(out)!=GAMES//2: raise RuntimeError("not enough independently validated historical games")
    return out,invalid

def samples(games,result_only=False):
    x=[];y=[];z=[]
    for r in games:
        moves=r["mov"]
        if len(moves)<2: raise RuntimeError("game lacks known-color state")
        # Identical game weighting; short games repeat evenly spaced legal positions.
        chosen=np.floor(np.linspace(1,len(moves)-1,POSITIONS)).astype(int)
        b=DarkChessBoardPy(bag=r["hid"])
        k=0
        for ply,move in enumerate(moves):
            while k<len(chosen) and chosen[k]==ply:
                if b.side_to_move==NONE: raise ValueError("unknown side")
                feat=extract_features(b)
                result=float(r["res"])
                q=None if result_only or "q" not in r else float(r["q"][ply])
                depth=int(r["qd"][ply]) if "qd" in r else 0
                x.append(feat);y.append(value_target(result,q,depth,0.5));z.append(result);k+=1
            b.make_move(int(move),validate=True)
        if k!=POSITIONS: raise RuntimeError("position count")
    return np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),np.asarray(z,dtype=np.float32)

def combine(a,b):
    return tuple(np.concatenate([x,y],axis=0) for x,y in zip(a,b))

def fit(name,data,holdout,base,output):
    x,y,_=data
    if len(x)!=GAMES*POSITIONS: raise ValueError("unequal sample count")
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    model=initialize_challenger(str(base))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    losses=[]
    start=time.process_time()
    for epoch in range(2):
        order=rng.permutation(len(x))
        for i in range(0,len(x),BATCH):
            inds=order[i:i+BATCH]
            aug=[augment_features(x[j],y[j],int(rng.integers(4)),bool(rng.integers(2))) for j in inds]
            features=torch.from_numpy(np.stack([a[0] for a in aug]))
            targets=torch.tensor([a[1] for a in aug],dtype=torch.float32).reshape(-1,1)
            opt.zero_grad()
            loss=torch.nn.functional.mse_loss(model(features),targets)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
            loss.backward();opt.step();losses.append(float(loss.detach()))
    cpu=time.process_time()-start
    if len(losses)!=UPDATES: raise RuntimeError("unequal optimizer updates")
    model.eval()
    with torch.no_grad():
        pred=model(torch.from_numpy(holdout[0])).numpy().reshape(-1)
    result_mse=float(np.mean((pred-holdout[2])**2))
    teacher_mse=float(np.mean((pred-holdout[1])**2))
    directory=output/name;directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/"model.nnue"
    torch.save({"format_version":MODEL_VERSION,"input_size":CURRENT_INPUT_SIZE,"state_dict":model.state_dict()},checkpoint)
    export_model(checkpoint,directory)
    manifest=json.loads((directory/"champion.json").read_text())
    return {"arm":name,"training_games":GAMES,"samples":len(x),"updates":len(losses),
            "batch_size":BATCH,"cpu_seconds":cpu,"mean_train_loss":float(np.mean(losses)),
            "heldout_result_mse":result_mse,"heldout_teacher_mse":teacher_mse,"sha256":manifest["sha256"]}

def main(new_dir,output):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True,exist_ok=True)
    new=[]
    for p in sorted(new_dir.rglob("batch.jsonl.gz")):
        for r in records(p):
            validate_record(r)
            if r.get("search_mode")!="native-full-chance-timed": raise ValueError("wrong source")
            new.append(r)
    if len(new)!=240 or len({deal(r) for r in new})!=240 or len({r["id"] for r in new})!=240:
        raise RuntimeError("expected 240 unique training-only games")
    new.sort(key=key)
    train,holdout=new[:GAMES],new[GAMES:]
    if len(holdout)!=48: raise RuntimeError("heldout count")
    token=os.environ["HF_TOKEN"]
    revision=HfApi(token=token).repo_info(repo_id=DATASET,repo_type="dataset").sha
    old_path=hf_hub_download(repo_id=DATASET,repo_type="dataset",filename="replay_buffer.jsonl.gz",revision=revision,token=token)
    old,invalid=sample_old(old_path,{deal(r) for r in new})
    mixed_new=samples(train[:GAMES//2])
    old_full=samples(old)
    old_outcome=samples(old,True)
    held=samples(holdout)
    base=Path("models/champion.nnue")
    export_model(base,output/"base")
    base_hash=json.loads((output/"base/champion.json").read_text())["sha256"]
    current=json.loads(Path("frontend/public/models/champion.json").read_text())["sha256"]
    if base_hash!=current: raise RuntimeError("checkpoint and published binary differ")
    model=initialize_challenger(str(base)).eval()
    with torch.no_grad(): prediction=model(torch.from_numpy(held[0])).numpy().reshape(-1)
    report={"seed":SEED,"hf_revision":revision,"base_sha256":base_hash,"invalid_old_sample":invalid,
            "new_train_deal_hashes":[deal(r) for r in train],"new_holdout_deal_hashes":[deal(r) for r in holdout],
            "old_train_deal_hashes":[deal(r) for r in old],"base_heldout_result_mse":float(np.mean((prediction-held[2])**2)),
            "base_heldout_teacher_mse":float(np.mean((prediction-held[1])**2)),
            "arms":[],"scope":"equal fixed CPU optimizer workload; heldout loss is diagnostic, independent paired matches determine strength"}
    for name,data in [("A",samples(train)),("B",combine(mixed_new,old_full)),("C",combine(mixed_new,old_outcome))]:
        report["arms"].append(fit(name,data,held,base,output))
    (output/"training.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("deal_hashes")},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--new",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.new,a.output)
