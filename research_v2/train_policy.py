"""Train a small public-state policy on the already verified new replay pilot."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/training"))
from train import extract_features, augment_features
from board import DarkChessBoardPy, NONE
from policy_train import PolicyNet, policy_loss, transform_actions, transform_legal_mask
from validate_replay import validate_record

SEED=612422
def digest(x): return hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()
def key(r): return digest([SEED,digest(r["hid"]),r["id"]])
def examples(games):
    rows=[]
    for rec in games:
        b=DarkChessBoardPy(bag=rec["hid"])
        valid=[i for i,p in enumerate(rec["pi"]) if i>0 and p and rec["qd"][i]>0]
        chosen=set(valid[j] for j in np.unique(np.floor(np.linspace(0,len(valid)-1,min(32,len(valid)))).astype(int))) if valid else set()
        for i,m in enumerate(rec["mov"]):
            if i in chosen:
                if b.side_to_move==NONE: raise ValueError("unknown side")
                actions=np.full(16,-1,dtype=np.int64);weights=np.zeros(16,dtype=np.float32)
                for j,(a,p) in enumerate(rec["pi"][i]):
                    actions[j]=a;weights[j]=p
                legal=np.zeros(1024,dtype=np.uint8);legal[list(b.generate_legal_moves())]=1
                rows.append((extract_features(b),actions,weights,np.packbits(legal,bitorder="little")))
            b.make_move(int(m),validate=True)
    return rows

def batch(rows,rng=None):
    out=[]
    for x,a,w,l in rows:
        sym=int(rng.integers(4)) if rng is not None else 0
        swap=bool(rng.integers(2)) if rng is not None else False
        out.append((augment_features(x,0.,sym,swap)[0],transform_actions(a,sym),w,transform_legal_mask(l,sym)))
    return tuple(torch.from_numpy(np.stack(items)) for items in zip(*out))

def main(root,out):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(SEED)
    games=[]
    for p in sorted(Path(root).rglob("batch.jsonl.gz")):
        with gzip.open(p,"rt") as f:
            for line in f:
                r=json.loads(line);validate_record(r);games.append(r)
    if len(games)!=240 or len({digest(r["hid"]) for r in games})!=240:raise RuntimeError("invalid dataset")
    games.sort(key=key)
    tr,va=examples(games[:192]),examples(games[192:])
    if not tr or not va:raise RuntimeError("missing policy examples")
    model=PolicyNet();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    updates=0
    for _ in range(4):
        order=rng.permutation(len(tr))
        for i in range(0,len(order),128):
            x,a,w,l=batch([tr[j] for j in order[i:i+128]],rng)
            opt.zero_grad();loss=policy_loss(model(x),a,w,l)
            if not torch.isfinite(loss):raise ValueError("nonfinite policy loss")
            loss.backward();opt.step();updates+=1
    model.eval()
    x,a,w,l=batch(va)
    with torch.no_grad():
        logits=model(x)
        ce=float(policy_loss(logits,a,w,l))
        uniform=float(policy_loss(torch.zeros_like(logits),a,w,l))
        predicted=logits.masked_fill(~l,-1e9).argmax(1)
        teacher=a.gather(1,w.argmax(1,keepdim=True)).squeeze(1)
        accuracy=float((predicted==teacher).float().mean())
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    state=model.state_dict();data=b"".join(state[k].detach().cpu().numpy().astype("<f4").tobytes() for k in ("net.0.weight","net.0.bias","net.2.weight","net.2.bias"))
    (out/"policy.bin").write_bytes(data)
    report={"training_games":192,"heldout_games":48,"training_positions":len(tr),"heldout_positions":len(va),
            "updates":updates,"heldout_cross_entropy":ce,"uniform_cross_entropy":uniform,"heldout_top_teacher_accuracy":accuracy,
            "sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),"seed":SEED,
            "scope":"public policy teacher; heldout accuracy is not strength"}
    (out/"policy.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main(*sys.argv[1:])
