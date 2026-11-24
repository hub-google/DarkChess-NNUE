"""V2 production data, training, arena and promotion contracts.

No remote mutation here. Only the explicit master-only promotion workflow may
apply a locally verified candidate and push it. Research artifacts are not
promotion authority.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src/training"),str(ROOT/"arena"),str(ROOT/"research_v2")]
WORKERS=15
GAMES_PER_WORKER=16
PAIRS_PER_SHARD=32
SHARDS_PER_BUDGET=8
BUDGETS=(25,100)
ALPHA=.025
MIN_GAIN=.005

def require(condition,message):
    if not condition: raise ValueError(message)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(",",":")).encode()).hexdigest()
def load(path):
    return json.loads(Path(path).read_text())
def save(path,obj):
    Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
def run_meta():
    return {"run":int(os.environ["GITHUB_RUN_ID"]),"attempt":int(os.environ["GITHUB_RUN_ATTEMPT"]),"commit":os.environ["GITHUB_SHA"]}
def seed_for(run,attempt,slot):
    require(run>0 and 0<attempt<256 and 0<slot<256,"invalid seed inputs")
    return run*65536+attempt*256+slot
def records(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def model_hash(path):
    p=Path(path)
    m=load(p/"champion.json")
    require(m["format"]==1 and m["inputSize"]==498,"wrong model format")
    require((p/"champion.bin").stat().st_size==m["bytes"],"model size mismatch")
    h=sha(p/"champion.bin")
    require(h==m["sha256"],"model hash mismatch")
    return h
def current_model():
    return model_hash(ROOT/"frontend/public/models")
def merge(root,out,budget):
    from validate_replay import validate_record
    meta=run_meta()
    h=current_model()
    out.mkdir(parents=True,exist_ok=True)
    all_games=[];ids=set();deals=set();shards=[]
    for worker in range(1,WORKERS+1):
        p=root/f"native-worker-{worker}"
        report=load(p/"throughput.json")
        seed=seed_for(meta["run"],meta["attempt"],worker)
        require(report["seed"]==seed and report["games"]==GAMES_PER_WORKER and report["budget_ms"]==budget,"wrong selfplay shard")
        games=list(records(p/"batch.jsonl.gz"))
        require(len(games)==GAMES_PER_WORKER,"missing games")
        for r in games:
            validate_record(r)
            d=digest(r["hid"])
            require(r["id"] not in ids and d not in deals,"duplicate game or deal")
            require(r["model"]=="nnue-"+h[:12] and r["seed"]==seed and r["budget_ms"]==budget and r["search_mode"]=="native-full-chance-timed","wrong teacher or source")
            require(all(k in r for k in ("q","v","qd","qn","qt","pi")),"missing search labels")
            ids.add(r["id"]);deals.add(d);all_games.append(r)
        require(sum(len(r["mov"]) for r in games)==report["moves"],"move count mismatch")
        shards.append({**report,"worker":worker,"file_sha256":sha(p/"batch.jsonl.gz")})
    require(len(all_games)==WORKERS*GAMES_PER_WORKER,"wrong total games")
    with (out/"new-games.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0,filename="") as f:
            for r in all_games: f.write((json.dumps(r,separators=(",",":"))+"\n").encode())
    manifest={**meta,"schema":1,"games":len(all_games),"model_sha256":h,"budget_ms":budget,
              "replay_sha256":sha(out/"new-games.jsonl.gz"),"shards":shards}
    save(out/"manifest.json",manifest)
    print(json.dumps(manifest,indent=2))

def train_cycle(root,out,source_run,source_attempt,recipe):
    import numpy as np
    import torch
    from train_ablation import (key, deal, sample_old, samples, combine, fit, GAMES,
                                DATASET, HfApi, hf_hub_download)
    from validate_replay import validate_record
    from export_nnue import export_model
    meta=run_meta()
    manifest=load(root/"manifest.json")
    h=current_model()
    require(manifest["run"]==source_run and manifest["attempt"]==source_attempt,"wrong source run")
    require(manifest["commit"]==meta["commit"] and manifest["model_sha256"]==h,"stale source model or code")
    require(manifest["games"]==WORKERS*GAMES_PER_WORKER and sha(root/"new-games.jsonl.gz")==manifest["replay_sha256"],"invalid replay artifact")
    new=list(records(root/"new-games.jsonl.gz"))
    require(len(new)==240 and len({deal(r) for r in new})==240,"wrong source games")
    for r in new: validate_record(r)
    new.sort(key=key)
    train,holdout=new[:GAMES],new[GAMES:]
    old=[];revision=None;invalid=0
    if recipe=="new-only":
        data=samples(train)
    elif recipe in ("mixed-teacher","mixed-outcomes"):
        token=os.environ["HF_TOKEN"]
        revision=HfApi(token=token).repo_info(repo_id=DATASET,repo_type="dataset").sha
        old_path=hf_hub_download(repo_id=DATASET,repo_type="dataset",filename="replay_buffer.jsonl.gz",revision=revision,token=token)
        old,invalid=sample_old(old_path,{deal(r) for r in new})
        data=combine(samples(train[:GAMES//2]),samples(old,recipe=="mixed-outcomes"))
    else: raise ValueError("unknown recipe")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    base=ROOT/"models/champion.nnue"
    export_model(base,out/"base")
    require(model_hash(out/"base")==h,"checkpoint and binary disagree")
    result=fit("candidate",data,samples(holdout),base,out)
    candidate=model_hash(out/"candidate")
    require(candidate==result["sha256"] and candidate!=h,"candidate hash or unchanged checkpoint")
    report={**meta,"schema":1,"source_run":source_run,"source_attempt":source_attempt,
            "source_manifest_sha256":sha(root/"manifest.json"),"recipe":recipe,"base_sha256":h,
            "candidate_sha256":candidate,"checkpoint_sha256":sha(out/"candidate/model.nnue"),
            "all_new_deals":[deal(r) for r in new],"old_train_deals":[deal(r) for r in old],
            "hf_revision":revision,"invalid_old_sample":invalid,"fit":result}
    save(out/"training.json",report)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("_deals")},indent=2))

def verify_training(root,run,attempt,commit):
    t=load(root/"training.json")
    require(t["run"]==run and t["attempt"]==attempt and t["commit"]==commit,"stale training provenance")
    require(t["base_sha256"]==model_hash(root/"base")==current_model(),"wrong baseline")
    require(t["candidate_sha256"]==model_hash(root/"candidate"),"wrong candidate")
    require(t["checkpoint_sha256"]==sha(root/"candidate/model.nnue"),"wrong checkpoint")
    require(t["candidate_sha256"]!=t["base_sha256"],"identical candidate")
    return t

def attach(root,result,training_run,training_attempt,slot,budget):
    meta=run_meta()
    t=verify_training(root,training_run,training_attempt,meta["commit"])
    r=load(result/"match.json")
    seed=seed_for(meta["run"],meta["attempt"],64+slot)
    require(r["seed"]==seed and r["budget_ms"]==budget and r["planned_pairs"]==PAIRS_PER_SHARD,"wrong arena setup")
    require(r["incomplete_games"]==0 and len(r["pair_scores"])==PAIRS_PER_SHARD,"incomplete arena")
    r.update(meta,slot=slot,training_run=training_run,training_attempt=training_attempt,
             training_sha256=sha(root/"training.json"),base_sha256=t["base_sha256"],
             candidate_sha256=t["candidate_sha256"],replay_sha256=sha(result/"match.jsonl.gz"))
    save(result/"match.json",r)

def pair_from_records(rows,seed,pair):
    """Compute candidate points independently from actual terminal outcomes."""
    require(len(rows)==2,"missing paired game")
    firsts=set();scores=[];bag=rows[0]["hid"];opening=int(rows[0]["mov"][0])
    require(opening//32==opening%32 and 0<=opening<1024,"invalid first flip")
    color_first=0 if bag[opening//32]<7 else 1
    for r in rows:
        require(r["hid"]==bag and r["mov"][0]==opening,"mismatched deal or opening")
        prefix=f"ablation-{seed}-{pair}-"
        require(r["id"].startswith(prefix),"wrong game ID")
        first=int(r["id"][len(prefix):])
        require(first in (0,1) and first not in firsts,"not swapped")
        firsts.add(first)
        result=r["res"]
        require(result in (-1,0,1),"invalid result")
        winner=0 if result==1 else 1
        candidate_color=color_first if first==1 else 1-color_first
        scores.append(.5 if result==0 else float(winner==candidate_color))
    return sum(scores),digest(bag)

def aggregate(shards,training,out,training_run,training_attempt):
    from validate_replay import validate_record
    from pair_stats import analyze
    meta=run_meta()
    t=verify_training(training,training_run,training_attempt,meta["commit"])
    excluded=set(t["all_new_deals"]+t["old_train_deals"]);seen=set();results={}
    for bi,budget in enumerate(BUDGETS):
        scores=[]
        for i in range(SHARDS_PER_BUDGET):
            slot=bi*SHARDS_PER_BUDGET+i+1
            p=shards/f"v2-arena-{slot}"
            r=load(p/"match.json")
            expected=seed_for(meta["run"],meta["attempt"],64+slot)
            require(all(r.get(k)==v for k,v in meta.items()),"wrong arena run")
            require(r["slot"]==slot and r["seed"]==expected and r["budget_ms"]==budget,"wrong shard")
            require(r["planned_pairs"]==PAIRS_PER_SHARD and len(r["pair_scores"])==PAIRS_PER_SHARD and r["incomplete_games"]==0,"incomplete shard")
            require(r["training_sha256"]==sha(training/"training.json") and r["base_sha256"]==t["base_sha256"] and r["candidate_sha256"]==t["candidate_sha256"],"mixed model versions")
            require(r["replay_sha256"]==sha(p/"match.jsonl.gz"),"replay hash mismatch")
            games=list(records(p/"match.jsonl.gz"))
            require(len(games)==2*PAIRS_PER_SHARD,"missing games")
            for pair in range(PAIRS_PER_SHARD):
                rows=games[2*pair:2*pair+2]
                for game in rows:validate_record(game)
                score,d=pair_from_records(rows,expected,pair)
                require(score==r["pair_scores"][pair],"reported score mismatch")
                require(d not in seen and d not in excluded,"duplicate or training deal in arena")
                seen.add(d);scores.append(score)
        stat=analyze(scores,SHARDS_PER_BUDGET*PAIRS_PER_SHARD,alpha=ALPHA,minimum_gain=MIN_GAIN)
        results[str(budget)]={"scores":scores,"stats":stat}
    gate={**meta,"schema":1,"training_run":training_run,"training_attempt":training_attempt,
          "training_sha256":sha(training/"training.json"),"base_sha256":t["base_sha256"],
          "candidate_sha256":t["candidate_sha256"],"verified_games":len(seen)*2,"results":results,
          "passed":all(r["stats"]["decision"]=="SUPERIOR" for r in results.values())}
    out.mkdir(parents=True,exist_ok=True);save(out/"gate.json",gate)
    print(json.dumps({**gate,"results":{k:v["stats"] for k,v in results.items()}},indent=2))
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("passed="+str(gate["passed"]).lower()+"\n")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:
            f.write("## 獨立配對棋力與晉升門檻\n")
            for b,r in results.items():
                s=r["stats"];f.write(f"- {b}ms：{s['pairs']} 組，候選得分 {s['candidate_score']:.2%}，{s['decision']}\n")
            f.write("晉升通過\n" if gate["passed"] else "未通過，不更換正式冠軍。\n")

def check_gate(g,t,training_sha,commit,arena_run,arena_attempt):
    from pair_stats import analyze
    require(g["run"]==arena_run and g["attempt"]==arena_attempt and g["commit"]==commit,"stale gate")
    require(g["training_run"]==t["run"] and g["training_attempt"]==t["attempt"] and g["training_sha256"]==training_sha,"wrong candidate provenance")
    require(g["base_sha256"]==t["base_sha256"] and g["candidate_sha256"]==t["candidate_sha256"],"model hashes changed")
    require(g["passed"] is True and g["verified_games"]==2*PAIRS_PER_SHARD*SHARDS_PER_BUDGET*len(BUDGETS),"incomplete gate")
    require(set(g["results"])==set(map(str,BUDGETS)),"missing budget")
    for budget in BUDGETS:
        s=g["results"][str(budget)]
        actual=analyze(s["scores"],SHARDS_PER_BUDGET*PAIRS_PER_SHARD,alpha=ALPHA,minimum_gain=MIN_GAIN)
        require(actual==s["stats"] and actual["decision"]=="SUPERIOR","gate failed independent recomputation")

def promote(training,gate_dir,training_run,training_attempt,arena_run,arena_attempt):
    from export_nnue import export_model
    require(os.environ["GITHUB_REF"]=="refs/heads/master","production write forbidden")
    meta=run_meta()
    t=verify_training(training,training_run,training_attempt,meta["commit"])
    gate=load(gate_dir/"gate.json")
    check_gate(gate,t,sha(training/"training.json"),meta["commit"],arena_run,arena_attempt)
    with tempfile.TemporaryDirectory() as tmp:
        export_model(training/"candidate/model.nnue",Path(tmp))
        require(model_hash(Path(tmp))==t["candidate_sha256"],"checkpoint export mismatch")
        shutil.copy2(training/"candidate/model.nnue",ROOT/"models/champion.nnue")
        for name in ("champion.json","champion.bin"):shutil.copy2(Path(tmp)/name,ROOT/"frontend/public/models"/name)
    save(ROOT/"metrics/promotion-v2.json",{"training":t,"gate":gate})
    print("Validated and staged candidate; workflow must compare-and-swap master before publishing.")

if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("merge");q.add_argument("root",type=Path);q.add_argument("out",type=Path);q.add_argument("budget",type=int,choices=BUDGETS)
    q=sub.add_parser("train");q.add_argument("root",type=Path);q.add_argument("out",type=Path);q.add_argument("run",type=int);q.add_argument("attempt",type=int);q.add_argument("recipe",choices=["new-only","mixed-teacher","mixed-outcomes"])
    q=sub.add_parser("attach");q.add_argument("root",type=Path);q.add_argument("result",type=Path);q.add_argument("run",type=int);q.add_argument("attempt",type=int);q.add_argument("slot",type=int);q.add_argument("budget",type=int,choices=BUDGETS)
    q=sub.add_parser("aggregate");q.add_argument("shards",type=Path);q.add_argument("training",type=Path);q.add_argument("out",type=Path);q.add_argument("run",type=int);q.add_argument("attempt",type=int)
    q=sub.add_parser("promote");q.add_argument("training",type=Path);q.add_argument("gate",type=Path);q.add_argument("run",type=int);q.add_argument("attempt",type=int);q.add_argument("arena_run",type=int);q.add_argument("arena_attempt",type=int)
    a=p.parse_args()
    if a.cmd=="merge":merge(a.root,a.out,a.budget)
    elif a.cmd=="train":train_cycle(a.root,a.out,a.run,a.attempt,a.recipe)
    elif a.cmd=="attach":attach(a.root,a.result,a.run,a.attempt,a.slot,a.budget)
    elif a.cmd=="aggregate":aggregate(a.shards,a.training,a.out,a.run,a.attempt)
    else:promote(a.training,a.gate,a.run,a.attempt,a.arena_run,a.arena_attempt)
