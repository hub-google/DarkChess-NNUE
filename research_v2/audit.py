"""Read-only inventory and deterministic replay sample. No model-value claims."""
import argparse
import collections
import gzip
import hashlib
import heapq
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from validate_replay import validate_record

def audit(path, sample_per_group=128):
    counts=collections.Counter()
    results=collections.Counter()
    versions=collections.Counter()
    labels=collections.Counter()
    groups=collections.defaultdict(list)
    ids=set()
    unique_games=set()
    for serial,line in enumerate(gzip.open(path,"rt",encoding="utf-8")):
        if not line.strip(): continue
        counts["records"]+=1
        try:
            r=json.loads(line)
            if not isinstance(r,dict): raise ValueError("record not object")
        except (ValueError,TypeError):
            counts["malformed_json"]+=1
            continue
        gid=str(r.get("id") or "")
        if not gid: counts["missing_id"]+=1
        elif gid in ids: counts["duplicate_id"]+=1
        else: ids.add(gid)
        versions[str(r.get("ver","missing"))]+=1
        results[str(r.get("res","missing"))]+=1
        moves=r.get("mov")
        length=len(moves) if isinstance(moves,list) else 0
        if not length: counts["missing_moves"]+=1
        if "hid" in r and length:
            signature=hashlib.sha256(json.dumps([r["hid"],moves],separators=(",",":")).encode()).digest()
            if signature in unique_games: counts["duplicate_deal_and_moves"]+=1
            else: unique_games.add(signature)
        for name in ("q","v","qd","pi","qn","qt"):
            field=r.get(name)
            if isinstance(field,list) and len(field)==length and length: labels[name]+=1
        group=str(r.get("ver","missing"))+" | "+str(r.get("model","missing"))
        key=int.from_bytes(hashlib.sha256((gid+":"+str(serial)).encode()).digest()[:8],"big")
        heap=groups[group]
        item=(-key,serial,r)
        if len(heap)<sample_per_group: heapq.heappush(heap,item)
        elif item>heap[0]: heapq.heapreplace(heap,item)
    validation={}
    for group,heap in sorted(groups.items()):
        valid=0; errors=collections.Counter(); plies=0
        for _,_,r in heap:
            try:
                plies+=validate_record(r); valid+=1
            except Exception as e:
                errors[type(e).__name__+": "+str(e)[:160]]+=1
        validation[group]={"sampled":len(heap),"valid":valid,"valid_plies":plies,"errors":dict(errors)}
    if not counts["records"]: raise RuntimeError("empty dataset")
    return {"counts":dict(counts),"unique_ids":len(ids),"unique_deal_and_moves":len(unique_games),"versions":dict(versions),"outcomes":dict(results),"complete_label_arrays":dict(labels),"sample_validation":validation,"method":"full inventory; deterministic per-version/model sample; no training value conclusion"}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--local");p.add_argument("--output",default="research_v2/audit.json");p.add_argument("--sample",type=int,default=128);a=p.parse_args()
    if a.sample<1: raise ValueError("sample must be positive")
    path=a.local
    if not path:
        from huggingface_hub import hf_hub_download
        token=os.environ.get("HF_TOKEN")
        if not token: raise RuntimeError("HF_TOKEN is missing")
        path=hf_hub_download("hub-google/DarkChess-NNUE-Data","replay_buffer.jsonl.gz",repo_type="dataset",token=token)
    report=audit(path,a.sample)
    Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
