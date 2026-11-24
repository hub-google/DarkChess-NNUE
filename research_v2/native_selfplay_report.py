"""Validate private research batches and aggregate without modifying HF or champion."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/training"))
from validate_replay import validate_record

def main(root, output):
    reports=[]
    all_ids=set()
    for p in sorted(Path(root).rglob("batch.jsonl.gz")):
        r=json.loads(p.with_name("throughput.json").read_text())
        ids=set()
        depths=[]
        with gzip.open(p,"rt") as fp:
            for line in fp:
                rec=json.loads(line)
                validate_record(rec)
                if rec["id"] in all_ids or rec["id"] in ids or rec["seed"]!=r["seed"] or rec["budget_ms"]!=r["budget_ms"]:
                    raise ValueError("duplicate or wrong source replay")
                if rec["search_mode"]!="native-full-chance-timed": raise ValueError("wrong teacher")
                ids.add(rec["id"])
                depths.extend(rec["qd"])
        if len(ids)!=r["games"] or len(depths)!=r["moves"] or sum(d==0 for d in depths)!=r["depth_zero"]:
            raise ValueError("incomplete shard")
        all_ids.update(ids)
        r["mean_depth"]=sum(depths)/len(depths)
        r["games_per_wall_hour"]=3600*r["games"]/r["wall_seconds"]
        r["file_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
        reports.append(r)
    expected={4102501:48,4102502:48,4102503:48,4102504:48,4110001:24,4110002:24}
    if len(reports)!=len(expected) or {r["seed"]:r["games"] for r in reports}!=expected:
        raise ValueError("missing predefined pilot shards")
    result={"games":len(all_ids),"shards":reports,"scope":"training-only seeds, no heldout arena games, pilot throughput not strength"}
    Path(output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main(sys.argv[1],sys.argv[2])
