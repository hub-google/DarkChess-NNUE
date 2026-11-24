"""Alternate order on the same CPU. Every executable checks root parity."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

def main(model,corpus,out):
    data=Path(corpus).read_bytes()
    variants=["reference","delta","tt","combined","native_simd"]
    rows={k:[] for k in variants}
    for i in range(7):
        order=variants[i%len(variants):]+variants[:i%len(variants)]
        if i%2: order.reverse()
        for name in order:
            r=json.loads(subprocess.check_output(["/tmp/dc-"+name,model],input=data,timeout=180))
            if not r["parity"] or not r["budget_fallback"]: raise RuntimeError("parity")
            rows[name].append(r)
    results={}
    for name,rs in rows.items():
        med=statistics.median(x["ms"] for x in rs)
        results[name]={"median_ms":med,"nodes":rs[0]["nodes"],"cache_hits":rs[0]["cache_hits"],
                       "speedup":statistics.median(x["ms"] for x in rows["reference"])/med,
                       "root_scores":rs[0]["root_scores"],"max_error":max(x["max_error"] for x in rs)}
    result={"variants":results,"raw":rows,"scope":"same exact-depth corpus, seven alternating runs; host ISA flags, not proof of chess strength"}
    Path(out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(results,indent=2))
if __name__=="__main__": main(*sys.argv[1:])
