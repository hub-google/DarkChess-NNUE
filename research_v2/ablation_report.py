"""All five comparisons are fixed-horizon exploratory tests, never deletion authority."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"arena"))
from pair_stats import analyze

def main(root,training,output):
    expected={"base-A":7101,"base-B":7201,"base-C":7301,"A-B":7401,"A-C":7501}
    report={"training":json.loads(Path(training).read_text()),"matches":{}}
    for name,seed in expected.items():
        rows=[]
        for offset in range(4):
            r=json.loads((Path(root)/f"ablation-{name}-{seed+offset}"/"match.json").read_text())
            if r["seed"]!=seed+offset or r["planned_pairs"]!=32 or len(r["pair_scores"])!=32 or r["incomplete_games"] or r["budget_ms"]!=25:
                raise ValueError("missing or invalid shard")
            rows.extend(r["pair_scores"])
        report["matches"][name]=analyze(rows,128,alpha=.01)
    report["scope"]="Five preregistered comparisons; Bonferroni alpha .01; exploratory only. Non-significance is not equivalence or permission to delete old data."
    Path(output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["matches"],indent=2))
    print(report["scope"])
if __name__=="__main__": main(*sys.argv[1:])
