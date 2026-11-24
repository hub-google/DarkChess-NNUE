import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"arena"))
from pair_stats import analyze
def main(root,out):
    expected=[("expecti","mcts",8101),("expecti","puct",8201),("mcts","puct",8301)]
    result={}
    for a,b,seed in expected:
        scores=[];over=0
        for s in range(seed,seed+4):
            r=json.loads((Path(root)/f"monte-{a}-{b}-{s}"/"match.json").read_text())
            if r["baseline"]!=a or r["candidate"]!=b or r["seed"]!=s or r["budget_ms"]!=25 or r["planned_pairs"]!=16 or r["incomplete_games"] or len(r["pair_scores"])!=16:
                raise ValueError("invalid experiment shard")
            scores.extend(r["pair_scores"])
            over+=sum(x["over_budget_2x"] for x in r["stats"])
        result[a+"-"+b]={"comparison":analyze(scores,64,alpha=.05/3),"over_budget_2x":over}
    report={"results":result,"scope":"three exploratory paired comparisons; no auto-adoption; MCTS tree depth is not minimax depth"}
    Path(out).write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main(*sys.argv[1:])
