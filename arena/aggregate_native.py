"""Validate all predeclared shards; no production promotion."""
import json
from pathlib import Path
from pair_stats import analyze

EXPECTED={25:8,100:4}
def aggregate(root):
    rows=[json.loads(p.read_text()) for p in Path(root).rglob("native.json")]
    results={}
    seen_deals=set()
    for budget,shards in EXPECTED.items():
        batch=[r for r in rows if r["budget_ms"]==budget]
        if len(batch)!=shards: raise ValueError(f"missing shards for {budget}ms")
        scores=[]; seeds=set(); stats=[{},{}]
        for r in batch:
            if r["seed"] in seeds or r["incomplete_games"] or r["planned_pairs"]!=16 or len(r["pair_scores"])!=16 or len(r["games"])!=16:
                raise ValueError("incomplete or duplicate shard")
            seeds.add(r["seed"])
            for game,score in zip(r["games"],r["pair_scores"]):
                if game["deal_hash"] in seen_deals or sorted(g["first"] for g in game["games"])!=[0,1] or sum(g["score"] for g in game["games"])!=score:
                    raise ValueError("invalid matched pair")
                seen_deals.add(game["deal_hash"])
            scores.extend(r["pair_scores"])
            for i,s in enumerate(r["stats"]):
                for k,v in s.items():
                    stats[i][k]=max(stats[i].get(k,0),v) if k=="max_search_ms" else stats[i].get(k,0)+v
        if len({(r["baseline_sha"],r["candidate_sha"],r["model_sha256"]) for r in batch})!=1:
            raise ValueError("mixed versions")
        for s in stats:
            s["mean_depth"]=s["depth_sum"]/s["moves"]
            s["mean_search_ms"]=s["search_ms"]/s["moves"]
            s["mean_decision_ms"]=s["decision_ms"]/s["moves"]
            s["nodes_per_search_second"]=1000*s["nodes"]/max(s["search_ms"],1e-12)
        results[str(budget)]={"stats":analyze(scores,shards*16,alpha=.025),"baseline_ts":stats[0],"native_cpp":stats[1],"seeds":sorted(seeds)}
    if len(rows)!=sum(EXPECTED.values()): raise ValueError("unexpected extra shards")
    return {"results":results,"scope":"exploratory frozen TS versus native, identical weights, fixed search budgets, Bonferroni alpha .025 per budget; no automatic promotion"}

if __name__=="__main__":
    import sys,os
    result=aggregate(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:
            f.write("## 凍結 TS 對 C++ 原生等時間探索性實驗\n\n")
            for budget,r in result["results"].items():
                s=r["stats"];a=r["baseline_ts"];b=r["native_cpp"]
                f.write(f"- {budget}ms：{s['pairs']} 組，C++ 得分率 {s['candidate_score']:.2%}，保守區間 {s['score_interval_hoeffding']}，判定 {s['decision']}；TS / C++ 平均深度 {a['mean_depth']:.2f} / {b['mean_depth']:.2f}；平均總決策時間 {a['mean_decision_ms']:.2f} / {b['mean_decision_ms']:.2f} ms。\n")
            f.write("\n這是探索性比較，不自動晉升、不宣稱世界排名。\n")
