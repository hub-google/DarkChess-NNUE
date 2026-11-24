"""Fixed-horizon paired analysis. No sequential promotion from this module.

Pair scores are candidate's total points over two swapped games: 0,.5,1,1.5,2.
Under the exchangeability null each nonzero pair deviation has an independent
random sign conditional on its magnitude. Draws and within-pair correlation are
preserved. The confidence interval uses a conservative bounded-mean inequality.
"""
import argparse
import json
import math
from pathlib import Path

def analyze(scores, planned_pairs, alpha=0.05, minimum_gain=0.0):
    if not isinstance(planned_pairs,int) or planned_pairs<1:
        raise ValueError("positive preregistered pair count required")
    if not 0<alpha<1 or not 0<=minimum_gain<0.5:
        raise ValueError("invalid thresholds")
    if len(scores)>planned_pairs or any(x not in (0,.5,1,1.5,2) for x in scores):
        raise ValueError("invalid pair data")
    n=len(scores)
    counts=[scores.count(x/2) for x in range(5)]
    if not n:
        return {"pairs":0,"planned_pairs":planned_pairs,"decision":"INCOMPLETE","pentanomial":counts}
    rate=sum(scores)/(2*n)
    # Hoeffding two-sided confidence interval for independent bounded pair means.
    radius=math.sqrt(math.log(2/alpha)/(2*n))
    lo=max(0.,rate-radius)
    hi=min(1.,rate+radius)
    # Exact conditional sign-randomization distribution. Integer half-point units.
    # sum of Binomial(m,.5) and 2*Binomial(k,.5).
    m=counts[1]+counts[3]
    k=counts[0]+counts[4]
    threshold=counts[3]+2*counts[4]
    def pmf(t):
        if t==0: return [1.]
        logp=-t*math.log(2)
        return [math.exp(math.lgamma(t+1)-math.lgamma(i+1)-math.lgamma(t-i+1)+logp) for i in range(t+1)]
    a,b=pmf(m),pmf(k)
    tails=[0.]*(m+2)
    for i in range(m,-1,-1): tails[i]=tails[i+1]+a[i]
    p=min(1.,max(0.,sum(x*tails[max(0,min(m+1,threshold-2*j))] for j,x in enumerate(b))))
    elo=None if rate in (0.,1.) else 400*math.log10(rate/(1-rate))
    decision="INCOMPLETE" if n<planned_pairs else (
      "SUPERIOR" if p<=alpha and lo>0.5+minimum_gain else
      "INFERIOR" if hi<0.5 else "INCONCLUSIVE")
    return {"pairs":n,"planned_pairs":planned_pairs,"pentanomial":counts,
      "candidate_score":rate,"score_interval_hoeffding":[lo,hi],
      "relative_elo_descriptive":elo,"one_sided_exchangeability_p":p,
      "alpha":alpha,"minimum_score_gain":minimum_gain,"decision":decision,
      "note":"Fixed-horizon test; descriptive Elo only. No optional stopping; independent pairs required."}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("--output",required=True)
    p.add_argument("--alpha",type=float,default=.05);p.add_argument("--minimum-gain",type=float,default=0.)
    a=p.parse_args();r=json.loads(Path(a.input).read_text())
    if r.get("incomplete_games",0): raise SystemExit("Incomplete games: cannot assess promotion")
    out=analyze(r["pair_scores"],r["planned_pairs"],a.alpha,a.minimum_gain)
    Path(a.output).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
