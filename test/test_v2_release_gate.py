import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"pipeline"))
from release import seed_for,pair_from_records,check_gate,PAIRS_PER_SHARD,SHARDS_PER_BUDGET,ALPHA,MIN_GAIN
from pair_stats import analyze

class ReleaseGateTests(unittest.TestCase):
    def test_seed_domains(self):
        self.assertNotEqual(seed_for(100,1,1),seed_for(100,1,65))
        self.assertNotEqual(seed_for(100,1,1),seed_for(100,2,1))
    def test_pair_recomputation(self):
        bag=[0]*32
        a={"hid":bag,"mov":[0],"id":"ablation-10-0-0","res":-1}
        b={"hid":bag,"mov":[0],"id":"ablation-10-0-1","res":1}
        self.assertEqual(pair_from_records([a,b],10,0)[0],2)
        with self.assertRaises(ValueError):pair_from_records([a,a],10,0)
    def gate(self):
        n=PAIRS_PER_SHARD*SHARDS_PER_BUDGET
        scores=[2.]*n
        stat=analyze(scores,n,alpha=ALPHA,minimum_gain=MIN_GAIN)
        t={"run":1,"attempt":1,"base_sha256":"a","candidate_sha256":"b"}
        g={"run":2,"attempt":1,"commit":"c","training_run":1,"training_attempt":1,
           "training_sha256":"d","base_sha256":"a","candidate_sha256":"b",
           "passed":True,"verified_games":4*n,
           "results":{str(k):{"scores":scores[:],"stats":stat.copy()} for k in (25,100)}}
        return t,g
    def test_promotion_recomputes_complete_scores(self):
        t,g=self.gate();check_gate(g,t,"d","c",2,1)
        for edit in ("partial","stale","forged","wrong_hash"):
            broken=copy.deepcopy(g)
            if edit=="partial":broken["results"]["25"]["scores"].pop()
            if edit=="stale":broken["commit"]="other"
            if edit=="forged":broken["results"]["100"]["scores"]=[1.]*len(broken["results"]["100"]["scores"])
            if edit=="wrong_hash":broken["candidate_sha256"]="other"
            with self.assertRaises(ValueError):check_gate(broken,t,"d","c",2,1)
if __name__=="__main__":unittest.main()
