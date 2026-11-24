import itertools
import unittest
from pair_stats import analyze

class PairStatsTests(unittest.TestCase):
    def test_all_draws(self):
        r=analyze([1.]*100,100)
        self.assertEqual(r["pentanomial"],[0,0,100,0,0])
        self.assertEqual(r["decision"],"INCONCLUSIVE")
        self.assertAlmostEqual(r["one_sided_exchangeability_p"],1)
    def test_exact_sign_distribution(self):
        for scores in ([0,1.5,2],[.5,1.5,2,2],[0,0,1,2], [0.5,1,1.5]):
            vals=[int(2*(s-1)) for s in scores if s!=1]
            target=sum(vals)
            possible=[sum(abs(v)*d for v,d in zip(vals,ds)) for ds in itertools.product([-1,1],repeat=len(vals))]
            expected=sum(x>=target for x in possible)/len(possible)
            self.assertAlmostEqual(analyze(list(scores),len(scores))["one_sided_exchangeability_p"],expected)
    def test_incomplete_cannot_pass(self):
        self.assertEqual(analyze([2.]*200,201)["decision"],"INCOMPLETE")
    def test_strong_signal(self):
        self.assertEqual(analyze([2.]*200,200)["decision"],"SUPERIOR")
        self.assertEqual(analyze([0.]*200,200)["decision"],"INFERIOR")
    def test_bad_scores(self):
        for x in (float("nan"),-1,.1,3):
            with self.assertRaises(ValueError):analyze([x],1)
    def test_a_a_calibration(self):
        import random
        rng=random.Random(0x218a)
        passes=0
        for _ in range(1000):
            scores=[rng.choice([0,.5,1,1.5,2]) for _ in range(100)]
            passes+=analyze(scores,100)["decision"]=="SUPERIOR"
        self.assertLessEqual(passes,50)
if __name__=="__main__":unittest.main()
