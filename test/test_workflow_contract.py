"""Formal V2 five-workflow contracts; no schedule or implicit repeat."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/".github/workflows"
class WorkflowContractTests(unittest.TestCase):
    def read(self,name):return (ROOT/name).read_text(encoding="utf-8")
    def test_exact_five(self):
        self.assertEqual({p.name for p in ROOT.glob("*.yml")},{"self_play.yml","train.yml","arena.yml","promote.yml","deploy_pages.yml"})
    def test_no_periodic_or_self_dispatch(self):
        for p in ROOT.glob("*.yml"):
            s=p.read_text()
            self.assertNotRegex(s,r"(?m)^  schedule:")
            self.assertIn("github.ref == 'refs/heads/master'",s)
        self.assertNotIn("gh workflow run self_play",self.read("self_play.yml"))
    def test_chain_is_explicit_and_guarded(self):
        s=self.read("self_play.yml")
        self.assertIn("max-parallel: 15",s)
        self.assertIn("needs: verified",s)
        self.assertIn("inputs.continue_cycle",s)
        self.assertIn("pipeline/release.py merge",s)
        self.assertIn("upload_batch.py",s)
        self.assertIn("pipeline/release.py train",self.read("train.yml"))
        a=self.read("arena.yml")
        self.assertIn("pipeline/release.py aggregate",a)
        self.assertIn("inputs.allow_promotion && needs.gate.outputs.passed == 'true'",a)
        p=self.read("promote.yml")
        self.assertIn("pipeline/release.py promote",p)
        self.assertIn('test "$(git rev-parse origin/master)" = "$GITHUB_SHA"',p)
        self.assertNotIn("--force",p)
    def test_site_build_and_live_acceptance(self):
        d=self.read("deploy_pages.yml")
        for token in ("bash engine/build_wasm.sh","VITE_DEFAULT_SEARCH_ENGINE=wasm","engine/browser_wasm_test.cjs","engine/live_smoke.cjs","needs: [build,deploy]"):
            self.assertIn(token,d)
if __name__=="__main__":unittest.main()
