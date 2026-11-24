"""Production scheduling and isolation contracts; research is read-only."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/".github"/"workflows"

class WorkflowContractTests(unittest.TestCase):
    def read(self,name): return (ROOT/name).read_text(encoding="utf-8")
    def test_manual_only_production_and_explicit_continuation(self):
        s=self.read("self_play.yml")
        t=self.read("train.yml")
        for text in (s,t):
            self.assertIn("workflow_dispatch:",text)
            self.assertNotRegex(text,r"(?m)^  (schedule|push):")
            self.assertIn("github.ref == 'refs/heads/master'",text)
        self.assertIn("default: false",s)
        self.assertIn("inputs.continuous && !cancelled()",s)
        self.assertIn("--ref master --field continuous=true",s)
        self.assertNotIn("recovery-watchdog:",s)
        self.assertNotIn("monthly-keepalive:",s)
        self.assertIn("CURRENT_STATUS",s)
    def test_writer_and_artifact_contract(self):
        s=self.read("self_play.yml")
        for required in ("max-parallel: 15","BUDGET=18000","needs: selfplay-workers",
                         "needs.upload.result == 'success'","needs.selfplay-workers.result == 'success'",
                         "pattern: selfplay-worker-*-${{ github.run_id }}","          path: .",
                         "merge-multiple: true","CHECKPOINTS\" -ne 15",
                         "VERIFIED all 15 worker Artifacts","HF upload or remote read-back failed",
                         "resume_state/*.json","resume_state/*.memo","fail-fast: false","trap stop_child INT TERM"):
            self.assertIn(required,s)
        self.assertEqual(s.count("python -u src/workers/upload_batch.py"),1)
        self.assertEqual(s.count("  upload:"),1)
        self.assertNotIn("selfplay-wave-",s)
    def test_training_data_and_promotion_gates(self):
        t=self.read("train.yml")
        for required in ("needs: hf-replay-consolidation","group: darkchess-hf-stage-writer",
                         "new_replay_gate.py","replay_gate.py","validate_training_readiness.py",
                         "git add models/research.nnue .github/training_last_snapshot.json",
                         "steps.validate.outputs.passed == 'true'","steps.production_probe.outcome == 'success'",
                         "steps.promote.outputs.promoted == 'true'"):
            self.assertIn(required,t)
        for retired in ("hf-backfill:","python-regression:","six-ply-profile:","depth-pilot:","bootstrap-research:","policy_train.py"):
            self.assertNotIn(retired,t)
    def test_no_automatic_evidence_deletion_or_cancelled_upload(self):
        for p in ROOT.glob("*.yml"):
            s=p.read_text(encoding="utf-8")
            self.assertNotIn("prune_action_runs.py",s)
            for line in s.splitlines():
                if line.strip().startswith("if:") and "always()" in line:
                    self.assertIn("!cancelled()",line,(p.name,line))
    def test_deployment_is_master_only(self):
        self.assertIn("github.ref == 'refs/heads/master'",self.read("deploy_pages.yml"))

if __name__=="__main__": unittest.main()
