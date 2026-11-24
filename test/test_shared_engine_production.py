"""All formal game decisions share the C++ search; TS remains emergency fallback."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class SharedDecisionEngineProductionTests(unittest.TestCase):
    def test_same_cpp_header(self):
        for name in ("selfplay.cpp","model_match.cpp","wasm_bridge.cpp"):
            self.assertIn('#include "search.hpp"',(ROOT/"engine"/name).read_text())
    def test_public_worker(self):
        app=(ROOT/"frontend/src/App.tsx").read_text()
        worker=(ROOT/"frontend/src/engine/search.worker.ts").read_text()
        self.assertIn("publicClone()",app)
        self.assertIn("publicHistoryRef.current.slice()",app)
        self.assertIn("loadWasmEngine",worker)
        self.assertIn("board.legalMoves()",worker)
    def test_fixed_time_and_full_replay(self):
        for name in ("self_play.yml","arena.yml"):
            s=(ROOT/".github/workflows"/name).read_text()
            self.assertIn("validate_replay.py",s)
            self.assertNotIn("sprt_shared.js",s)
        self.assertNotIn("node .shared-build/src/workers/self_play_shared.js",(ROOT/".github/workflows/self_play.yml").read_text())
if __name__=="__main__":unittest.main()
