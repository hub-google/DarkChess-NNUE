"""Verify cross-run artifacts originate from a successful master workflow."""
import json
import os
import subprocess
import sys
run=int(sys.argv[1]);attempt=int(sys.argv[2]);allowed=sys.argv[3:]
if run<=0 or attempt<=0:raise SystemExit("invalid run")
if run==int(os.environ["GITHUB_RUN_ID"]):
    if attempt!=int(os.environ["GITHUB_RUN_ATTEMPT"]):raise SystemExit("wrong current attempt")
else:
    path=f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run}"
    info=json.loads(subprocess.check_output(["gh","api",path],text=True))
    if (info["head_branch"]!="master" or info["status"]!="completed" or info["conclusion"]!="success"
        or info["run_attempt"]!=attempt or info["head_sha"]!=os.environ["GITHUB_SHA"]
        or info["path"].split("@")[0] not in allowed):
        raise SystemExit("refusing unverified, stale, or research run")
