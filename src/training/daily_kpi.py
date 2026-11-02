"""Fast daily DarkChess training KPI ledger; no additional matches or training."""
import argparse
import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

COLUMNS = [
    "date_taipei", "run_id", "recorded_at_utc", "replay_games", "new_games_estimate",
    "train_valid_games", "train_samples", "training_loss_last100",
    "validation_loss", "sprt_games", "sprt_score_pct", "sprt_elo_estimate",
    "sprt_decision", "sprt_champion", "sprt_challenger",
    "production_probe_status", "champion_promotion", "notes",
]


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def estimate_elo(score):
    if not 0 < score < 1:
        return ""
    return round(400 * math.log10(score / (1 - score)), 2)


def collect():
    snap = read_json("training_snapshot_candidate.json")
    prior = read_json(".github/training_last_snapshot.json")
    cache = read_json("training_cache/manifest.json")
    sprt = read_json("sprt_results.json")
    probe = read_json("production_probe_results.json")
    log = Path("training_kpi.log")
    losses = re.findall(r"Loss:\s*([0-9]+(?:\.[0-9]+)?)", log.read_text(encoding="utf-8", errors="replace")) if log.exists() else []
    now = datetime.now(ZoneInfo("UTC"))
    current = snap.get("total_games")
    previous = prior.get("total_games")
    delta = (max(0, int(current) - int(previous)) if current is not None and previous is not None and snap.get("ids_sha256") != prior.get("ids_sha256") else "")
    score = sprt.get("score_rate")
    scored = float(score) if isinstance(score, (float, int)) and sprt.get("games_played", 0) else None
    promotion = os.environ.get("KPI_PROMOTED", "unknown")
    probe_status = ("pass" if probe.get("production_full_depth_moves", 0) > 0 and probe.get("games_played") == 2 else "not_recorded")
    return {
        "date_taipei": now.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat(),
        "run_id": os.environ.get("GITHUB_RUN_ID", "manual"),
        "recorded_at_utc": now.isoformat(timespec="seconds"),
        "replay_games": current if current is not None else "",
        "new_games_estimate": delta,
        "train_valid_games": cache.get("valid_games", ""),
        "train_samples": cache.get("samples", ""),
        "training_loss_last100": losses[-1] if losses else "",
        "validation_loss": "",
        "sprt_games": sprt.get("games_played", ""),
        "sprt_score_pct": round(scored * 100, 3) if scored is not None else "",
        "sprt_elo_estimate": estimate_elo(scored) if scored is not None else "",
        "sprt_decision": sprt.get("decision", ""),
        "sprt_champion": sprt.get("champion_model", ""),
        "sprt_challenger": sprt.get("challenger_model", ""),
        "production_probe_status": probe_status,
        "champion_promotion": promotion,
        "notes": "No additional match; Elo is an unadjusted estimate; blank means unavailable.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="metrics/daily_kpi.csv")
    args = parser.parse_args()
    row = collect()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output.exists():
        with output.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    # Keep one daily row, choosing the latest observation within the day.
    rows = [r for r in rows if r.get("date_taipei") != row["date_taipei"]]
    rows.append(row)
    rows.sort(key=lambda r: r["date_taipei"])
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n### 每日訓練 KPI（不增加對弈）\n\n")
            f.write("| 指標 | 數值 |\n|---|---|\n")
            for k in COLUMNS:
                if k not in ("notes",):
                    f.write(f"| {k} | {row.get(k, '')} |\n")
            f.write("\n空白代表尚未量測；Elo 只是本次 SPRT 得分率換算，並非歷史 Elo 評級。\n")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
