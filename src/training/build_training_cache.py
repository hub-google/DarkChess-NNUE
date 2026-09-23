import glob
import gzip
import heapq
import json
import os
from pathlib import Path
import shutil
import zlib

import numpy as np
import torch

from board import DarkChessBoardPy, NONE
from nnue_eval import ModelEvaluator
from replay_format import is_supported_replay_version
from search import ChanceSearch
from train import (
    CURRENT_INPUT_SIZE,
    extract_features,
    load_model_file,
    select_training_plies,
)


def _candidate_push(heap, capacity, score, serial, record):
    if capacity <= 0:
        return
    item = (float(score), int(serial), record)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def _valid_game_header(game):
    version = str(game.get("ver", ""))
    if not is_supported_replay_version(version):
        return False
    moves = game.get("mov")
    if not isinstance(moves, list) or not moves:
        return False
    result = float(game.get("res", 99))
    if result not in (-1.0, 0.0, 1.0):
        return False
    root_values = game.get("q")
    static_values = game.get("v")
    if root_values is not None and len(root_values) != len(moves):
        return False
    if static_values is not None and len(static_values) != len(moves):
        return False
    return True


class CacheShardWriter:
    def __init__(self, output_dir, shard_size):
        self.output_dir = Path(output_dir)
        self.shard_size = max(1, int(shard_size))
        self.shard_index = 0
        self.total_samples = 0
        self._rows = []

    def add(self, row):
        self._rows.append(row)
        if len(self._rows) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._rows:
            return
        path = self.output_dir / f"cache_{self.shard_index:04d}.npz"
        features = np.stack([row["features"] for row in self._rows]).astype(np.float16)
        np.savez_compressed(
            path,
            features=features,
            targets=np.asarray([row["target"] for row in self._rows], dtype=np.float32),
            q=np.asarray([row["q"] for row in self._rows], dtype=np.float32),
            v=np.asarray([row["v"] for row in self._rows], dtype=np.float32),
            priority=np.asarray([row["priority"] for row in self._rows], dtype=np.float32),
            sample_index=np.asarray([row["sample_index"] for row in self._rows], dtype=np.int64),
            game_ts=np.asarray([row["game_ts"] for row in self._rows], dtype=np.int64),
            ply=np.asarray([row["ply"] for row in self._rows], dtype=np.int16),
            source_model=np.asarray([row["source_model"] for row in self._rows], dtype="S64"),
        )
        self.total_samples += len(self._rows)
        self.shard_index += 1
        self._rows.clear()


def _candidate_records(top_heap, random_heap):
    merged = {}
    for _, _, record in top_heap + random_heap:
        merged[(record["game_id"], record["ply"])] = record
    return merged


def build_base_cache(files, output_dir, max_positions, max_samples, shard_size, candidate_capacity):
    writer = CacheShardWriter(output_dir, shard_size)
    top_capacity = max(1, candidate_capacity // 2) if candidate_capacity else 0
    random_capacity = max(0, candidate_capacity - top_capacity)
    top_heap = []
    random_heap = []
    serial = 0
    invalid_games = 0
    valid_games = 0

    for replay_path in files:
        with gzip.open(replay_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if writer.total_samples + len(writer._rows) >= max_samples:
                    break
                if not line.strip():
                    continue
                try:
                    game = json.loads(line)
                    if not _valid_game_header(game):
                        continue
                    moves = [int(move) for move in game["mov"]]
                    result = float(game["res"])
                    selected = set(
                        int(ply)
                        for ply in select_training_plies(
                            game,
                            moves,
                            max_positions,
                        )
                    )
                    if not selected:
                        continue

                    root_values = game.get("q")
                    static_values = game.get("v")
                    board = DarkChessBoardPy(bag=game["hid"])
                    pending_rows = []

                    for ply, move in enumerate(moves):
                        if board.side_to_move != NONE and ply in selected:
                            q_old = (
                                float(root_values[ply])
                                if root_values is not None
                                else float(result)
                            )
                            v_old = (
                                float(static_values[ply])
                                if static_values is not None
                                else q_old
                            )
                            if not -1.0 <= q_old <= 1.0:
                                raise ValueError(f"invalid q at ply {ply}: {q_old}")
                            if not -1.0 <= v_old <= 1.0:
                                raise ValueError(f"invalid v at ply {ply}: {v_old}")
                            target = (
                                0.5 * result + 0.5 * q_old
                                if root_values is not None
                                else result
                            )
                            source_model = game.get("model", "unknown")
                            if board.side_to_move == 0:
                                source_model = game.get("red_model", source_model)
                            elif board.side_to_move == 1:
                                source_model = game.get("black_model", source_model)
                            sample_index = writer.total_samples + len(writer._rows) + len(pending_rows)
                            pending_rows.append(
                                {
                                    "features": extract_features(board, CURRENT_INPUT_SIZE),
                                    "target": float(target),
                                    "q": q_old,
                                    "v": v_old,
                                    "priority": abs(q_old - v_old),
                                    "sample_index": sample_index,
                                    "game_ts": int(game.get("ts", 0)),
                                    "ply": ply,
                                    "source_model": str(source_model),
                                }
                            )

                        board.make_move(move, validate=True)

                    over, replay_result = board.is_game_over()
                    if not over or float(replay_result) != result:
                        raise ValueError(
                            f"terminal mismatch: recorded={result}, actual={(over, replay_result)}"
                        )

                    valid_games += 1
                    for row in pending_rows:
                        if writer.total_samples + len(writer._rows) >= max_samples:
                            break
                        writer.add(row)
                        if root_values is None:
                            continue
                        record = {
                            "game_id": str(game["id"]),
                            "ply": int(row["ply"]),
                            "sample_index": int(row["sample_index"]),
                            "q_old": float(row["q"]),
                            "result": result,
                        }
                        old_gap = float(row["priority"])
                        hash_score = zlib.crc32(
                            f"{record['game_id']}:{record['ply']}".encode("utf-8")
                        ) / float(2**32 - 1)
                        _candidate_push(
                            top_heap,
                            top_capacity,
                            old_gap,
                            serial,
                            record,
                        )
                        _candidate_push(
                            random_heap,
                            random_capacity,
                            hash_score,
                            serial,
                            record,
                        )
                        serial += 1
                except Exception as error:
                    invalid_games += 1
                    if invalid_games <= 5:
                        print(f"[Cache] Skipping invalid game: {error}")
            if writer.total_samples + len(writer._rows) >= max_samples:
                break

    writer.flush()
    candidates = _candidate_records(top_heap, random_heap)
    print(
        f"[Cache] Built {writer.total_samples} samples from {valid_games} games; "
        f"skipped {invalid_games} invalid games; reanalysis candidates={len(candidates)}."
    )
    return writer.total_samples, valid_games, candidates


def _iter_target_games(files, wanted_game_ids):
    remaining = set(wanted_game_ids)
    if not remaining:
        return
    for replay_path in files:
        with gzip.open(replay_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not remaining:
                    return
                if not line.strip():
                    continue
                game = json.loads(line)
                game_id = str(game.get("id", ""))
                if game_id in remaining:
                    remaining.remove(game_id)
                    yield game


def _rank_candidates_with_current_model(files, candidates, evaluator, limit):
    by_game = {}
    for record in candidates.values():
        by_game.setdefault(record["game_id"], {})[record["ply"]] = record

    ranked = []
    for game in _iter_target_games(files, by_game):
        wanted = by_game[str(game["id"])]
        board = DarkChessBoardPy(bag=game["hid"])
        for ply, move in enumerate(game["mov"]):
            if ply in wanted:
                record = wanted[ply]
                v_new = float(np.clip(evaluator(board), -1.0, 1.0))
                ranked.append(
                    (
                        abs(float(record["q_old"]) - v_new),
                        record,
                        v_new,
                    )
                )
            board.make_move(int(move), validate=True)

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[: max(0, int(limit))]


def _reanalyze_selected(files, ranked, evaluator, depth, node_budget):
    wanted_by_game = {}
    for priority, record, v_new in ranked:
        wanted_by_game.setdefault(record["game_id"], {})[record["ply"]] = (
            priority,
            record,
            v_new,
        )

    overrides = []
    for game in _iter_target_games(files, wanted_by_game):
        wanted = wanted_by_game[str(game["id"])]
        board = DarkChessBoardPy(bag=game["hid"])
        for ply, move in enumerate(game["mov"]):
            if ply in wanted:
                priority, record, v_new = wanted[ply]
                analysis = ChanceSearch(
                    evaluator=evaluator,
                    max_depth=depth,
                    node_budget=node_budget,
                ).analyze(board)
                q_new = float(analysis.value)
                target_new = 0.5 * float(record["result"]) + 0.5 * q_new
                overrides.append(
                    (
                        int(record["sample_index"]),
                        target_new,
                        q_new,
                        float(v_new),
                        float(priority),
                        int(analysis.nodes),
                        int(analysis.depth),
                    )
                )
                print(
                    f"[Reanalysis] sample={record['sample_index']} "
                    f"priority={priority:.4f} q_old={record['q_old']:+.4f} "
                    f"v_new={v_new:+.4f} q_new={q_new:+.4f} "
                    f"nodes={analysis.nodes} depth={analysis.depth}"
                )
            board.make_move(int(move), validate=True)
    overrides.sort(key=lambda row: row[0])
    return overrides


def write_reanalysis_overrides(path, overrides):
    path = Path(path)
    if overrides:
        columns = list(zip(*overrides))
        np.savez_compressed(
            path,
            sample_index=np.asarray(columns[0], dtype=np.int64),
            target=np.asarray(columns[1], dtype=np.float32),
            q=np.asarray(columns[2], dtype=np.float32),
            v_new=np.asarray(columns[3], dtype=np.float32),
            priority=np.asarray(columns[4], dtype=np.float32),
            nodes=np.asarray(columns[5], dtype=np.int32),
            depth=np.asarray(columns[6], dtype=np.int16),
        )
    else:
        np.savez_compressed(
            path,
            sample_index=np.zeros(0, dtype=np.int64),
            target=np.zeros(0, dtype=np.float32),
            q=np.zeros(0, dtype=np.float32),
            v_new=np.zeros(0, dtype=np.float32),
            priority=np.zeros(0, dtype=np.float32),
            nodes=np.zeros(0, dtype=np.int32),
            depth=np.zeros(0, dtype=np.int16),
        )


def main():
    dataset_dir = Path(os.environ.get("DATASET_DIR", "datasets"))
    output_dir = Path(os.environ.get("TRAINING_CACHE_DIR", "training_cache"))
    files = sorted(glob.glob(str(dataset_dir / "**/*.jsonl.gz"), recursive=True))
    if not files:
        raise RuntimeError("No replay files found for training-cache build.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_positions = int(os.environ.get("MAX_POSITIONS_PER_GAME", "4"))
    max_samples = int(os.environ.get("MAX_TRAINING_SAMPLES", "2000000"))
    shard_size = int(os.environ.get("TRAINING_CACHE_SHARD_SIZE", "50000"))
    candidate_capacity = int(os.environ.get("REANALYSIS_CANDIDATES", "4096"))
    reanalysis_limit = int(os.environ.get("REANALYSIS_MAX_POSITIONS", "64"))
    reanalysis_depth = int(os.environ.get("REANALYSIS_DEPTH", "10"))
    reanalysis_node_budget = int(
        os.environ.get("REANALYSIS_NODE_BUDGET", "50000")
    )

    total_samples, valid_games, candidates = build_base_cache(
        files,
        output_dir,
        max_positions,
        max_samples,
        shard_size,
        candidate_capacity,
    )

    overrides = []
    champion_path = Path(
        os.environ.get("CHAMPION_PATH", "models/champion.nnue")
    )
    if champion_path.exists() and candidates and reanalysis_limit > 0:
        torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
        evaluator = ModelEvaluator(load_model_file(champion_path))
        ranked = _rank_candidates_with_current_model(
            files,
            candidates,
            evaluator,
            reanalysis_limit,
        )
        overrides = _reanalyze_selected(
            files,
            ranked,
            evaluator,
            reanalysis_depth,
            reanalysis_node_budget,
        )
    else:
        print("[Reanalysis] Skipped: no champion/candidates or limit is zero.")

    override_path = output_dir / "reanalysis_overrides.npz"
    write_reanalysis_overrides(override_path, overrides)
    manifest = {
        "format_version": 1,
        "input_size": CURRENT_INPUT_SIZE,
        "samples": total_samples,
        "valid_games": valid_games,
        "shards": len(list(output_dir.glob("cache_*.npz"))),
        "reanalysis_overrides": len(overrides),
        "raw_replay_modified": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"[Cache] Manifest: {manifest}")


if __name__ == "__main__":
    main()
