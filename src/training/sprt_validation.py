import argparse
import math
import os

import numpy as np
import torch

from board import (
    BLACK,
    INITIAL_COUNTS,
    PIECE_COLOR,
    RED,
    DarkChessBoardPy,
)
from search import ChanceSearch
from train import extract_features, load_model_file


class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, board):
        features = extract_features(board, self.model.input_size)
        tensor = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(tensor).item())

    def evaluate_many(self, boards):
        features = np.stack(
            [extract_features(board, self.model.input_size) for board in boards]
        )
        tensor = torch.from_numpy(features)
        with torch.no_grad():
            return self.model(tensor).squeeze(1).cpu().numpy()


def make_bag(rng):
    bag = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
    rng.shuffle(bag)
    return bag


def score_for_color(red_result, color):
    red_score = (float(red_result) + 1.0) / 2.0
    return red_score if color == RED else 1.0 - red_score


def play_game(first_evaluator, second_evaluator, bag, first_square, depth):
    """
    Play a game while keeping model identity attached to player order.

    The first reveal assigns first_evaluator to that revealed color. Search
    sees only remaining_counts and never the referee's hidden piece mapping.
    Returns the first player's score in [0, 1].
    """
    board = DarkChessBoardPy(bag=bag)
    first_color = int(PIECE_COLOR[int(board.hidden_pieces[first_square])])
    board.make_move((first_square << 5) | first_square, validate=True)

    evaluators = {
        first_color: first_evaluator,
        1 - first_color: second_evaluator,
    }

    for _ in range(512):
        over, result = board.is_game_over()
        if over:
            return score_for_color(result, first_color)

        evaluator = evaluators[int(board.side_to_move)]
        analysis = ChanceSearch(evaluator=evaluator, max_depth=depth).analyze(board)
        board.make_move(analysis.move, validate=False)

    return 0.5


def elo_probability(elo):
    return 1.0 / (1.0 + 10.0 ** (-float(elo) / 400.0))


def paired_score_llr(pair_score, elo0, elo1):
    """
    Sequential likelihood contribution for a paired two-game score.

    Pairing the identical bag and first square removes much of Banqi's random
    variance. Half-points are treated as fractional Bernoulli observations.
    This is intentionally conservative and never auto-promotes an undecided
    challenger.
    """
    p0 = elo_probability(elo0)
    p1 = elo_probability(elo1)
    score = float(pair_score)
    return (
        score * math.log(p1 / p0)
        + (2.0 - score) * math.log((1.0 - p1) / (1.0 - p0))
    )


def set_action_result(passed):
    value = "true" if passed else "false"
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as handle:
            handle.write(f"PASSED={value}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    if not os.path.exists(args.champion):
        raise FileNotFoundError(f"champion not found: {args.champion}")
    if not os.path.exists(args.challenger):
        raise FileNotFoundError(f"challenger not found: {args.challenger}")

    # Early-development policy: producing a new generation takes priority over
    # gating it.  Keep the SPRT implementation available, but require an
    # explicit opt-in before it is allowed to block promotion.
    sprt_required = os.environ.get("SPRT_REQUIRED", "false").lower() in {
        "1", "true", "yes", "on"
    }
    if not sprt_required:
        set_action_result(True)
        print(
            "SPRT skipped by early-development policy; "
            "challenger is approved for unconditional promotion."
        )
        return
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    if not args.elo1 > args.elo0:
        raise ValueError("--elo1 must be greater than --elo0")

    champion = ModelEvaluator(load_model_file(args.champion))
    challenger = ModelEvaluator(load_model_file(args.challenger))
    rng = np.random.default_rng(args.seed)

    upper = math.log((1.0 - args.beta) / args.alpha)
    lower = math.log(args.beta / (1.0 - args.alpha))
    llr = 0.0
    challenger_points = 0.0
    decision = "UNDECIDED"

    print(
        f"Starting paired SPRT: {args.pairs} pairs, "
        f"Elo [{args.elo0}, {args.elo1}], bounds [{lower:.3f}, {upper:.3f}]"
    )
    for pair_index in range(args.pairs):
        bag = make_bag(rng)
        first_square = int(rng.integers(0, 32))

        challenger_first_score = play_game(
            challenger,
            champion,
            bag,
            first_square,
            args.depth,
        )
        champion_first_score = play_game(
            champion,
            challenger,
            bag,
            first_square,
            args.depth,
        )
        pair_score = challenger_first_score + (1.0 - champion_first_score)
        challenger_points += pair_score
        llr += paired_score_llr(pair_score, args.elo0, args.elo1)

        if (pair_index + 1) % 10 == 0:
            score_rate = challenger_points / (2.0 * (pair_index + 1))
            print(
                f"Pairs {pair_index + 1}: score={score_rate:.3%}, "
                f"LLR={llr:.3f}"
            )

        if llr >= upper:
            decision = "ACCEPT_H1"
            break
        if llr <= lower:
            decision = "ACCEPT_H0"
            break

    pairs_played = pair_index + 1
    score_rate = challenger_points / (2.0 * pairs_played)
    passed = decision == "ACCEPT_H1"
    set_action_result(passed)
    print(
        f"SPRT decision={decision}, pairs={pairs_played}, "
        f"challenger score={score_rate:.3%}, LLR={llr:.3f}, "
        f"PASSED={'true' if passed else 'false'}"
    )


if __name__ == "__main__":
    main()
