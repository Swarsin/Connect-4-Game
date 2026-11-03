from __future__ import annotations

import math
import random

import pytest

from c4.eval.arena import GameRecord, play_match
from c4.eval.benchmark import BenchItem, load_benchmark, score_predictions
from c4.eval.elo import fit_bradley_terry, ratings
from c4.eval.openings import two_move_openings
from c4.solver import optimal_moves, outcome_preserving_moves, solver_available


def synthetic_games(true_elo: dict[str, float], n: int, seed: int = 0) -> list[GameRecord]:
    rng = random.Random(seed)
    names = list(true_elo)
    games = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            p = 1 / (1 + 10 ** ((true_elo[b] - true_elo[a]) / 400))
            for k in range(n):
                w = 1 if rng.random() < p else 2
                if k % 2:
                    games.append(GameRecord(b, a, "", "", 3 - w))
                else:
                    games.append(GameRecord(a, b, "", "", w))
    return games


def test_bradley_terry_recovers_ratings() -> None:
    true = {"random": 0.0, "a": 200.0, "b": 400.0, "c": 700.0}
    games = synthetic_games(true, 600)
    rs = {r.name: r for r in ratings(games, bootstrap=50, prior=0.5)}
    for name, e in true.items():
        assert rs[name].lo - 40 <= e <= rs[name].hi + 40, (name, rs[name])
    assert rs["random"].elo == pytest.approx(0.0)


def test_bradley_terry_finite_with_perfect_record() -> None:
    games = [GameRecord("strong", "random", "", "", 1) for _ in range(20)]
    elo = fit_bradley_terry(games)
    assert math.isfinite(elo["strong"]) and elo["strong"] > 300


def test_openings_and_match_counts() -> None:
    ops = two_move_openings()
    assert len(ops) == 49 and len(set(ops)) == 49
    res = play_match("random", "random", ["", "4"], seed=1)
    assert res.n == 4
    assert {g.first for g in res.games} == {"random"}


def test_score_predictions() -> None:
    items = [BenchItem("", [-2, -1, 0, 1, 0, -1, -2]), BenchItem("44", [-3, -3, -2, 1, -2, -3, -3])]
    r = score_predictions(items, [3, 2], [0.9, -0.9])
    assert r["all"]["optimal"] == 0.5
    assert r["all"]["preserving"] == 0.5
    assert r["all"]["value_acc"] == 0.5
    assert optimal_moves(items[0].scores) == [3]
    assert outcome_preserving_moves([-1, 0, 1, None, 1, 0, -1]) == [2, 4]


def test_benchmark_files_are_consistent() -> None:
    from c4.engine import Board

    for name in ("c4_mixed_dev", "pons_test_l3_r1"):
        items = load_benchmark(f"data/benchmarks/{name}.jsonl", limit=200)
        for it in items:
            b = Board.from_moves(it.moves)
            assert not b.is_terminal()
            legal = set(b.legal_moves())
            assert {c for c, s in enumerate(it.scores) if s is not None} == legal
            # an immediate win must be the best (fastest) score
            for c in b.winning_moves(True):
                assert it.scores[c] == it.best


@pytest.mark.solver
@pytest.mark.skipif(not solver_available(), reason="solver not built")
def test_solver_known_values() -> None:
    from c4.solver import Solver

    with Solver() as s:
        assert s.analyze("") == [-2, -1, 0, 1, 0, -1, -2]
        sc = s.analyze("444444")
        assert sc[3] is None
        assert s.score("44") == 1
