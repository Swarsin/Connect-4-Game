"""Fast evaluation of policy-value networks: batched benchmark and batched arena games."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from c4.agents import make_agent
from c4.alphazero.mcts import Evaluator, SearchParams, new_root, run_search
from c4.engine import Board
from c4.eval.arena import GameRecord, MatchResult
from c4.eval.benchmark import BenchItem, score_predictions
from c4.eval.openings import get_openings


def _argmax(x: Sequence[float], rng: np.random.Generator) -> int:
    a = np.asarray(x, dtype=np.float64)
    return int(rng.choice(np.flatnonzero(a == a.max())))


def net_moves(
    evaluator: Evaluator,
    boards: Sequence[Board],
    sims: int,
    params: SearchParams,
    chunk: int = 512,
    seed: int = 0,
) -> tuple[list[int], list[float]]:
    """Greedy move + value for each board. ``sims == 0`` -> raw policy/value head."""
    rng = np.random.default_rng(seed)
    moves: list[int] = []
    values: list[float] = []
    for i in range(0, len(boards), chunk):
        part = boards[i : i + chunk]
        if sims == 0:
            pri, val = evaluator.evaluate(part)
            moves += [_argmax(p, rng) for p in pri]
            values += [float(v) for v in val]
        else:
            roots = [new_root(b) for b in part]
            run_search(roots, part, evaluator, params, sims=sims)
            moves += [_argmax(r.N, rng) for r in roots]
            values += [r.value() for r in roots]
    return moves, values


def benchmark_net(
    evaluator: Evaluator, items: Sequence[BenchItem], sims: int, params: SearchParams
) -> dict[str, Any]:
    boards = [Board.from_moves(it.moves) for it in items]
    moves, values = net_moves(evaluator, boards, sims, params)
    return score_predictions(items, moves, values)


def arena_net(
    evaluator: Evaluator,
    name: str,
    opponent: str,
    sims: int,
    params: SearchParams,
    openings: str | Sequence[str] = "one",
    seed: int = 0,
    shared_opponent: bool = False,
) -> MatchResult:
    """Play the network (greedy, ``sims`` simulations) against an agent spec, all games in
    lockstep so network evaluations are batched. Each opening is played with both colours.

    ``shared_opponent`` reuses one opponent instance for every game (safe for stateless agents
    such as the solver or Minimax, and much faster than spawning a solver process per game)."""
    ops = get_openings(openings) if isinstance(openings, str) else list(openings)
    rng = np.random.default_rng(seed)
    shared = make_agent(opponent, seed=seed) if shared_opponent else None
    games = []
    for k, (o, net_first) in enumerate((o, f) for o in ops for f in (True, False)):
        opp = shared if shared is not None else make_agent(opponent, seed=seed + k)
        games.append({"b": Board.from_moves(o), "first": net_first, "opp": opp, "opening": o})

    active = [g for g in games if not g["b"].is_terminal()]
    while active:
        net_turn = [g for g in active if (g["b"].player_to_move() == 1) == g["first"]]
        if net_turn:
            boards = [g["b"] for g in net_turn]
            mv, _ = net_moves(evaluator, boards, sims, params, seed=int(rng.integers(1 << 30)))
            for g, m in zip(net_turn, mv, strict=True):
                g["b"].play(m)
        ids = {id(g) for g in net_turn}
        for g in active:
            if id(g) in ids or g["b"].is_terminal():
                continue
            g["b"].play(g["opp"].select_move(g["b"]))
        active = [g for g in active if not g["b"].is_terminal()]

    res = MatchResult(name, opponent)
    for g in games:
        b = g["b"]
        w = b.winner()
        first, second = (name, opponent) if g["first"] else (opponent, name)
        res.games.append(GameRecord(first, second, g["opening"], b.to_moves(), w))
        if w == 0:
            res.draws += 1
        elif (w == 1) == g["first"]:
            res.a_wins += 1
        else:
            res.b_wins += 1
    return res
