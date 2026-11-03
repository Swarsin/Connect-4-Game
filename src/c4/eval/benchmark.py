"""Accuracy against perfect play on solver-labelled benchmark files.

Metrics (see docs/evaluation.md):

* ``optimal``: chosen move is among the columns with the maximum solver score.
* ``preserving``: chosen move keeps the theoretical result (win stays win, draw stays draw).
* ``value_acc``: (agents that report a value) value bucketed to loss/draw/win with thresholds
  ±1/3 matches the sign of the best solver score; ``value_mse`` vs that sign.

All metrics are also broken down by game stage (number of moves already played).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from c4.engine import Board

STAGES = ((0, 10), (11, 20), (21, 30), (31, 42))
STAGE_KEYS = [f"{lo}-{hi}" for lo, hi in STAGES]


@dataclass
class BenchItem:
    moves: str
    scores: list[int | None]

    @property
    def n_moves(self) -> int:
        return len(self.moves)

    @property
    def best(self) -> int:
        return max(s for s in self.scores if s is not None)


def load_benchmark(path: str | Path, limit: int | None = None) -> list[BenchItem]:
    """Load benchmark items. ``limit`` takes an evenly spaced sample across the file (files are
    ordered by game stage), so stage breakdowns stay representative."""
    items = []
    with Path(path).open() as f:
        for line in f:
            d = json.loads(line)
            items.append(BenchItem(d["moves"], d["scores"]))
    if limit and limit < len(items):
        import numpy as np

        idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
        items = [items[i] for i in idx]
    return items


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def stage_of(n: int) -> str:
    for lo, hi in STAGES:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "?"


def score_predictions(
    items: Sequence[BenchItem],
    moves: Sequence[int],
    values: Sequence[float | None] | None = None,
) -> dict[str, Any]:
    """Aggregate metrics given each item's chosen move (and optional value prediction)."""
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i, (it, mv) in enumerate(zip(items, moves, strict=True)):
        best = it.best
        s = it.scores[mv]
        if s is None:
            raise ValueError(f"illegal move {mv} predicted for '{it.moves}'")
        for key in ("all", stage_of(it.n_moves)):
            a = acc[key]
            a["n"] += 1
            a["optimal"] += float(s == best)
            a["preserving"] += float(_sign(s) == _sign(best))
            if values is not None and values[i] is not None:
                v = float(values[i])  # type: ignore[arg-type]
                bucket = 1 if v > 1 / 3 else -1 if v < -1 / 3 else 0
                a["n_value"] += 1
                a["value_acc"] += float(bucket == _sign(best))
                a["value_mse"] += (v - _sign(best)) ** 2

    out: dict[str, Any] = {}
    for key, a in acc.items():
        n = a["n"]
        row = {"n": int(n), "optimal": a["optimal"] / n, "preserving": a["preserving"] / n}
        if a.get("n_value"):
            row["value_acc"] = a["value_acc"] / a["n_value"]
            row["value_mse"] = a["value_mse"] / a["n_value"]
        out[key] = row
    order = ["all"] + [f"{lo}-{hi}" for lo, hi in STAGES]
    return {k: out[k] for k in order if k in out}


def evaluate_fn(
    items: Sequence[BenchItem], fn: Callable[[Board], tuple[int, float | None]]
) -> dict[str, Any]:
    moves, values = [], []
    for it in items:
        m, v = fn(Board.from_moves(it.moves))
        moves.append(m)
        values.append(v)
    return score_predictions(items, moves, values if any(v is not None for v in values) else None)


def evaluate_agent(agent: Any, items: Sequence[BenchItem]) -> dict[str, Any]:
    """Benchmark any :class:`~c4.agents.base.Agent`. Agents that implement
    ``select_with_value(board) -> (move, value)`` also get value metrics."""

    def fn(b: Board) -> tuple[int, float | None]:
        agent.reset(0)
        if hasattr(agent, "select_with_value"):
            return agent.select_with_value(b)
        return agent.select_move(b), None

    return evaluate_fn(items, fn)


def _spec_worker(args: tuple[str, list[str]]) -> list[tuple[int, float | None]]:
    from c4.agents import make_agent

    spec, positions = args
    agent = make_agent(spec, seed=0)
    out = []
    for moves in positions:
        agent.reset(0)
        b = Board.from_moves(moves)
        if hasattr(agent, "select_with_value"):
            out.append(agent.select_with_value(b))
        else:
            out.append((agent.select_move(b), None))
    return out


def evaluate_spec(spec: str, items: Sequence[BenchItem], workers: int = 1) -> dict[str, Any]:
    """Benchmark an agent given by spec string, optionally across worker processes."""
    import multiprocessing as mp

    positions = [it.moves for it in items]
    workers = max(1, min(workers, len(positions)))
    chunks = [positions[i::workers] for i in range(workers)]
    if workers == 1:
        parts = [_spec_worker((spec, chunks[0]))]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_spec_worker, [(spec, ch) for ch in chunks])
    preds: list[tuple[int, float | None]] = [None] * len(positions)  # type: ignore[list-item]
    for w, part in enumerate(parts):
        for j, p in enumerate(part):
            preds[w + j * workers] = p
    moves = [p[0] for p in preds]
    values = [p[1] for p in preds]
    return score_predictions(items, moves, values if any(v is not None for v in values) else None)


def format_table(results: dict[str, dict[str, Any]], metric_keys=("optimal", "preserving")) -> str:
    """Markdown table: one row per result name, columns = metric x stage (aligned across rows)."""
    if not results:
        return ""
    stages = ["all"] + STAGE_KEYS
    cols = [f"{m} {s}" for m in metric_keys for s in stages]
    lines = ["| agent | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, res in results.items():
        cells = [
            f"{100 * res[s][m]:.1f}" if s in res and m in res[s] else "-"
            for m in metric_keys
            for s in stages
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
