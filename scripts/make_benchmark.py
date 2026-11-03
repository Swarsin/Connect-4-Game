"""Build the "mixed" benchmark: solver-labelled positions sampled from realistic games.

Games are played between a varied pool of agents (random, Minimax depth 1-6, UCT MCTS) with some
random moves mixed in. Positions are sampled evenly across move numbers 0-40, de-duplicated
(mirror images count as the same position) and split *by game* into dev and test sets.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from collections import defaultdict
from pathlib import Path

from c4.agents import make_agent
from c4.engine import Board
from c4.solver import analyze_many

POOL = [
    "random",
    "minimax:d=1",
    "minimax:d=2",
    "minimax:d=3",
    "minimax:d=4",
    "minimax:d=6",
    "mcts:it=200",
    "mcts:it=1000",
]


def _games(args: tuple[int, int, float]) -> list[str]:
    n, seed, eps = args
    rng = random.Random(seed)
    agents = {s: make_agent(s, seed=seed) for s in POOL}
    out = []
    for _ in range(n):
        a, b = rng.choice(POOL), rng.choice(POOL)
        for ag in (agents[a], agents[b]):
            ag.reset(rng.randrange(1 << 30))
        board = Board()
        while not board.is_terminal():
            ag = agents[a] if board.player_to_move() == 1 else agents[b]
            col = rng.choice(board.legal_moves()) if rng.random() < eps else ag.select_move(board)
            board.play(col)
        out.append(board.to_moves())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--per-ply", type=int, default=125)
    ap.add_argument("--max-ply", type=int, default=40)
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default="data/benchmarks")
    ap.add_argument("--name", default="c4_mixed")
    args = ap.parse_args()

    t = time.time()
    per = args.games // args.workers + 1
    with mp.get_context("spawn").Pool(args.workers) as pool:
        games = [
            g
            for gs in pool.map(
                _games, [(per, args.seed + i, args.eps) for i in range(args.workers)]
            )
            for g in gs
        ][: args.games]
    print(f"played {len(games)} games in {time.time() - t:.0f}s")

    rng = random.Random(args.seed)
    split = {i: ("dev" if rng.random() < args.dev_frac else "test") for i in range(len(games))}
    by_ply: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for gi, g in enumerate(games):
        for ply in range(min(len(g), args.max_ply + 1)):
            by_ply[ply].append((gi, g[:ply]))

    chosen: list[tuple[str, str]] = []  # (split, moves)
    seen: set[int] = set()
    for ply in sorted(by_ply):
        cands = by_ply[ply]
        rng.shuffle(cands)
        taken = 0
        for gi, moves in cands:
            if taken >= args.per_ply:
                break
            b = Board.from_moves(moves)
            if b.is_terminal():
                continue
            k = b.mirror_key()
            if k in seen:
                continue
            seen.add(k)
            chosen.append((split[gi], moves))
            taken += 1

    t = time.time()
    scores = analyze_many([m for _, m in chosen], workers=args.workers)
    print(f"labelled {len(chosen)} positions in {time.time() - t:.0f}s")

    out_dir = Path(args.out_dir)
    files = {s: (out_dir / f"{args.name}_{s}.jsonl").open("w") for s in ("dev", "test")}
    for (sp, moves), sc in zip(chosen, scores, strict=True):
        best = max(s for s in sc if s is not None)
        files[sp].write(
            json.dumps({"moves": moves, "n_moves": len(moves), "score": best, "scores": sc}) + "\n"
        )
    for f in files.values():
        f.close()
    counts = {s: sum(1 for sp, _ in chosen if sp == s) for s in ("dev", "test")}
    print(f"wrote {counts}")


if __name__ == "__main__":
    main()
