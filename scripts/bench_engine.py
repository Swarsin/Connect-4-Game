"""Benchmark random-playout throughput of the bitboard engine vs the legacy stack board.

The legacy ``Stack``/``Board`` classes are extracted from ``legacy/game.py`` source (without
importing pygame) so the comparison is like-for-like.
"""

from __future__ import annotations

import ast
import random
import time
from pathlib import Path

from c4.engine import Board

ROOT = Path(__file__).resolve().parents[1]


def load_legacy_board():
    src = (ROOT / "legacy" / "game.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("Stack", "Board")]
    # drop pygame-only method
    for cls in keep:
        cls.body = [
            m for m in cls.body if not (isinstance(m, ast.FunctionDef) and m.name == "DisplayBoard")
        ]
    mod = ast.Module(body=keep, type_ignores=[])
    ns: dict = {
        "EMPTY": 0,
        "random": random,
        "math": __import__("math"),
        "copy": __import__("copy"),
    }
    exec(compile(mod, "legacy_board", "exec"), ns)
    return ns["Board"]


def bench_new(n: int, rng: random.Random) -> float:
    t = time.perf_counter()
    for _ in range(n):
        b = Board()
        while not b.is_terminal():
            b.play_unchecked(rng.choice(b.legal_moves()))
    return n / (time.perf_counter() - t)


def bench_legacy(n: int, rng: random.Random, LegacyBoard) -> float:
    t = time.perf_counter()
    for _ in range(n):
        b = LegacyBoard()
        piece = 1
        while not b.is_terminal_node():
            b.DropPiece(rng.choice(b.GetValidMoves()), piece)
            piece = 3 - piece
    return n / (time.perf_counter() - t)


def bench_copy_new(n: int) -> float:
    b = Board.from_moves("4453")
    t = time.perf_counter()
    for i in range(n):
        c = b.copy()
        c.play(i % 7)
    return n / (time.perf_counter() - t)


def bench_copy_legacy(n: int, LegacyBoard) -> float:
    import copy

    b = LegacyBoard()
    for col, p in ((3, 1), (3, 2), (4, 1), (2, 2)):
        b.DropPiece(col, p)
    t = time.perf_counter()
    for i in range(n):
        c = copy.deepcopy(b)
        c.DropPiece(i % 7, 1)
    return n / (time.perf_counter() - t)


def main() -> None:
    LegacyBoard = load_legacy_board()
    rng = random.Random(0)
    new = bench_new(5000, rng)
    old = bench_legacy(300, rng, LegacyBoard)
    print(
        f"random playouts  bitboard: {new:10.0f} games/s   legacy: {old:8.0f} games/s   ({new / old:.1f}x)"
    )
    new = bench_copy_new(200_000)
    old = bench_copy_legacy(5_000, LegacyBoard)
    print(
        f"copy+play (search node) bitboard: {new:10.0f} ops/s   legacy: {old:8.0f} ops/s   ({new / old:.1f}x)"
    )


if __name__ == "__main__":
    main()
