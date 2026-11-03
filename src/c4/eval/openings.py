"""Fixed opening suite for matches: all 49 two-move openings, each played with both colours."""

from __future__ import annotations

from itertools import product


def two_move_openings() -> list[str]:
    return [f"{a}{b}" for a, b in product(range(1, 8), repeat=2)]


def one_move_openings() -> list[str]:
    return [str(a) for a in range(1, 8)]


def get_openings(name: str = "two") -> list[str]:
    if name in ("two", "2"):
        return two_move_openings()
    if name in ("one", "1"):
        return one_move_openings()
    if name in ("none", "empty", "0"):
        return [""]
    raise ValueError(f"unknown opening set {name!r}")
