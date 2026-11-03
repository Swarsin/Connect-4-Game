"""Left-right mirror symmetry helpers."""

from __future__ import annotations

import numpy as np

from c4.engine.board import WIDTH


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Mirror ``[..., rows, cols]`` planes left-right."""
    return np.ascontiguousarray(planes[..., ::-1])


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    """Mirror a ``[..., 7]`` policy vector left-right."""
    return np.ascontiguousarray(policy[..., ::-1])


def mirror_col(col: int) -> int:
    return WIDTH - 1 - col


def mirror_moves(moves: str) -> str:
    """Mirror a 1-indexed move string, e.g. ``"12"`` -> ``"76"``."""
    return "".join(str(WIDTH + 1 - int(ch)) for ch in moves)
