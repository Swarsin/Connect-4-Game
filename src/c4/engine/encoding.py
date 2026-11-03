"""Neural-network input encoding (see docs/conventions.md, item 3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from c4.engine.board import H1, HEIGHT, WIDTH, Board

# _IDX[row, col] = bit index of that cell
_IDX = np.array([[c * H1 + r for c in range(WIDTH)] for r in range(HEIGHT)], dtype=np.int64)


def _bits(x: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(x.to_bytes(8, "little"), dtype=np.uint8), bitorder="little")


def to_planes(board: Board, dtype=np.float32) -> np.ndarray:
    """Return a ``[2, 6, 7]`` array: channel 0 = side to move, channel 1 = opponent."""
    me = _bits(board.current)[_IDX]
    opp = _bits(board.current ^ board.mask)[_IDX]
    return np.stack([me, opp]).astype(dtype, copy=False)


def to_planes_batch(boards: Sequence[Board], dtype=np.float32) -> np.ndarray:
    out = np.empty((len(boards), 2, HEIGHT, WIDTH), dtype=dtype)
    for i, b in enumerate(boards):
        out[i, 0] = _bits(b.current)[_IDX]
        out[i, 1] = _bits(b.current ^ b.mask)[_IDX]
    return out
