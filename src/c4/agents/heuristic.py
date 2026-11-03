"""Hand-crafted evaluation used by the Minimax baseline.

Fixes the legacy one-sided heuristic: every window of four cells is scored for *both* players.
"""

from __future__ import annotations

from dataclasses import dataclass

from c4.engine.board import COLUMN, H1, HEIGHT, WIDTH, Board


def _window_masks() -> list[int]:
    masks = []
    for r in range(HEIGHT):
        for c in range(WIDTH):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= rr < HEIGHT and 0 <= cc < WIDTH for rr, cc in cells):
                    masks.append(sum(1 << (cc * H1 + rr) for rr, cc in cells))
    return masks


WINDOWS = _window_masks()
assert len(WINDOWS) == 69


@dataclass(frozen=True)
class HeuristicWeights:
    two: int = 2
    three: int = 5
    opp_two: int = -2
    opp_three: int = -4
    centre: int = 3


DEFAULT_WEIGHTS = HeuristicWeights()


def evaluate(board: Board, w: HeuristicWeights = DEFAULT_WEIGHTS) -> int:
    """Static score from the perspective of the side to move (positive = good for it)."""
    me = board.current
    opp = board.current ^ board.mask
    score = w.centre * ((me & COLUMN[3]).bit_count() - (opp & COLUMN[3]).bit_count())
    for m in WINDOWS:
        a = (me & m).bit_count()
        b = (opp & m).bit_count()
        if a and b:
            continue
        if a == 3:
            score += w.three
        elif a == 2:
            score += w.two
        elif b == 3:
            score += w.opp_three
        elif b == 2:
            score += w.opp_two
    return score
