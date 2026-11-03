from __future__ import annotations

from c4.agents.base import BaseAgent
from c4.engine import Board


class RandomAgent(BaseAgent):
    """Uniformly random legal moves."""

    name = "random"

    def select_move(self, board: Board) -> int:
        return self.rng.choice(board.legal_moves())
