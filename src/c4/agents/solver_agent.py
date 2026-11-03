"""Perfect player backed by the Pons solver."""

from __future__ import annotations

from typing import Any

from c4.agents.base import BaseAgent
from c4.engine import Board
from c4.solver import Solver, optimal_moves, outcome_preserving_moves


class SolverAgent(BaseAgent):
    """``mode='strong'``: random among score-maximising moves (fastest win / slowest loss).
    ``mode='weak'``: random among moves that preserve the theoretical outcome."""

    def __init__(self, mode: str = "strong", seed: int | None = None) -> None:
        super().__init__(seed)
        self.mode = mode
        self.name = "solver" if mode == "strong" else f"solver:mode={mode}"
        self._solver: Solver | None = None

    @property
    def solver(self) -> Solver:
        if self._solver is None:
            self._solver = Solver()
        return self._solver

    def select_move(self, board: Board) -> int:
        scores = self.solver.analyze(board.to_moves())
        moves = optimal_moves(scores) if self.mode == "strong" else outcome_preserving_moves(scores)
        return self.rng.choice(moves)

    def analyze(self, board: Board) -> dict[str, Any]:
        return {"scores": self.solver.analyze(board.to_moves())}
