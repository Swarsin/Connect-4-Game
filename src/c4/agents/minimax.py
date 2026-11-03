"""Negamax with alpha-beta pruning, transposition table and centre-first move ordering."""

from __future__ import annotations

import math
from typing import Any

from c4.agents.base import BaseAgent
from c4.agents.heuristic import DEFAULT_WEIGHTS, HeuristicWeights, evaluate
from c4.engine import CENTRE_ORDER, N_CELLS, Board

WIN = 1_000_000
EXACT, LOWER, UPPER = 0, 1, 2


class MinimaxAgent(BaseAgent):
    def __init__(
        self,
        depth: int = 4,
        seed: int | None = None,
        use_tt: bool = True,
        weights: HeuristicWeights = DEFAULT_WEIGHTS,
    ) -> None:
        super().__init__(seed)
        self.depth = depth
        self.use_tt = use_tt
        self.weights = weights
        self.name = f"minimax:d={depth}"
        self.nodes = 0
        self._tt: dict[int, tuple[int, int, int, int]] = {}

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._tt.clear()

    # ------------------------------------------------------------------ search
    def _negamax(self, b: Board, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if b.last_move_won():
            return -(WIN + depth)  # side to move has lost; sooner losses are worse
        if b.n_moves == N_CELLS:
            return 0
        if depth == 0:
            return evaluate(b, self.weights)
        if b.winning_moves(True):
            return WIN + depth - 1

        alpha0 = alpha
        key = b.key()
        tt_move = -1
        if self.use_tt and key in self._tt:
            d, val, flag, tt_move = self._tt[key]
            if d >= depth:
                if flag == EXACT:
                    return val
                if flag == LOWER:
                    alpha = max(alpha, val)
                elif flag == UPPER:
                    beta = min(beta, val)
                if alpha >= beta:
                    return val

        moves = [c for c in CENTRE_ORDER if not b.mask & (1 << (c * 7 + 5))]
        if tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)

        best, best_move = -math.inf, moves[0]
        for c in moves:
            b.play_unchecked(c)
            v = -self._negamax(b, depth - 1, -beta, -alpha)
            b.undo()
            if v > best:
                best, best_move = v, c
            alpha = max(alpha, v)
            if alpha >= beta:
                break

        if self.use_tt:
            flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
            self._tt[key] = (depth, int(best), flag, best_move)
        return best

    def move_scores(self, board: Board) -> dict[int, float]:
        """Scores of every legal move. Moves that cannot tie the best get an upper bound only."""
        b = board.copy()
        scores: dict[int, float] = {}
        best = -math.inf
        moves = [c for c in CENTRE_ORDER if b.is_legal(c)]
        for c in moves:
            b.play_unchecked(c)
            v = -self._negamax(b, self.depth - 1, -math.inf, -(best - 1))
            b.undo()
            scores[c] = v
            best = max(best, v)
        return scores

    def select_move(self, board: Board) -> int:
        if len(self._tt) > 2_000_000:
            self._tt.clear()
        scores = self.move_scores(board)
        best = max(scores.values())
        return self.rng.choice(sorted(c for c, v in scores.items() if v == best))

    def analyze(self, board: Board) -> dict[str, Any]:
        return {"scores": self.move_scores(board)}
