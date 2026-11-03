"""UCT Monte Carlo Tree Search with random rollouts (fixed version of the legacy bot).

Fixes over the legacy implementation:
* each node stores results from the perspective of the player who made the move into it, so
  the perspective alternates correctly during backpropagation;
* the side to move is derived from the board, never from shared mutable player objects;
* search is limited by iteration count (reproducible) rather than wall-clock time;
* the tree is reused between moves when the new position is a descendant of the old root.
"""

from __future__ import annotations

import math
from typing import Any

from c4.agents.base import BaseAgent
from c4.engine import Board


class _Node:
    __slots__ = ("move", "parent", "children", "untried", "wins", "visits", "just_moved")

    def __init__(self, move: int | None, parent: _Node | None, board: Board) -> None:
        self.move = move
        self.parent = parent
        self.children: list[_Node] = []
        self.untried = board.legal_moves()
        self.wins = 0.0
        self.visits = 0
        self.just_moved = 3 - board.player_to_move()

    def uct_child(self, c: float) -> _Node:
        log_n = math.log(self.visits)
        return max(
            self.children,
            key=lambda ch: ch.wins / ch.visits + c * math.sqrt(log_n / ch.visits),
        )


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        iterations: int = 2000,
        c: float = math.sqrt(2),
        seed: int | None = None,
        smart_rollouts: bool = False,
        reuse_tree: bool = True,
    ) -> None:
        super().__init__(seed)
        self.iterations = iterations
        self.c = c
        self.smart_rollouts = smart_rollouts
        self.reuse_tree = reuse_tree
        self.name = f"mcts:it={iterations}"
        self._root: _Node | None = None
        self._root_moves = ""

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._root, self._root_moves = None, ""

    def _rollout(self, b: Board) -> int:
        rng = self.rng
        while not b.is_terminal():
            moves = b.legal_moves()
            if self.smart_rollouts:
                win = b.winning_moves(True)
                if win:
                    b.play_unchecked(win[0])
                    break
                block = b.winning_moves(False)
                if block:
                    b.play_unchecked(block[0])
                    continue
            b.play_unchecked(rng.choice(moves))
        w = b.winner()
        assert w is not None
        return w

    def _get_root(self, board: Board) -> _Node:
        moves = board.to_moves()
        root = self._root
        if self.reuse_tree and root is not None and moves.startswith(self._root_moves):
            for ch in moves[len(self._root_moves) :]:
                col = int(ch) - 1
                nxt = next((n for n in root.children if n.move == col), None)
                if nxt is None:
                    root = None
                    break
                root = nxt
            if root is not None:
                root.parent = None
                self._root, self._root_moves = root, moves
                return root
        self._root, self._root_moves = _Node(None, None, board), moves
        return self._root

    def search(self, board: Board) -> _Node:
        root = self._get_root(board)
        rng = self.rng
        for _ in range(self.iterations):
            node, b = root, board.copy()
            while not node.untried and node.children:
                node = node.uct_child(self.c)
                b.play_unchecked(node.move)
            if node.untried:
                col = node.untried.pop(rng.randrange(len(node.untried)))
                b.play_unchecked(col)
                child = _Node(col, node, b)
                node.children.append(child)
                node = child
            winner = self._rollout(b)
            while node is not None:
                node.visits += 1
                if winner == 0:
                    node.wins += 0.5
                elif winner == node.just_moved:
                    node.wins += 1.0
                node = node.parent
        return root

    def select_move(self, board: Board) -> int:
        root = self.search(board)
        most = max(ch.visits for ch in root.children)
        return self.rng.choice(sorted(ch.move for ch in root.children if ch.visits == most))

    def analyze(self, board: Board) -> dict[str, Any]:
        root = self.search(board)
        visits = [0] * 7
        q: list[float | None] = [None] * 7
        for ch in root.children:
            visits[ch.move] = ch.visits
            q[ch.move] = ch.wins / ch.visits
        return {"visits": visits, "q": q}
