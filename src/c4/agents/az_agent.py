"""AlphaZero agent: network-guided PUCT search (``sims=0`` plays the raw policy)."""

from __future__ import annotations

from typing import Any

import numpy as np

from c4.agents.base import BaseAgent
from c4.engine import Board


class AZAgent(BaseAgent):
    def __init__(
        self,
        checkpoint: str,
        sims: int = 200,
        temperature: float = 0.0,
        c_puct: float = 2.0,
        add_noise: bool = False,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        from c4.alphazero.mcts import NetEvaluator, SearchParams
        from c4.alphazero.network import load_model, resolve_device

        dev = resolve_device(device)
        self.net = load_model(checkpoint, dev)
        self.evaluator = NetEvaluator(self.net, dev, amp=False)
        self.params = SearchParams(sims=sims, c_puct=c_puct)
        self.sims = sims
        self.temperature = temperature
        self.add_noise = add_noise
        self.np_rng = np.random.default_rng(seed)
        self.name = f"az:{checkpoint},sims={sims}"

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.np_rng = np.random.default_rng(seed)

    def _search(self, board: Board):
        from c4.alphazero.mcts import new_root, run_search

        root = new_root(board)
        run_search([root], [board], self.evaluator, self.params, self.np_rng, self.add_noise)
        return root

    def select_with_value(self, board: Board) -> tuple[int, float]:
        from c4.alphazero.mcts import choose_from_priors, choose_move

        if self.sims == 0:
            pri, val = self.evaluator.evaluate([board])
            return choose_from_priors(pri[0], self.temperature, self.np_rng), float(val[0])
        root = self._search(board)
        return choose_move(root, self.temperature, self.np_rng), root.value()

    def select_move(self, board: Board) -> int:
        return self.select_with_value(board)[0]

    def analyze(self, board: Board) -> dict[str, Any]:
        pri, val = self.evaluator.evaluate([board])
        out: dict[str, Any] = {"priors": pri[0].tolist(), "net_value": float(val[0])}
        if self.sims > 0:
            root = self._search(board)
            out.update(visits=list(root.N), q=[root.q(a) for a in range(7)], value=root.value())
        return out
