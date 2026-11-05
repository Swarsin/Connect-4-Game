"""PUCT Monte Carlo Tree Search guided by a policy-value network.

Searches run over *many trees in lockstep*: each simulation step descends every tree to one leaf,
evaluates all those leaves in a single batched network call, then expands and backs up. This is
used for self-play (many games), benchmarking (many positions) and single-position play
(a batch of one).

Conventions (docs/conventions.md): every value is from the perspective of the side to move at the
node it belongs to, and is negated when backed up to the parent. ``W[a]``/``N[a]`` on a node are
the statistics of taking move ``a`` *for the side to move at that node*.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from c4.engine import N_CELLS, TOP, WIDTH, Board


class Evaluator(Protocol):
    def evaluate(self, boards: Sequence[Board]) -> tuple[np.ndarray, np.ndarray]:
        """Return (priors ``[n, 7]`` masked to legal moves and summing to 1, values ``[n]``)."""
        ...


@dataclass
class SearchParams:
    sims: int = 200
    c_puct: float = 2.0
    fpu: float = 0.0
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25


class Node:
    __slots__ = ("P", "N", "W", "children", "legal", "expanded", "terminal", "tv", "n")

    def __init__(self) -> None:
        self.P: list[float] = [0.0] * WIDTH
        self.N: list[int] = [0] * WIDTH
        self.W: list[float] = [0.0] * WIDTH
        self.children: list[Node | None] = [None] * WIDTH
        self.legal: list[int] = []
        self.expanded = False
        self.terminal = False
        self.tv = 0.0  # terminal value for the side to move
        self.n = 0  # total visits through this node's edges

    def q(self, a: int) -> float:
        return self.W[a] / self.N[a] if self.N[a] else 0.0

    def value(self) -> float:
        """Mean backed-up value from this node's side-to-move perspective."""
        return sum(self.W) / self.n if self.n else 0.0

    def visit_dist(self) -> np.ndarray:
        v = np.asarray(self.N, dtype=np.float64)
        s = v.sum()
        return v / s if s else v


def _legal(board: Board) -> list[int]:
    m = board.mask
    return [c for c in range(WIDTH) if not m & TOP[c]]


def _select(node: Node, c_puct: float, fpu: float) -> int:
    sq = c_puct * math.sqrt(node.n + 1)
    best_a, best_s = -1, -math.inf
    P, N, W = node.P, node.N, node.W
    for a in node.legal:
        n = N[a]
        q = W[a] / n if n else fpu
        s = q + sq * P[a] / (1 + n)
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def _expand(node: Node, priors: np.ndarray) -> None:
    p = priors.tolist()
    node.P = p
    node.expanded = True


def add_dirichlet(node: Node, alpha: float, eps: float, rng: np.random.Generator) -> None:
    if not node.legal or eps <= 0:
        return
    noise = rng.dirichlet([alpha] * len(node.legal))
    for a, nz in zip(node.legal, noise, strict=True):
        node.P[a] = (1 - eps) * node.P[a] + eps * float(nz)


def new_root(board: Board) -> Node:
    node = Node()
    if board.is_terminal():
        node.terminal = True
        node.tv = -1.0 if board.last_move_won() else 0.0
    else:
        node.legal = _legal(board)
    return node


def run_search(
    roots: Sequence[Node],
    boards: Sequence[Board],
    evaluator: Evaluator,
    params: SearchParams,
    rng: np.random.Generator | None = None,
    add_noise: bool = False,
    sims: int | None = None,
) -> None:
    """Run ``sims`` simulations on every (root, board) pair in place."""
    sims = params.sims if sims is None else sims
    c_puct, fpu = params.c_puct, params.fpu

    # expand unexpanded roots first (one batched call)
    todo = [i for i, r in enumerate(roots) if not r.expanded and not r.terminal]
    if todo:
        pri, _ = evaluator.evaluate([boards[i] for i in todo])
        for k, i in enumerate(todo):
            _expand(roots[i], pri[k])
    if add_noise:
        assert rng is not None
        for r in roots:
            if not r.terminal:
                add_dirichlet(r, params.dirichlet_alpha, params.dirichlet_eps, rng)

    for _ in range(sims):
        leaves: list[tuple[list[tuple[Node, int]], Node, Board]] = []
        for root, rboard in zip(roots, boards, strict=True):
            if root.terminal:
                continue
            node, b = root, rboard.copy()
            path: list[tuple[Node, int]] = []
            while True:
                a = _select(node, c_puct, fpu)
                path.append((node, a))
                b.play_unchecked(a)
                child = node.children[a]
                if child is None:
                    child = Node()
                    node.children[a] = child
                    if b.last_move_won():
                        child.terminal, child.tv = True, -1.0
                    elif b.n_moves == N_CELLS:
                        child.terminal, child.tv = True, 0.0
                    else:
                        child.legal = _legal(b)
                node = child
                if node.terminal or not node.expanded:
                    break
            if node.terminal:
                _backup(path, node.tv)
            else:
                leaves.append((path, node, b))

        if leaves:
            pri, val = evaluator.evaluate([lf[2] for lf in leaves])
            for k, (path, node, _) in enumerate(leaves):
                if not node.expanded:  # same leaf can't appear twice: one leaf per tree
                    _expand(node, pri[k])
                _backup(path, float(val[k]))


def _backup(path: list[tuple[Node, int]], v: float) -> None:
    for node, a in reversed(path):
        v = -v
        node.N[a] += 1
        node.W[a] += v
        node.n += 1


def choose_move(root: Node, temperature: float, rng: np.random.Generator) -> int:
    """Pick a move from root visit counts. ``temperature == 0`` -> argmax (random tie-break)."""
    counts = np.asarray(root.N, dtype=np.float64)
    if counts.sum() == 0:  # no search: fall back to priors
        counts = np.asarray(root.P, dtype=np.float64)
    if counts.sum() == 0:  # terminal root or no information at all
        raise ValueError("cannot choose a move: no search information for this position")
    if temperature <= 1e-6:
        best = np.flatnonzero(counts == np.max(counts))
        return int(rng.choice(best))
    p = counts ** (1.0 / temperature)
    p /= p.sum()
    return int(rng.choice(WIDTH, p=p))


def choose_from_priors(priors: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-6:
        best = np.flatnonzero(priors == priors.max())
        return int(rng.choice(best))
    p = np.asarray(priors, dtype=np.float64) ** (1.0 / temperature)
    if p.sum() == 0:
        raise ValueError("cannot choose a move: no search information for this position")
    p /= p.sum()
    return int(rng.choice(WIDTH, p=p))


# ---------------------------------------------------------------------------- evaluators
class UniformEvaluator:
    """Uniform priors over legal moves and value 0 (for tests: MCTS with no knowledge)."""

    def evaluate(self, boards: Sequence[Board]) -> tuple[np.ndarray, np.ndarray]:
        pri = np.zeros((len(boards), WIDTH), dtype=np.float32)
        for i, b in enumerate(boards):
            legal = _legal(b)
            pri[i, legal] = 1.0 / len(legal)
        return pri, np.zeros(len(boards), dtype=np.float32)


class NetEvaluator:
    """Batched network evaluation with an optional LRU cache keyed by position.

    With ``fast=True`` the network is snapshotted into a BN-folded (fp16 + CUDA graphs on GPU)
    inference copy, so create a new evaluator after the weights change.
    """

    def __init__(
        self, net, device=None, amp: bool = True, cache_size: int = 0, fast: bool = True
    ) -> None:
        import torch

        self.torch = torch
        self.net = net.eval()
        self.device = device or next(net.parameters()).device
        self.amp = amp and self.device.type == "cuda"
        self.fast = None
        if fast:
            from c4.alphazero.inference import FastInference

            self.fast = FastInference(net, self.device, half=self.amp)
        self.cache_size = cache_size
        self.cache: OrderedDict[int, tuple[np.ndarray, float]] = OrderedDict()
        self.calls = 0
        self.positions = 0

    def clear_cache(self) -> None:
        self.cache.clear()

    def _forward(self, boards: Sequence[Board]) -> tuple[np.ndarray, np.ndarray]:
        from c4.engine.encoding import to_planes_batch

        torch = self.torch
        x = torch.from_numpy(to_planes_batch(boards))
        if self.fast is not None:
            out = self.fast(x).numpy()
        else:
            with torch.inference_mode(), torch.autocast(self.device.type, enabled=self.amp):
                logits, v = self.net(x.to(self.device))
            out = torch.cat([logits.float(), v.float()], 1).cpu().numpy()
        logits = out[:, :WIDTH].astype(np.float64)
        for i, b in enumerate(boards):
            m = b.mask
            for c in range(WIDTH):
                if m & TOP[c]:
                    logits[i, c] = -np.inf
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        self.calls += 1
        self.positions += len(boards)
        return p.astype(np.float32), out[:, WIDTH].astype(np.float32)

    def evaluate(self, boards: Sequence[Board]) -> tuple[np.ndarray, np.ndarray]:
        if not self.cache_size:
            return self._forward(boards)
        n = len(boards)
        pri = np.empty((n, WIDTH), dtype=np.float32)
        val = np.empty(n, dtype=np.float32)
        miss: list[int] = []
        keys = [b.key() for b in boards]
        for i, k in enumerate(keys):
            hit = self.cache.get(k)
            if hit is None:
                miss.append(i)
            else:
                pri[i], val[i] = hit
        if miss:
            p, v = self._forward([boards[i] for i in miss])
            for j, i in enumerate(miss):
                pri[i], val[i] = p[j], v[j]
                self.cache[keys[i]] = (p[j], float(v[j]))
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return pri, val
