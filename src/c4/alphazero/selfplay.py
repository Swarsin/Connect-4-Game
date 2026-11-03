"""Self-play data generation: many games in lockstep per process, optionally many processes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from c4.alphazero.config import MCTSConfig, NetConfig
from c4.alphazero.mcts import (
    Evaluator,
    NetEvaluator,
    Node,
    SearchParams,
    choose_move,
    new_root,
    run_search,
)
from c4.engine import Board
from c4.engine.encoding import to_planes


def search_params(cfg: MCTSConfig) -> SearchParams:
    return SearchParams(cfg.sims, cfg.c_puct, cfg.fpu, cfg.dirichlet_alpha, cfg.dirichlet_eps)


@dataclass
class SelfPlayResult:
    planes: np.ndarray
    pi: np.ndarray
    z: np.ndarray
    q: np.ndarray
    lengths: list[int] = field(default_factory=list)
    winners: list[int] = field(default_factory=list)
    games: list[str] = field(default_factory=list)

    @property
    def n_games(self) -> int:
        return len(self.lengths)

    def stats(self) -> dict[str, float]:
        n = max(1, self.n_games)
        return {
            "games": self.n_games,
            "positions": len(self.z),
            "mean_length": float(np.mean(self.lengths)) if self.lengths else 0.0,
            "p1_win_rate": sum(w == 1 for w in self.winners) / n,
            "p2_win_rate": sum(w == 2 for w in self.winners) / n,
            "draw_rate": sum(w == 0 for w in self.winners) / n,
        }

    @staticmethod
    def concat(parts: Sequence[SelfPlayResult]) -> SelfPlayResult:
        parts = [p for p in parts if len(p.z)]
        if not parts:
            return SelfPlayResult(
                np.zeros((0, 2, 6, 7), np.uint8), np.zeros((0, 7), np.float32),
                np.zeros(0, np.int8), np.zeros(0, np.float32),
            )  # fmt: skip
        return SelfPlayResult(
            np.concatenate([p.planes for p in parts]),
            np.concatenate([p.pi for p in parts]),
            np.concatenate([p.z for p in parts]),
            np.concatenate([p.q for p in parts]),
            [x for p in parts for x in p.lengths],
            [x for p in parts for x in p.winners],
            [x for p in parts for x in p.games],
        )


class _Game:
    __slots__ = ("board", "root", "planes", "pis", "players", "qs")

    def __init__(self, opening: str = "") -> None:
        self.board = Board.from_moves(opening)
        self.root: Node = new_root(self.board)
        self.planes: list[np.ndarray] = []
        self.pis: list[np.ndarray] = []
        self.players: list[int] = []
        self.qs: list[float] = []


def play_games(
    evaluator: Evaluator,
    n_games: int,
    parallel: int,
    cfg: MCTSConfig,
    rng: np.random.Generator,
) -> SelfPlayResult:
    """Play ``n_games`` self-play games, ``parallel`` at a time, and return training samples."""
    params = search_params(cfg)
    planes, pis, zs, qs = [], [], [], []
    lengths, winners, records = [], [], []
    active: list[_Game] = []
    started = 0

    while started < n_games or active:
        while len(active) < parallel and started < n_games:
            active.append(_Game())
            started += 1
        run_search(
            [g.root for g in active],
            [g.board for g in active],
            evaluator,
            params,
            rng,
            add_noise=True,
        )

        still: list[_Game] = []
        for g in active:
            b = g.board
            g.planes.append(to_planes(b, np.uint8))
            g.pis.append(g.root.visit_dist().astype(np.float32))
            g.players.append(b.player_to_move())
            g.qs.append(g.root.value())
            temp = cfg.temp if b.n_moves < cfg.temp_moves else 0.0
            a = choose_move(g.root, temp, rng)
            b.play(a)
            if b.is_terminal():
                w = b.winner()
                assert w is not None
                z = [0 if w == 0 else (1 if p == w else -1) for p in g.players]
                planes.extend(g.planes)
                pis.extend(g.pis)
                zs.extend(z)
                qs.extend(g.qs)
                lengths.append(b.n_moves)
                winners.append(w)
                records.append(b.to_moves())
                continue
            child = g.root.children[a] if cfg.tree_reuse else None
            g.root = child if child is not None else new_root(b)
            still.append(g)
        active = still

    return SelfPlayResult(
        np.asarray(planes, dtype=np.uint8).reshape(-1, 2, 6, 7),
        np.asarray(pis, dtype=np.float32).reshape(-1, 7),
        np.asarray(zs, dtype=np.int8),
        np.asarray(qs, dtype=np.float32),
        lengths,
        winners,
        records,
    )


# ---------------------------------------------------------------------------- multi-process
_W: dict = {}


def _init_worker(net_cfg: dict, device: str, amp: bool, cache_size: int, threads: int) -> None:
    import torch

    from c4.alphazero.network import C4Net

    torch.set_num_threads(threads)
    net = C4Net(NetConfig(**net_cfg)).to(device).eval()
    _W.update(net=net, device=torch.device(device), amp=amp, cache=cache_size, version=None)


def _worker_play(args: tuple[str, int, int, int, dict]) -> SelfPlayResult:
    import torch

    weights_path, n_games, parallel, seed, mcts_cfg = args
    version = os.path.getmtime(weights_path)
    if _W["version"] != version:
        sd = torch.load(weights_path, map_location=_W["device"], weights_only=False)
        _W["net"].load_state_dict(sd["model"])
        _W["net"].eval()
        _W["version"] = version
    ev = NetEvaluator(_W["net"], _W["device"], amp=_W["amp"], cache_size=_W["cache"])
    return play_games(ev, n_games, parallel, MCTSConfig(**mcts_cfg), np.random.default_rng(seed))


class SelfPlayPool:
    """Persistent pool of worker processes, each holding its own copy of the network."""

    def __init__(
        self, workers: int, net_cfg: NetConfig, device: str, amp: bool = True, cache_size: int = 0
    ) -> None:
        import multiprocessing as mp

        self.workers = workers
        threads = max(1, (os.cpu_count() or 2) // max(1, workers))
        self.ex = ProcessPoolExecutor(
            workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(net_cfg.model_dump(), device, amp, cache_size, min(threads, 2)),
        )

    def generate(
        self, weights_path: str, n_games: int, parallel: int, mcts: MCTSConfig, seed: int
    ) -> SelfPlayResult:
        per = [n_games // self.workers + (i < n_games % self.workers) for i in range(self.workers)]
        futs = [
            self.ex.submit(
                _worker_play, (weights_path, k, parallel, seed + 1000 * i, mcts.model_dump())
            )
            for i, k in enumerate(per)
            if k > 0
        ]
        return SelfPlayResult.concat([f.result() for f in futs])

    def close(self) -> None:
        self.ex.shutdown(wait=True, cancel_futures=True)
