"""Measure self-play throughput (games/hour, simulations/s, network batch size) for a config.

python scripts/bench_selfplay.py --config configs/main.yaml --games 256
python scripts/bench_selfplay.py --config configs/main.yaml --set selfplay.workers=1
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from c4.alphazero.config import load_config
from c4.alphazero.mcts import NetEvaluator
from c4.alphazero.network import C4Net, resolve_device, save_model
from c4.alphazero.selfplay import SelfPlayPool, play_games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--profile", action="store_true", help="cProfile a single-process run")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = resolve_device(cfg.runtime.device)
    net = C4Net(cfg.net).to(device).eval()
    sp = cfg.selfplay
    print(f"net {cfg.net.channels}x{cfg.net.blocks} on {device}; sims {cfg.mcts.sims}; "
          f"workers {sp.workers} x {sp.parallel_games_per_worker} parallel games")  # fmt: skip

    if sp.workers <= 1 or a.profile:
        ev = NetEvaluator(
            net, device, amp=cfg.train.amp, cache_size=200_000 if sp.eval_cache else 0
        )
        t = time.time()
        if a.profile:
            import cProfile
            import pstats

            pr = cProfile.Profile()
            pr.enable()
        res = play_games(
            ev, a.games, sp.parallel_games_per_worker, cfg.mcts, np.random.default_rng(0)
        )
        dt = time.time() - t
        if a.profile:
            pr.disable()
            pstats.Stats(pr).sort_stats("cumulative").print_stats(20)
        print(f"network calls {ev.calls}, mean batch {ev.positions / max(1, ev.calls):.1f}")
    else:
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.pt"
            save_model(net, w)
            pool = SelfPlayPool(
                sp.workers, cfg.net, str(device), cfg.train.amp, 200_000 if sp.eval_cache else 0
            )
            pool.generate(str(w), sp.workers * 2, 2, cfg.mcts, 0)  # warm-up (CUDA init)
            t = time.time()
            res = pool.generate(str(w), a.games, sp.parallel_games_per_worker, cfg.mcts, 1)
            dt = time.time() - t
            pool.close()
    moves = sum(res.lengths)
    print(f"{res.n_games} games, {moves} moves in {dt:.1f}s")
    print(f"games/hour: {res.n_games / dt * 3600:,.0f}")
    print(f"simulations/s: {moves * cfg.mcts.sims / dt:,.0f}")
    print(res.stats())


if __name__ == "__main__":
    main()
