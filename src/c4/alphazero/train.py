"""AlphaZero training entry point.

    python -m c4.alphazero.train --config configs/smoke.yaml
    python -m c4.alphazero.train --config configs/main.yaml --resume auto
    python -m c4.alphazero.train --config configs/main.yaml --set train.lr=0.01 --set mcts.sims=100

Each iteration: self-play -> add to replay buffer -> train -> (periodically) evaluate -> checkpoint.
Stops cleanly (saving everything) when ``runtime.max_wall_hours`` would be exceeded or on SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from c4.alphazero.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from c4.alphazero.config import Config, load_config
from c4.alphazero.mcts import NetEvaluator
from c4.alphazero.network import C4Net, resolve_device, save_model
from c4.alphazero.replay_buffer import ReplayBuffer
from c4.alphazero.selfplay import SelfPlayPool, play_games, search_params
from c4.alphazero.tracking import Tracker, download_latest_checkpoint
from c4.alphazero.trainer import Trainer

_STOP = False


def _on_signal(signum, frame) -> None:  # pragma: no cover - signal handling
    global _STOP
    print(f"[train] received signal {signum}: will stop after this iteration")
    _STOP = True


def evaluate(net: C4Net, cfg: Config, it: int, device: torch.device) -> dict[str, float]:
    from c4.eval.az_eval import arena_net, benchmark_net
    from c4.eval.benchmark import load_benchmark

    ev = NetEvaluator(net, device, amp=cfg.train.amp)
    params = search_params(cfg.mcts)
    out: dict[str, float] = {}
    for opp in cfg.eval.arena:
        t = time.time()
        res = arena_net(
            ev, f"iter{it}", opp, cfg.eval.arena_sims, params, cfg.eval.arena_openings, seed=it
        )
        out[f"arena/{opp}"] = res.a_score
        print(f"  [eval] {res.summary()} ({time.time() - t:.0f}s)")
    if cfg.eval.benchmark and Path(cfg.eval.benchmark).exists():
        items = load_benchmark(cfg.eval.benchmark, cfg.eval.benchmark_limit)
        for sims in cfg.eval.benchmark_sims:
            t = time.time()
            r = benchmark_net(ev, items, sims, params)["all"]
            for k in ("optimal", "preserving", "value_acc", "value_mse"):
                if k in r:
                    out[f"bench_s{sims}/{k}"] = r[k]
            print(
                f"  [eval] benchmark sims={sims}: optimal {100 * r['optimal']:.1f}% "
                f"preserving {100 * r['preserving']:.1f}% ({time.time() - t:.0f}s)"
            )
    return out


def train(cfg: Config, resume: str = "auto", max_iters: int | None = None) -> Path:
    global _STOP
    _STOP = False
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.runtime.device)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    net = C4Net(cfg.net).to(device)
    trainer = Trainer(net, cfg.train, device)
    buf = ReplayBuffer(cfg.train.buffer_positions)
    start_it = 0
    extra: dict = {"games": 0, "positions": 0, "wall_hours": 0.0, "wandb_id": None}

    if resume == "auto" and not (run_dir / "state.pt").exists():
        download_latest_checkpoint(cfg, run_dir)
    state = None
    src_dir = run_dir if resume in ("auto", "none") else Path(resume)
    if resume != "none":
        state = load_checkpoint(src_dir, device)
    if state is not None:
        net.load_state_dict(state["model"])
        trainer.load_state_dict(state["trainer"])
        restore_rng(state["rng"], np_rng)
        extra.update(state["extra"])
        start_it = state["iteration"] + 1
        if (src_dir / "buffer.npz").exists():
            buf = ReplayBuffer.load(src_dir / "buffer.npz", cfg.train.buffer_positions)
        print(f"[train] resumed from iteration {state['iteration']} (buffer {len(buf)} positions)")
    (run_dir / "config.json").write_text(json.dumps(cfg.model_dump(), indent=1))

    tracker = Tracker(cfg, run_dir, resume_id=extra.get("wandb_id"))
    extra["wandb_id"] = tracker.run_id
    print(f"[train] {cfg.run_name}: {net.n_params() / 1e6:.2f}M params on {device}")

    weights = run_dir / "latest_weights.pt"
    save_model(net, weights)
    sp = cfg.selfplay
    pool = None
    if sp.workers > 1:
        pool = SelfPlayPool(
            sp.workers, cfg.net, str(device), cfg.train.amp, 200_000 if sp.eval_cache else 0
        )

    old_handlers = {s: signal.signal(s, _on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    session_start = time.time()
    iter_times: list[float] = []
    last = start_it - 1
    end_it = (
        cfg.train.max_iters if max_iters is None else min(cfg.train.max_iters, start_it + max_iters)
    )
    try:
        for it in range(start_it, end_it):
            t0 = time.time()
            lr = trainer.set_iteration(it)
            seed = int(np_rng.integers(1 << 31))
            if pool is not None:
                res = pool.generate(
                    str(weights), sp.games_per_iter, sp.parallel_games_per_worker, cfg.mcts, seed
                )
            else:
                ev = NetEvaluator(
                    net, device, amp=cfg.train.amp, cache_size=200_000 if sp.eval_cache else 0
                )
                res = play_games(
                    ev,
                    sp.games_per_iter,
                    sp.parallel_games_per_worker,
                    cfg.mcts,
                    np.random.default_rng(seed),
                )
            t_sp = time.time() - t0
            buf.add(res.planes, res.pi, res.z, res.q)
            extra["games"] += res.n_games
            extra["positions"] += len(res.z)

            metrics: dict[str, float] = {f"selfplay/{k}": v for k, v in res.stats().items()}
            t1 = time.time()
            if len(buf) >= cfg.train.min_buffer:
                steps = max(
                    1, math.ceil(len(res.z) * cfg.train.sample_reuse / cfg.train.batch_size)
                )
                metrics.update(trainer.train_steps(buf, steps, np_rng))
                metrics["train/steps"] = steps
            t_tr = time.time() - t1
            save_model(net, weights)

            if cfg.eval.every_iters and (it + 1) % cfg.eval.every_iters == 0:
                metrics.update(evaluate(net, cfg, it, device))

            dt = time.time() - t0
            iter_times.append(dt)
            extra["wall_hours"] += dt / 3600
            metrics.update({
                "train/lr": lr, "buffer/size": len(buf), "total/games": extra["games"],
                "total/positions": extra["positions"], "total/gpu_hours": extra["wall_hours"],
                "time/iter_s": dt, "time/selfplay_s": t_sp, "time/train_s": t_tr,
                "time/games_per_hour": res.n_games / max(1e-9, t_sp) * 3600,
            })  # fmt: skip
            tracker.log(metrics, it)
            loss = metrics.get("loss/total")
            print(
                f"[it {it:4d}] games {res.n_games} len {metrics['selfplay/mean_length']:.1f} "
                f"p1 {metrics['selfplay/p1_win_rate']:.2f} buf {len(buf)} "
                + (f"loss {loss:.3f} " if loss is not None else "")
                + f"| sp {t_sp:.0f}s tr {t_tr:.0f}s ({metrics['time/games_per_hour']:.0f} games/h)"
            )
            last = it

            elapsed_h = (time.time() - session_start) / 3600
            next_h = elapsed_h + max(iter_times[-3:]) / 3600 * 1.2
            out_of_time = next_h > cfg.runtime.max_wall_hours
            final = out_of_time or _STOP or it == end_it - 1
            if final or (it + 1) % cfg.checkpoint.every_iters == 0:
                save_checkpoint(
                    run_dir, it, net, trainer, buf, np_rng, cfg.model_dump(), extra, cfg.checkpoint
                )
                if final or (cfg.eval.every_iters and (it + 1) % cfg.eval.every_iters == 0):
                    files = [run_dir / "state.pt"] + ([run_dir / "buffer.npz"] if final else [])
                    tracker.upload_checkpoint(files, it)
            if out_of_time:
                print(
                    f"[train] stopping: wall-clock budget ({cfg.runtime.max_wall_hours}h) reached"
                )
                break
            if _STOP:
                break
    finally:
        if pool is not None:
            pool.close()
        for s, h in old_handlers.items():
            signal.signal(s, h)
        tracker.finish()
    print(f"[train] done, last iteration {last}; run dir {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="c4.alphazero.train")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto", help="'auto' (run dir / W&B), 'none', or a run dir")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. mcts.sims=100")
    ap.add_argument("--iters", type=int, default=None, help="max iterations this session")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    cfg = load_config(a.config, a.set)
    train(cfg, a.resume, a.iters)


if __name__ == "__main__":
    main()
