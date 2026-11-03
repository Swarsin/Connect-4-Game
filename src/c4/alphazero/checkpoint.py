"""Atomic, resumable checkpoints.

Layout of a run directory::

    runs/<name>/
      state.pt            full training state (model, optimiser, scaler, RNGs, counters, config)
      buffer.npz          replay buffer
      latest_weights.pt   weights read by self-play workers
      models/model_iter_0005.pt   weights-only snapshots (keep_last + every keep_every-th)
      metrics.jsonl       one line per iteration
"""

from __future__ import annotations

import contextlib
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from c4.alphazero.config import CheckpointConfig
from c4.alphazero.network import C4Net, save_model
from c4.alphazero.replay_buffer import ReplayBuffer
from c4.alphazero.trainer import Trainer


def rng_state(np_rng: np.random.Generator) -> dict[str, Any]:
    st = {
        "python": random.getstate(),
        "numpy": np_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict[str, Any], np_rng: np.random.Generator) -> None:
    random.setstate(st["python"])
    np_rng.bit_generator.state = st["numpy"]
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            torch.cuda.set_rng_state_all(st["cuda"])


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def model_path(run_dir: Path, it: int) -> Path:
    return run_dir / "models" / f"model_iter_{it:04d}.pt"


def save_checkpoint(
    run_dir: Path,
    iteration: int,
    net: C4Net,
    trainer: Trainer,
    buf: ReplayBuffer,
    np_rng: np.random.Generator,
    cfg_dict: dict,
    extra: dict[str, Any],
    ck: CheckpointConfig,
    save_buffer: bool = True,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "iteration": iteration,
        **{k: v for k, v in extra.items() if k in ("games", "positions")},
    }
    mp = model_path(run_dir, iteration)
    save_model(net, mp, meta)
    if save_buffer:
        buf.save(run_dir / "buffer.npz")
    state = {
        "iteration": iteration,
        "model": net.state_dict(),
        "net_cfg": net.cfg.model_dump(),
        "trainer": trainer.state_dict(),
        "rng": rng_state(np_rng),
        "config": cfg_dict,
        "extra": extra,
    }
    _atomic_save(state, run_dir / "state.pt")
    prune_models(run_dir, ck)
    return mp


def prune_models(run_dir: Path, ck: CheckpointConfig) -> None:
    files = sorted((run_dir / "models").glob("model_iter_*.pt"))
    its = [int(re.findall(r"(\d+)", f.stem)[-1]) for f in files]
    keep = set(its[-ck.keep_last :]) | {i for i in its if ck.keep_every and i % ck.keep_every == 0}
    for f, i in zip(files, its, strict=True):
        if i not in keep:
            f.unlink(missing_ok=True)


def load_checkpoint(run_dir: Path, device: torch.device) -> dict[str, Any] | None:
    p = run_dir / "state.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location=device, weights_only=False)
