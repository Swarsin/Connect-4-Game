"""Validated training configuration (YAML -> pydantic). Supports ``extends: other.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NetConfig(_Strict):
    channels: int = 128
    blocks: int = 10
    value_hidden: int = 64


class MCTSConfig(_Strict):
    sims: int = 200
    c_puct: float = 2.0
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25
    temp_moves: int = 12
    temp: float = 1.0
    fpu: float = 0.0
    tree_reuse: bool = True


class SelfPlayConfig(_Strict):
    workers: int = 4
    parallel_games_per_worker: int = 128
    games_per_iter: int = 2048
    eval_cache: bool = True


class LRSchedule(_Strict):
    type: Literal["constant", "step", "cosine"] = "step"
    milestones_iters: list[int] = Field(default_factory=lambda: [60, 120])
    gamma: float = 0.1
    total_iters: int = 200
    warmup_iters: int = 0


class TrainConfig(_Strict):
    batch_size: int = 1024
    optimizer: Literal["sgd", "adamw"] = "sgd"
    lr: float = 0.02
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_schedule: LRSchedule = Field(default_factory=LRSchedule)
    sample_reuse: float = 6.0
    buffer_positions: int = 1_000_000
    min_buffer: int = 50_000
    augment_mirror: bool = True
    value_target: Literal["z", "zq"] = "z"  # zq = (z + root_q) / 2
    amp: bool = True
    grad_clip: float | None = 5.0
    max_iters: int = 1000


class EvalConfig(_Strict):
    every_iters: int = 5
    arena: list[str] = Field(default_factory=lambda: ["minimax:d=4", "mcts:it=2000"])
    arena_openings: str = "one"  # "one" = 7 openings x 2 colours = 14 games; "two" = 98 games
    arena_sims: int = 50
    benchmark: str | None = "data/benchmarks/c4_mixed_dev.jsonl"
    benchmark_sims: list[int] = Field(default_factory=lambda: [0, 200])
    benchmark_limit: int | None = 1000


class CheckpointConfig(_Strict):
    every_iters: int = 1
    keep_last: int = 3
    keep_every: int = 10


class RuntimeConfig(_Strict):
    device: str = "auto"
    max_wall_hours: float = 11.0
    out_dir: str = "runs"


class TrackingConfig(_Strict):
    wandb: bool = True
    project: str = "c4-alphazero"
    entity: str | None = None
    upload_checkpoints: bool = True
    tags: list[str] = Field(default_factory=list)


class Config(_Strict):
    run_name: str = "az-main"
    seed: int = 0
    extends: str | None = None
    net: NetConfig = Field(default_factory=NetConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.runtime.out_dir) / self.run_name


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.get("extends")
    if parent:
        base = _load_raw((path.parent / parent).resolve())
        base.pop("extends", None)
        raw = _deep_merge(base, raw)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config. ``overrides`` are ``dotted.key=value`` strings (YAML-parsed values)."""
    raw = _load_raw(Path(path).resolve())
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return Config.model_validate(raw)
