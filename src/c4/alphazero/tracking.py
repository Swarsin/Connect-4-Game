"""Experiment tracking: Weights & Biases when available, always a local metrics.jsonl."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from c4.alphazero.config import Config


class Tracker:
    def __init__(self, cfg: Config, run_dir: Path, resume_id: str | None = None) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.metrics_file = run_dir / "metrics.jsonl"
        self.wandb = None
        self.run_id: str | None = resume_id
        if cfg.tracking.wandb and (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE")):
            try:
                import wandb

                run = wandb.init(
                    project=cfg.tracking.project,
                    entity=cfg.tracking.entity,
                    name=cfg.run_name,
                    id=resume_id,
                    resume="allow",
                    config=cfg.model_dump(),
                    tags=cfg.tracking.tags,
                    dir=str(run_dir),
                )
                self.wandb = wandb
                self.run_id = run.id
            except Exception as e:  # never let tracking kill a training run
                print(f"[tracking] W&B disabled: {e}")
        elif cfg.tracking.wandb:
            print("[tracking] WANDB_API_KEY not set: logging to metrics.jsonl only")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_file.open("a") as f:
            f.write(json.dumps({"iteration": step, **metrics}) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def upload_checkpoint(self, files: list[Path], iteration: int) -> None:
        if self.wandb is None or not self.cfg.tracking.upload_checkpoints:
            return
        try:
            art = self.wandb.Artifact(
                f"{self.cfg.run_name}-ckpt", type="checkpoint", metadata={"iteration": iteration}
            )
            for f in files:
                if f.exists():
                    art.add_file(str(f))
            self.wandb.log_artifact(art, aliases=["latest", f"iter{iteration}"])
        except Exception as e:
            print(f"[tracking] artifact upload failed: {e}")

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def download_latest_checkpoint(cfg: Config, run_dir: Path) -> bool:
    """Fetch ``<run>-ckpt:latest`` from W&B into ``run_dir`` (used when resuming on Kaggle)."""
    if not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb

        api = wandb.Api()
        entity = cfg.tracking.entity or api.default_entity
        art = api.artifact(f"{entity}/{cfg.tracking.project}/{cfg.run_name}-ckpt:latest")
        art.download(root=str(run_dir))
        print(f"[tracking] downloaded {art.name} -> {run_dir}")
        return True
    except Exception as e:
        print(f"[tracking] no checkpoint artifact to resume from: {e}")
        return False
