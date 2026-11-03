"""Policy-value residual network (see docs/conventions.md for input/output layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from c4.alphazero.config import NetConfig


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class C4Net(nn.Module):
    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NetConfig()
        ch = self.cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(self.cfg.blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(ch, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * 6 * 7, 7),
        )  # fmt: skip
        self.value_head = nn.Sequential(
            nn.Conv2d(ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
            nn.Linear(6 * 7, self.cfg.value_hidden), nn.ReLU(),
            nn.Linear(self.cfg.value_hidden, 1), nn.Tanh(),
        )  # fmt: skip

    def trunk(self, x: torch.Tensor, return_all: bool = False):
        h = self.stem(x)
        acts = [h]
        for blk in self.blocks:
            h = blk(h)
            acts.append(h)
        return acts if return_all else h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Softmax over legal moves only; ``legal`` is a bool tensor of the same shape."""
    logits = logits.float().masked_fill(~legal, float("-inf"))
    return torch.softmax(logits, dim=-1)


def save_model(net: C4Net, path: str | Path, meta: dict[str, Any] | None = None) -> None:
    """Weights-only model file (used by agents and export)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"model": net.state_dict(), "net_cfg": net.cfg.model_dump(), "meta": meta or {}}, tmp
    )
    tmp.replace(path)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> C4Net:
    """Load either a weights-only model file or a full training checkpoint."""
    d = torch.load(path, map_location=device, weights_only=False)
    net = C4Net(NetConfig(**d["net_cfg"]))
    net.load_state_dict(d["model"])
    net.to(device).eval()
    return net


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
