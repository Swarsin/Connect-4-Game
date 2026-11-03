"""Network optimisation: loss, optimiser, LR schedule, mixed precision."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from c4.alphazero.config import TrainConfig
from c4.alphazero.network import C4Net
from c4.alphazero.replay_buffer import ReplayBuffer


def lr_at(cfg: TrainConfig, it: int) -> float:
    s = cfg.lr_schedule
    base = cfg.lr
    if s.warmup_iters and it < s.warmup_iters:
        return base * (it + 1) / s.warmup_iters
    if s.type == "constant":
        return base
    if s.type == "step":
        return base * s.gamma ** sum(it >= m for m in s.milestones_iters)
    # cosine
    t = min(1.0, it / max(1, s.total_iters))
    return base * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))


def loss_fn(
    net: C4Net, x: torch.Tensor, pi: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, v = net(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    policy_loss = -(pi * logp).sum(-1).mean()
    value_loss = F.mse_loss(v.float().squeeze(-1), z)
    with torch.no_grad():
        entropy = -(logp.exp() * logp.nan_to_num(0.0)).sum(-1).mean()
        target_entropy = -(pi * torch.log(pi.clamp_min(1e-12))).sum(-1).mean()
    return policy_loss + value_loss, {
        "loss/policy": policy_loss.detach(),
        "loss/value": value_loss.detach(),
        "policy/entropy": entropy,
        "policy/target_entropy": target_entropy,
        "policy/kl": policy_loss.detach() - target_entropy,
    }


class Trainer:
    def __init__(self, net: C4Net, cfg: TrainConfig, device: torch.device) -> None:
        self.net, self.cfg, self.device = net, cfg, device
        if cfg.optimizer == "sgd":
            self.opt: torch.optim.Optimizer = torch.optim.SGD(
                net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                nesterov=True,
            )  # fmt: skip
        else:
            self.opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.steps = 0

    def set_iteration(self, it: int) -> float:
        lr = lr_at(self.cfg, it)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def train_steps(
        self, buf: ReplayBuffer, steps: int, rng: np.random.Generator
    ) -> dict[str, float]:
        self.net.train()
        sums: dict[str, float] = {}
        for _ in range(steps):
            x, pi, z, q = buf.sample(self.cfg.batch_size, rng, self.cfg.augment_mirror)
            if self.cfg.value_target == "zq":
                z = 0.5 * (z + q)
            xt = torch.from_numpy(x).to(self.device, non_blocking=True)
            pit = torch.from_numpy(pi).to(self.device, non_blocking=True)
            zt = torch.from_numpy(z).to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                loss, info = loss_fn(self.net, xt, pit, zt)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.steps += 1
            info["loss/total"] = loss.detach()
            for k, v in info.items():
                sums[k] = sums.get(k, 0.0) + float(v)
        self.net.eval()
        return {k: v / max(1, steps) for k, v in sums.items()}

    def state_dict(self) -> dict:
        return {
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.opt.load_state_dict(d["opt"])
        self.scaler.load_state_dict(d["scaler"])
        self.steps = d["steps"]
