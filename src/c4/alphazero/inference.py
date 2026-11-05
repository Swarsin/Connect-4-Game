"""Fast inference copy of :class:`C4Net` for search.

Small batches on a GPU are dominated by kernel-launch overhead, not arithmetic. This module:

1. folds every BatchNorm into the preceding convolution (eval-mode identity),
2. runs in fp16 on CUDA,
3. captures one CUDA graph per batch-size bucket (powers of two) so a whole forward pass is a
   single graph replay.

The copy is a *snapshot* of the weights at construction time.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from c4.alphazero.network import C4Net


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    assert bn.weight is not None and bn.bias is not None  # BatchNorm2d is always affine here
    assert bn.running_var is not None and bn.running_mean is not None
    w = conv.weight.detach()
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    w = w * scale.reshape(-1, 1, 1, 1)
    b = bn.bias.detach() - bn.running_mean.detach() * scale
    if conv.bias is not None:
        b = b + conv.bias.detach() * scale
    return w.contiguous(), b.contiguous()


def _fold_modules(conv: object, bn: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Narrow ``nn.Sequential``/``ModuleList`` lookups to the modules used by ``C4Net``."""
    assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
    return _fold(conv, bn)


class FusedNet(nn.Module):
    def __init__(self, net: C4Net) -> None:
        super().__init__()
        net = net.eval()
        p: list[torch.Tensor] = []
        p += _fold_modules(net.stem[0], net.stem[1])
        for blk in net.blocks:
            p += _fold_modules(blk.c1, blk.b1)
            p += _fold_modules(blk.c2, blk.b2)
        p += _fold_modules(net.policy_head[0], net.policy_head[1])
        policy_fc = cast(nn.Linear, net.policy_head[4])
        assert policy_fc.bias is not None  # Linear layers are constructed with bias
        p += [policy_fc.weight.detach(), policy_fc.bias.detach()]
        p += _fold_modules(net.value_head[0], net.value_head[1])
        value_fc1 = cast(nn.Linear, net.value_head[4])
        value_fc2 = cast(nn.Linear, net.value_head[6])
        assert value_fc1.bias is not None and value_fc2.bias is not None
        p += [value_fc1.weight.detach(), value_fc1.bias.detach()]
        p += [value_fc2.weight.detach(), value_fc2.bias.detach()]
        self.params = nn.ParameterList([nn.Parameter(t.clone(), requires_grad=False) for t in p])
        self.n_blocks = len(net.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``[n, 8]``: 7 policy logits followed by the value."""
        P = self.params
        h = F.relu(F.conv2d(x, P[0], P[1], padding=1))
        i = 2
        for _ in range(self.n_blocks):
            y = F.relu(F.conv2d(h, P[i], P[i + 1], padding=1))
            y = F.conv2d(y, P[i + 2], P[i + 3], padding=1)
            h = F.relu(h + y)
            i += 4
        pol = F.relu(F.conv2d(h, P[i], P[i + 1])).flatten(1)
        logits = F.linear(pol, P[i + 2], P[i + 3])
        val = F.relu(F.conv2d(h, P[i + 4], P[i + 5])).flatten(1)
        val = F.relu(F.linear(val, P[i + 6], P[i + 7]))
        val = torch.tanh(F.linear(val, P[i + 8], P[i + 9]))
        return torch.cat([logits, val], dim=1)


class FastInference:
    """Callable ``planes [n,2,6,7] float32 (cpu tensor) -> [n, 8] float32 (cpu tensor)``."""

    def __init__(self, net: C4Net, device: torch.device, half: bool = True, graphs: bool = True,
                 max_batch: int = 1024) -> None:  # fmt: skip
        self.device = device
        self.cuda = device.type == "cuda"
        self.dtype = torch.float16 if (half and self.cuda) else torch.float32
        self.model = FusedNet(net).to(device=device, dtype=self.dtype).eval()
        self.use_graphs = graphs and self.cuda
        self.max_batch = max_batch
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}

    def _bucket(self, n: int) -> int:
        b = 1
        while b < n:
            b *= 2
        return b

    def _graph(self, bs: int):
        if bs not in self._graphs:
            x = torch.zeros((bs, 2, 6, 7), device=self.device, dtype=self.dtype)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(2):
                    self.model(x)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.inference_mode():
                out = self.model(x)
            self._graphs[bs] = (g, x, out)
        return self._graphs[bs]

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if not self.use_graphs or n > self.max_batch:
            return self.model(x.to(self.device, self.dtype, non_blocking=True)).float().cpu()
        g, sx, out = self._graph(self._bucket(n))
        sx[:n].copy_(x.to(self.device, non_blocking=True))
        if n < sx.shape[0]:
            sx[n:].zero_()
        g.replay()
        return out[:n].float().cpu()
