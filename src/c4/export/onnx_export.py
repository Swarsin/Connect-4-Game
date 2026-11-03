"""Export :class:`C4Net` checkpoints to ONNX.

    python -m c4.export.onnx_export models/az-main-final.pt web/public/models/master.onnx

Input ``board`` float32 ``[N, 2, 6, 7]``; outputs ``policy_logits`` ``[N, 7]`` and ``value``
``[N, 1]`` (docs/conventions.md). BatchNorm is folded into the convolutions before export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from c4.alphazero.inference import FusedNet
from c4.alphazero.network import load_model


class _ExportWrapper(nn.Module):
    def __init__(self, fused: FusedNet) -> None:
        super().__init__()
        self.fused = fused

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.fused(board)
        return out[:, :7], out[:, 7:]


def export_onnx(checkpoint: str | Path, out: str | Path, opset: int = 17) -> Path:
    net = load_model(checkpoint, "cpu")
    model = _ExportWrapper(FusedNet(net)).eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 2, 6, 7)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={"board": {0: "N"}, "policy_logits": {0: "N"}, "value": {0: "N"}},
        opset_version=opset,
        dynamo=False,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    p = export_onnx(a.checkpoint, a.out, a.opset)
    print(f"exported {p} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
