"""Produce int8 (static, QDQ) and fp16 variants of an ONNX model and compare them.

    python -m c4.export.quantize web/public/models/master.onnx --out-dir /tmp/q

Reports size, CPU latency (batch 1 and 8) and benchmark accuracy (sims 0 and 200) per variant.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from c4.engine import Board
from c4.engine.encoding import to_planes_batch
from c4.export.verify import OnnxEvaluator, sample_positions


class _Calib:
    def __init__(self, n: int = 2000) -> None:
        boards = [
            Board.from_moves(m)
            for m in sample_positions(n, seed=3, bench="data/benchmarks/c4_mixed_dev.jsonl")
        ]
        self.data = iter([{"board": to_planes_batch([b])} for b in boards])

    def get_next(self):
        return next(self.data, None)


def quantize_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_name(dst.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre))
    quantize_static(
        str(pre), str(dst), _Calib(), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
    )  # fmt: skip
    pre.unlink(missing_ok=True)
    return dst


def convert_fp16(src: Path, dst: Path) -> Path:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    m = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(m, str(dst))
    return dst


def latency_ms(path: Path, batch: int, reps: int = 200) -> float:
    ev = OnnxEvaluator(path, threads=1)
    x = np.zeros((batch, 2, 6, 7), np.float32)
    for _ in range(10):
        ev.raw(x)
    t = time.perf_counter()
    for _ in range(reps):
        ev.raw(x)
    return (time.perf_counter() - t) / reps * 1000


def accuracy(path: Path, sims: int, limit: int) -> dict:
    from c4.alphazero.mcts import SearchParams
    from c4.eval.az_eval import benchmark_net
    from c4.eval.benchmark import load_benchmark

    items = load_benchmark("data/benchmarks/c4_mixed_test.jsonl", limit)
    return benchmark_net(OnnxEvaluator(path, threads=4), items, sims, SearchParams(sims=sims))[
        "all"
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sims", type=int, nargs="*", default=[0, 200])
    ap.add_argument("--report", default="results/export.json")
    a = ap.parse_args()
    src = Path(a.onnx)
    out = Path(a.out_dir or src.parent)
    out.mkdir(parents=True, exist_ok=True)
    variants = {
        "fp32": src,
        "int8": quantize_int8(src, out / f"{src.stem}.int8.onnx"),
        "fp16": convert_fp16(src, out / f"{src.stem}.fp16.onnx"),
    }
    report = {}
    for name, p in variants.items():
        row = {
            "file": str(p), "size_mb": p.stat().st_size / 1e6,
            "latency_b1_ms": latency_ms(p, 1), "latency_b8_ms": latency_ms(p, 8),
        }  # fmt: skip
        for s in a.sims:
            r = accuracy(p, s, a.limit)
            row[f"s{s}_optimal"] = r["optimal"]
            row[f"s{s}_preserving"] = r["preserving"]
        report[name] = row
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps({"source": str(src), "variants": report}, indent=1))


if __name__ == "__main__":
    main()
