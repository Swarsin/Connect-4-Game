"""ONNX Runtime evaluator, parity checks against PyTorch, and TypeScript test fixtures.

python -m c4.export.verify models/x.pt web/public/models/x.onnx --fixtures data/fixtures/net_cases.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from c4.engine import WIDTH, Board
from c4.engine.board import TOP
from c4.engine.encoding import to_planes_batch


class OnnxEvaluator:
    """MCTS evaluator backed by an ONNX model (same maths as the browser)."""

    def __init__(self, path: str | Path, threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def raw(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, v = self.sess.run(None, {"board": x.astype(np.float32)})
        return logits, v[:, 0]

    def evaluate(self, boards: Sequence[Board]) -> tuple[np.ndarray, np.ndarray]:
        logits, v = self.raw(to_planes_batch(boards))
        logits = logits.astype(np.float64)
        for i, b in enumerate(boards):
            for c in range(WIDTH):
                if b.mask & TOP[c]:
                    logits[i, c] = -np.inf
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        return p.astype(np.float32), v.astype(np.float32)


def sample_positions(
    n: int, seed: int = 0, bench: str = "data/benchmarks/c4_mixed_test.jsonl"
) -> list[str]:
    from c4.eval.benchmark import load_benchmark

    items = load_benchmark(bench)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=min(n, len(items)), replace=False)
    return [items[i].moves for i in idx]


def parity(checkpoint: str | Path, onnx_path: str | Path, n: int = 1000) -> float:
    import torch

    from c4.alphazero.network import load_model

    net = load_model(checkpoint, "cpu")
    boards = [Board.from_moves(m) for m in sample_positions(n)]
    x = to_planes_batch(boards)
    with torch.inference_mode():
        lt, vt = net(torch.from_numpy(x))
    lo, vo = OnnxEvaluator(onnx_path, threads=4).raw(x)
    return float(max(np.abs(lt.numpy() - lo).max(), np.abs(vt.numpy()[:, 0] - vo).max()))


def regression(checkpoint: str | Path, onnx_path: str | Path, n: int = 1000) -> dict[str, float]:
    """Compare fp32 PyTorch with the (possibly quantised) ONNX model on what matters:
    value error, policy total-variation distance and agreement of the greedy move."""
    import torch

    from c4.alphazero.network import load_model

    net = load_model(checkpoint, "cpu")
    boards = [Board.from_moves(m) for m in sample_positions(n)]
    x = to_planes_batch(boards)
    with torch.inference_mode():
        lt, vt = net(torch.from_numpy(x))
    lo, vo = OnnxEvaluator(onnx_path, threads=4).raw(x)

    def pol(logits: np.ndarray) -> np.ndarray:
        z = logits.astype(np.float64).copy()
        for i, b in enumerate(boards):
            for c in range(WIDTH):
                if b.mask & TOP[c]:
                    z[i, c] = -np.inf
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    pt, po = pol(lt.numpy()), pol(lo)
    return {
        "value_mae": float(np.abs(vt.numpy()[:, 0] - vo).mean()),
        "value_max": float(np.abs(vt.numpy()[:, 0] - vo).max()),
        "policy_tv_mean": float(0.5 * np.abs(pt - po).sum(1).mean()),
        "policy_tv_max": float(0.5 * np.abs(pt - po).sum(1).max()),
        "argmax_agreement": float((pt.argmax(1) == po.argmax(1)).mean()),
    }


def write_fixtures(
    onnx_path: str | Path, out: str | Path, n: int = 50, model_url: str | None = None
) -> None:
    moves = ["", "4", "44", "4453"] + sample_positions(n - 4, seed=1)
    boards = [Board.from_moves(m) for m in moves]
    policy, value = OnnxEvaluator(onnx_path).evaluate(boards)
    cases = [
        {"moves": m, "policy": [round(float(x), 6) for x in p], "value": round(float(v), 6)}
        for m, p, v in zip(moves, policy, value, strict=True)
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps({"model": model_url or Path(onnx_path).name, "cases": cases}, indent=1)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("onnx")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument(
        "--tol", type=float, default=1e-4, help="max |torch - onnx| raw logit tolerance"
    )
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--quantised", action="store_true", help="report policy/value regression only")
    a = ap.parse_args()
    if a.quantised:
        r = regression(a.checkpoint, a.onnx, a.n)
        print({k: round(v, 5) for k, v in r.items()})
        if r["argmax_agreement"] < 0.97 or r["policy_tv_mean"] > 0.02 or r["value_mae"] > 0.05:
            raise SystemExit("quantised model regression check FAILED")
    else:
        diff = parity(a.checkpoint, a.onnx, a.n)
        print(f"max |torch - onnx| over {a.n} positions: {diff:.2e}")
        if diff > a.tol:
            raise SystemExit(f"parity check FAILED (tolerance {a.tol})")
    if a.fixtures:
        write_fixtures(a.onnx, a.fixtures)
        print(f"wrote {a.fixtures}")


if __name__ == "__main__":
    main()
