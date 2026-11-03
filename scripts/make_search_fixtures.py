"""Generate PUCT search fixtures for the TypeScript engine parity test.

    python scripts/make_search_fixtures.py --onnx web/public/models/master.onnx \
        --out data/fixtures/search_cases.json --n 50 --sims 200

The reference uses the same ONNX model the browser ships and the same search parameters, so the
TypeScript port is compared like-for-like (docs/conventions.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from c4.alphazero.mcts import SearchParams, new_root, run_search
from c4.engine import Board
from c4.export.verify import OnnxEvaluator, sample_positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="web/public/models/master.onnx")
    ap.add_argument("--out", default="data/fixtures/search_cases.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--c-puct", type=float, default=2.0)
    a = ap.parse_args()

    evaluator = OnnxEvaluator(a.onnx, threads=8)
    params = SearchParams(sims=a.sims, c_puct=a.c_puct, fpu=0.0)
    positions = sample_positions(a.n, seed=11)
    cases = []
    for moves in positions:
        board = Board.from_moves(moves)
        root = new_root(board)
        run_search([root], [board], evaluator, params)
        visits = list(root.N)
        best = int(np.flatnonzero(np.asarray(visits) == max(visits))[0])
        cases.append(
            {
                "moves": moves,
                "best_move": best,
                "visits": visits,
                "value": round(root.value(), 5),
            }
        )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": Path(a.onnx).name,
                "sims": a.sims,
                "c_puct": a.c_puct,
                "fpu": 0.0,
                "cases": cases,
            },
            indent=1,
        )
    )
    print(f"wrote {len(cases)} search cases to {out}")


if __name__ == "__main__":
    main()
