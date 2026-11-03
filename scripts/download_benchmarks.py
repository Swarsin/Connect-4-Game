"""Download Pascal Pons' standard test sets and label every position with per-move solver scores.

Output: data/benchmarks/pons_<set>.jsonl with ``{"moves", "n_moves", "score", "scores"}`` per line.
Requires the solver (scripts/build_solver.sh).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from c4.solver import analyze_many

SETS = ["Test_L3_R1", "Test_L2_R1", "Test_L2_R2", "Test_L1_R1", "Test_L1_R2", "Test_L1_R3"]
URL = "http://blog.gamesolver.org/data/{}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/benchmarks")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sets", nargs="*", default=SETS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.sets:
        out = out_dir / f"pons_{name.lower()}.jsonl"
        if out.exists():
            print(f"skip {out} (exists)")
            continue
        raw = urllib.request.urlopen(URL.format(name), timeout=60).read().decode()
        rows = [line.split() for line in raw.splitlines() if line.strip()]
        positions = [r[0] for r in rows]
        t = time.time()
        scores = analyze_many(positions, workers=args.workers)
        with out.open("w") as f:
            for (moves, given), sc in zip(rows, scores, strict=True):
                best = max(s for s in sc if s is not None)
                if best != int(given):
                    raise SystemExit(f"score mismatch for {moves}: solver {best} vs file {given}")
                f.write(
                    json.dumps({"moves": moves, "n_moves": len(moves), "score": best, "scores": sc})
                    + "\n"
                )
        print(f"{name}: {len(rows)} positions labelled in {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
