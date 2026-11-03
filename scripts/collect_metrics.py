"""Copy a run's metrics into results/training for reproducible figures.

python scripts/collect_metrics.py --run runs/az-local --out results/training/az-local.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEEP = {
    "iteration",
    "total/games",
    "total/positions",
    "total/gpu_hours",
    "buffer/size",
    "train/lr",
    "train/steps",
    "loss/total",
    "loss/policy",
    "loss/value",
    "policy/entropy",
    "policy/target_entropy",
    "selfplay/mean_length",
    "selfplay/p1_win_rate",
    "time/games_per_hour",
    "time/iter_s",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/az-local")
    ap.add_argument("--out", default="results/training/az-local.jsonl")
    a = ap.parse_args()

    src = Path(a.run) / "metrics.jsonl"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        row = {}
        for k, v in raw.items():
            if k in KEEP or k.startswith(("arena/", "bench_")):
                row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)

    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"wrote {len(rows)} iterations to {out}")


if __name__ == "__main__":
    main()
