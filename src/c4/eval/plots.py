"""Generate the report figures from committed results.

python -m c4.eval.plots --out docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"

INK = "#1c1917"
MUTED = "#78716c"
LINE = "#e7e1d8"
BG = "#faf7f2"
TEAL = "#0f766e"
AMBER = "#f59e0b"
ACCENT = "#e4572e"
GREY = "#a8a29e"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": LINE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
    }
)


def _clean(ax: plt.Axes) -> None:
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def training_curve(out: Path) -> None:
    rows = [
        json.loads(x) for x in (RESULTS / "training" / "az-local.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if "iteration" in r]
    evals = [r for r in rows if "bench_s200/preserving" in r]
    eit = [r["iteration"] for r in evals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(
        eit,
        [100 * r["bench_s0/preserving"] for r in evals],
        color=MUTED,
        marker="o",
        ms=3,
        label="policy only",
    )
    ax1.plot(
        eit,
        [100 * r["bench_s200/optimal"] for r in evals],
        color=AMBER,
        marker="o",
        ms=3,
        label="200 sims, optimal",
    )
    ax1.plot(
        eit,
        [100 * r["bench_s200/preserving"] for r in evals],
        color=TEAL,
        marker="o",
        ms=3,
        label="200 sims, preserving",
    )
    ax1.set_xlabel("self-play iteration")
    ax1.set_ylabel("% of held-out positions")
    ax1.set_title("Perfect-play accuracy (dev split)")
    ax1.set_ylim(70, 101)
    ax1.legend(frameon=False, loc="lower right", fontsize=8)
    _clean(ax1)

    ax2.plot(
        eit,
        [100 * r["arena/minimax:d=4"] for r in evals],
        color=AMBER,
        marker="o",
        ms=3,
        label="vs Minimax d4",
    )
    ax2.plot(
        eit,
        [100 * r["arena/mcts:it=2000"] for r in evals],
        color=TEAL,
        marker="o",
        ms=3,
        label="vs MCTS 2k",
    )
    ax2.set_xlabel("self-play iteration")
    ax2.set_ylabel("match score (%)")
    ax2.set_title("Arena score, colour-swapped openings")
    ax2.set_ylim(50, 105)
    ax2.legend(frameon=False, loc="lower right", fontsize=8)
    _clean(ax2)

    fig.suptitle(
        "AlphaZero self-play, 46 iterations on one RTX 3070 (~3 GPU-hours)", y=1.02, fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out / "report-training.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def elo_chart(out: Path) -> None:
    ratings = json.loads((RESULTS / "elo" / "ratings.json").read_text())
    ratings.sort(key=lambda r: r["elo"])
    names = [r["name"] for r in ratings]
    vals = [r["elo"] for r in ratings]
    lo = [r["elo"] - r["lo"] for r in ratings]
    hi = [r["hi"] - r["elo"] for r in ratings]
    colors = [ACCENT if n.startswith("az") else (TEAL if n != "random" else GREY) for n in names]

    fig, ax = plt.subplots(figsize=(8, 0.38 * len(names) + 1.2))
    ax.barh(names, vals, color=colors, height=0.62)
    ax.errorbar(vals, names, xerr=[lo, hi], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
    right = max(v + h for v, h in zip(vals, hi, strict=True))
    for y, v in enumerate(vals):
        ax.text(v + hi[y] + 18, y, f"{v:.0f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Elo (anchored at random = 0)")
    ax.set_title("Rated against the shared match pool")
    ax.set_xlim(min(0, min(vals)) - 40, right + 110)
    _clean(ax)
    ax.grid(axis="x", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "report-elo.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def stage_chart(out: Path) -> None:
    def load(name: str) -> dict:
        return json.loads((RESULTS / "benchmarks" / name).read_text())["results"]

    agents = {
        "AlphaZero 800": load("az45_s800__c4_mixed_test.json"),
        "Minimax d8": load("minimax_d=8__c4_mixed_test.json"),
        "MCTS 2k": load("mcts_it=2000__c4_mixed_test.json"),
    }
    stages = ["0-10", "11-20", "21-30", "31-42"]
    colors = [TEAL, AMBER, MUTED]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, label in zip(
        axes, ("optimal", "preserving"), ("optimal move", "result-preserving move"), strict=True
    ):
        width = 0.26
        for i, (name, res) in enumerate(agents.items()):
            xs = [j + (i - 1) * width for j in range(len(stages))]
            ys = [100 * res[s][metric] for s in stages]
            ax.bar(xs, ys, width=width, color=colors[i], label=name)
        ax.set_xticks(range(len(stages)), [f"moves {s}" for s in stages])
        ax.set_ylim(60, 102)
        ax.set_ylabel("%")
        ax.set_title(label)
        _clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        "Held-out accuracy by game stage (c4_mixed_test, 3,761 positions)", y=1.08, fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "report-stages.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def export_chart(out: Path) -> None:
    data = json.loads((RESULTS / "export.json").read_text())
    variants = data["master_variants"]
    names = ["fp32", "fp16", "int8"]
    sizes = [variants[n]["size_mb"] for n in names]
    lat1 = [variants[n]["latency_b1_ms"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.bar(names, sizes, color=[MUTED, AMBER, TEAL])
    for i, v in enumerate(sizes):
        ax1.text(i, v + 0.3, f"{v:.1f} MB", ha="center", fontsize=8)
    ax1.set_ylabel("model size (MB)")
    ax1.set_title("Size")
    ax1.set_ylim(0, max(sizes) * 1.2)
    _clean(ax1)

    ax2.bar(names, lat1, color=[MUTED, AMBER, TEAL])
    for i, v in enumerate(lat1):
        ax2.text(i, v + 0.05, f"{v:.2f} ms", ha="center", fontsize=8)
    ax2.set_ylabel("batch-1 latency, CPU (ms)")
    ax2.set_title("Latency")
    ax2.set_ylim(0, max(lat1) * 1.2)
    _clean(ax2)
    fig.suptitle("Exported model variants (accuracy preserved within 0.2 pp)", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "report-export.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    training_curve(out)
    elo_chart(out)
    stage_chart(out)
    export_chart(out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
