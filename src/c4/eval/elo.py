"""Bradley-Terry / Elo ratings from game records, with bootstrap confidence intervals.

Draws count as half a win for each side. A small prior (``prior`` virtual draws per pair that
has played) keeps ratings finite when an agent wins every game. The anchor agent (default
``random``) is fixed at 0 Elo.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from c4.eval.arena import GameRecord


@dataclass
class Rating:
    name: str
    elo: float
    lo: float
    hi: float
    games: int
    score: float  # mean score over all its games


def _pair_stats(games: Sequence[GameRecord]) -> dict[tuple[str, str], list[float]]:
    """(a, b) with a < b -> [score of a, games]."""
    st: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for g in games:
        a, b = g.first, g.second
        sa = 0.5 if g.winner == 0 else 1.0 if g.winner == 1 else 0.0
        if a > b:
            a, b, sa = b, a, 1.0 - sa
        st[(a, b)][0] += sa
        st[(a, b)][1] += 1
    return st


def fit_bradley_terry(
    games: Sequence[GameRecord], anchor: str = "random", prior: float = 1.0, iters: int = 2000
) -> dict[str, float]:
    st = _pair_stats(games)
    players = sorted({p for k in st for p in k})
    if not players:
        return {}
    wins: dict[str, float] = defaultdict(float)
    n: dict[tuple[str, str], float] = {}
    for (a, b), (sa, cnt) in st.items():
        wins[a] += sa + prior / 2
        wins[b] += cnt - sa + prior / 2
        n[(a, b)] = n[(b, a)] = cnt + prior
    nbrs: dict[str, list[str]] = defaultdict(list)
    for a, b in st:
        nbrs[a].append(b)
        nbrs[b].append(a)

    r = {p: 1.0 for p in players}
    for _ in range(iters):  # minorisation-maximisation (Hunter 2004)
        new = {}
        for p in players:
            denom = sum(n[(p, q)] / (r[p] + r[q]) for q in nbrs[p])
            new[p] = wins[p] / denom if denom > 0 else r[p]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {p: v / g for p, v in new.items()}
        delta = max(abs(math.log(new[p] / r[p])) for p in players)
        r = new
        if delta < 1e-10:
            break

    elo = {p: 400.0 * math.log10(v) for p, v in r.items()}
    shift = elo.get(anchor, min(elo.values()))
    return {p: e - shift for p, e in elo.items()}


def ratings(
    games: Sequence[GameRecord],
    anchor: str = "random",
    bootstrap: int = 200,
    seed: int = 0,
    prior: float = 1.0,
) -> list[Rating]:
    base = fit_bradley_terry(games, anchor, prior)
    rng = random.Random(seed)
    by_pair: dict[tuple[str, str], list[GameRecord]] = defaultdict(list)
    for g in games:
        by_pair[tuple(sorted((g.first, g.second)))].append(g)  # type: ignore[index]
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap):
        resampled = [rng.choice(gs) for gs in by_pair.values() for _ in gs]
        for p, e in fit_bradley_terry(resampled, anchor, prior, iters=500).items():
            samples[p].append(e)

    total: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for g in games:
        sa = 0.5 if g.winner == 0 else 1.0 if g.winner == 1 else 0.0
        total[g.first][0] += sa
        total[g.first][1] += 1
        total[g.second][0] += 1 - sa
        total[g.second][1] += 1

    out = []
    for p, e in base.items():
        s = sorted(samples[p]) if samples[p] else [e]
        lo = s[int(0.025 * (len(s) - 1))]
        hi = s[int(0.975 * (len(s) - 1))]
        out.append(Rating(p, e, lo, hi, int(total[p][1]), total[p][0] / max(1, total[p][1])))
    return sorted(out, key=lambda x: -x.elo)


def format_ratings(rs: Sequence[Rating]) -> str:
    lines = ["| agent | Elo | 95% CI | games | score |", "|---|---|---|---|---|"]
    for r in rs:
        lines.append(
            f"| {r.name} | {r.elo:.0f} | [{r.lo:.0f}, {r.hi:.0f}] | {r.games} | {100 * r.score:.1f}% |"
        )
    return "\n".join(lines)
