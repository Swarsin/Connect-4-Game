"""Generate shared Python <-> TypeScript engine fixtures (data/fixtures/engine_cases.json).

For every prefix of a set of random games (plus hand-picked edge cases) record legal moves,
terminal flag, winner, side to move, immediate wins and the network input planes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from c4.engine import Board, to_planes

EDGE_CASES = [
    "1213141",
    "7676767",
    "1122334",
    "4455667",
    "12233434414",
    "76655454474",
    "17273757",
    "472424756645422154157737216635626711335331",
    "111111",
    "444444",
]


def record(b: Board) -> dict:
    planes = to_planes(b).astype(int).reshape(-1)
    return {
        "moves": b.to_moves(),
        "legal": b.legal_moves(),
        "terminal": b.is_terminal(),
        "winner": b.winner(),
        "to_move": b.player_to_move(),
        "wins_me": b.winning_moves(True),
        "wins_opp": b.winning_moves(False),
        "planes": "".join(map(str, planes.tolist())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/fixtures/engine_cases.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    games = list(EDGE_CASES)
    for _ in range(args.games):
        b = Board()
        while not b.is_terminal():
            b.play(rng.choice(b.legal_moves()))
        games.append(b.to_moves())

    seen: set[str] = set()
    cases = []
    for g in games:
        b = Board()
        for i in range(len(g) + 1):
            if i:
                b.play(int(g[i - 1]) - 1)
            if b.to_moves() not in seen:
                seen.add(b.to_moves())
                cases.append(record(b))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"version": 1, "planes_layout": "[2,6,7] row0=bottom", "cases": cases})
    )
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
