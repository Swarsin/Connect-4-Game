"""Batched (GPU) evaluation of an AlphaZero checkpoint.

    python -m c4.eval.az_report --ckpt runs/az-local/models/model_iter_0045.pt --name az45 \
        --sims 0 200 800 --arena-sims 200 --solver-games 30 --elo

Produces the same artefacts as the spec-string CLI, but runs all network inference in batches:

* `results/benchmarks/<name>_s<sims>__<file>.json` — perfect-play accuracy per budget
* `results/az/<name>.json` — combined summary
* `results/vs_solver/<name>.json` — record against the perfect solver per colour
* appends arena games to `results/elo/games.jsonl` and refits `results/elo/ratings.{json,md}`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from c4.eval.arena import GameRecord, load_games, save_games

RESULTS = Path("results")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in name)


def _drop_agent_games(path: Path, prefix: str) -> list[GameRecord]:
    """Remove previous games involving agents named ``prefix...`` so reruns are idempotent."""
    games = [
        g
        for g in load_games(path)
        if not (g.first.startswith(prefix) or g.second.startswith(prefix))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(g.__dict__) + "\n" for g in games))
    return games


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sims", type=int, nargs="*", default=[0, 200])
    ap.add_argument("--bench", nargs="*", default=[
        "data/benchmarks/c4_mixed_test.jsonl",
        "data/benchmarks/pons_test_l3_r1.jsonl",
    ])  # fmt: skip
    ap.add_argument("--bench-limit", type=int, default=None)
    ap.add_argument("--arena", nargs="*", default=["minimax:d=4", "minimax:d=8", "mcts:it=2000"])
    ap.add_argument("--arena-sims", type=int, default=200)
    ap.add_argument("--arena-openings", default="one")
    ap.add_argument("--solver-games", type=int, default=0, help="games per colour vs the solver")
    ap.add_argument("--openings-elo", default="one")
    ap.add_argument("--elo", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from c4.alphazero.mcts import NetEvaluator, SearchParams
    from c4.alphazero.network import load_model, resolve_device
    from c4.eval.az_eval import arena_net, benchmark_net
    from c4.eval.benchmark import load_benchmark

    name = a.name or Path(a.ckpt).stem
    agent = f"{name}:s{a.arena_sims}"
    device = resolve_device(a.device)
    net = load_model(a.ckpt, device)
    ev = NetEvaluator(net, device, amp=device.type == "cuda")
    params = SearchParams(sims=max([a.arena_sims, *a.sims]), c_puct=2.0)
    summary: dict[str, Any] = {"checkpoint": a.ckpt, "name": name, "device": str(device),
                               "params": vars(params), "benchmarks": {}, "arena": {}}  # fmt: skip

    for f in a.bench or []:
        items = load_benchmark(f, a.bench_limit)
        stem = Path(f).stem
        for sims in a.sims:
            t = time.time()
            res = benchmark_net(ev, items, sims, params)
            row = {"n": len(items), "sims": sims, **{k: v for k, v in res.items()}}
            summary["benchmarks"].setdefault(stem, {})[str(sims)] = row
            out = RESULTS / "benchmarks" / f"{_safe(name)}_s{sims}__{stem}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"agent": f"{name}:s{sims}", "benchmark": stem,
                                       "limit": a.bench_limit, "results": res}, indent=1))  # fmt: skip
            allr = res["all"]
            print(f"[bench] {stem} sims={sims}: optimal {100 * allr['optimal']:.1f}% "
                  f"preserving {100 * allr['preserving']:.1f}% "
                  + (f"value_acc {100 * allr['value_acc']:.1f}% " if "value_acc" in allr else "")
                  + f"({time.time() - t:.0f}s)")  # fmt: skip

    games_path = RESULTS / "elo" / "games.jsonl"
    baseline = _drop_agent_games(games_path, f"{name}:")
    new_games: list[GameRecord] = []
    for opp in a.arena or []:
        t = time.time()
        arena_res = arena_net(ev, agent, opp, a.arena_sims, params, a.arena_openings, seed=a.seed,
                              shared_opponent=True)  # fmt: skip
        new_games += arena_res.games
        summary["arena"][opp] = {"a_wins": arena_res.a_wins, "draws": arena_res.draws,
                                 "b_wins": arena_res.b_wins, "score": arena_res.a_score}  # fmt: skip
        print(f"[arena] {arena_res.summary()} ({time.time() - t:.0f}s)")

    if a.solver_games:
        t = time.time()
        solver_res = arena_net(ev, agent, "solver", a.arena_sims, params, [""] * a.solver_games,
                               seed=a.seed, shared_opponent=True)  # fmt: skip
        new_games += solver_res.games
        per_colour = {}
        for first in (True, False):
            gs = [g for g in solver_res.games if (g.first == agent) == first]
            wins = sum(1 for g in gs if g.winner == (1 if first else 2))
            losses = sum(1 for g in gs if g.winner == (2 if first else 1))
            lengths = [len(g.moves) for g in gs]
            key = "as_first" if first else "as_second"
            per_colour[key] = {
                "games": len(gs), "wins": wins, "draws": len(gs) - wins - losses, "losses": losses,
                "mean_length": sum(lengths) / len(lengths) if lengths else None,
                "max_length_rate": sum(1 for x in lengths if x >= 41) / max(1, len(lengths)),
            }  # fmt: skip
            print(f"[solver] {key}: {per_colour[key]}")
        summary["vs_solver"] = per_colour
        out = RESULTS / "vs_solver" / f"{_safe(name)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"agent": agent, **per_colour}, indent=1))
        print(f"[solver] {len(solver_res.games)} games ({time.time() - t:.0f}s)")

    if new_games:
        save_games(new_games, games_path)
        all_games = baseline + new_games
    else:
        all_games = baseline

    if a.elo and all_games:
        from c4.eval.elo import format_ratings, ratings

        rs = ratings(all_games, anchor="random", bootstrap=a.bootstrap)
        table = format_ratings(rs)
        (RESULTS / "elo" / "ratings.md").write_text(table + "\n")
        (RESULTS / "elo" / "ratings.json").write_text(
            json.dumps([r.__dict__ for r in rs], indent=1)
        )
        summary["elo"] = {r.name: round(r.elo) for r in rs}
        print("\n" + table)

    out = RESULTS / "az" / f"{_safe(name)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
