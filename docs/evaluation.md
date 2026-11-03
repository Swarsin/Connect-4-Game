# Evaluation protocol

The goal of this project is not only to play Connect 4 well, but to *measure* how well. This
document defines every number reported in `results/`.

## Ground truth: perfect play

Connect 4 is solved. `scripts/build_solver.sh` builds Pascal Pons' solver (AGPL-3.0, used as an
external process only, never bundled or committed) together with its 7x6 opening book. The solver
returns, for every legal move, a score from the perspective of the side to move:

* `> 0` — the side to move can force a win; larger is faster (score `22 - k` when the win is
  completed with the side's `k`-th stone),
* `0` — draw with perfect play,
* `< 0` — the side to move loses against perfect play.

Every benchmark position is stored with all seven scores
(`data/benchmarks/*.jsonl`, committed so no solver is needed at training or test time).

## Benchmarks

| file | contents |
|---|---|
| `pons_test_l3_r1` … `pons_test_l1_r3` | Pons' standard test sets (1,000 positions each), endgame → opening |
| `c4_mixed_dev` / `c4_mixed_test` | positions sampled evenly across plies 0–40 from games between a pool of agents (random, Minimax depth 1–6, MCTS 200/1,000 iterations) with 15% random moves; de-duplicated modulo mirror; split by game |

The mixed benchmark is what the network is tracked on during training (`dev`); `test` is only used
for final reporting. Pons' sets are used to compare with published solver work.

## Metrics

For a set of positions and an agent's chosen move per position:

* **optimal** — the chosen move is among the moves with the maximum solver score (fastest win /
  slowest loss). Strict.
* **result-preserving** — the chosen move keeps the theoretical result (win stays a win, draw stays
  a draw). Less strict, arguably what matters in practice.
* **value accuracy / MSE** (network agents) — the predicted value is bucketed into
  loss / draw / win with thresholds ±1/3 and compared with the sign of the best solver score.

All metrics are reported overall and by game stage (moves 0–10, 11–20, 21–30, 31–42).

## Match play and Elo

`c4.agents.registry` builds agents from compact specs (`random`, `minimax:d=6`,
`mcts:it=2000`, `az:ckpt=models/x.pt,sims=200`, `solver`). Matches play a fixed opening schedule
with colours swapped, games optionally distributed over worker processes; every record is appended
to `results/elo/games.jsonl`.

Ratings are a Bradley–Terry maximum-likelihood fit (minorisation–maximisation) with draws counted
as half a win, a small prior so a perfect record stays finite, and 200 bootstrap resamples for 95%
confidence intervals. `random` is fixed at 0 Elo.

Match-play results depend on the opening schedule; the default one-move schedule (7 × 2 = 14 games)
is quick but noisy, the two-move schedule (49 × 2 = 98 games) is the default for reported tables.

## Fixed legacy issues

The evaluation harness and the classic bots were written to replace these problems in the original
implementation:

1. `Backpropagate` never changed perspective, so every node received the same score.
2. `SimulateMove` mutated shared `player1`/`player2` turn flags, drifting across rollouts.
3. `UpdateMove` compared an integer to `Node` objects, so tree reuse never happened.
4. The Minimax evaluation scored only the AI's own windows and never penalised opponent threats.
5. Every search node deep-copied a board of `Stack` objects (38x slower than a bitboard copy).
6. MCTS was limited by wall-clock time, making results irreproducible.

## Reproducing

For a checkpoint, `python -m c4.eval.az_report` does all of the below in one batched GPU pass:
benchmarks at several simulation budgets, arena games, Elo refit, solver games and
`results/az/<name>.json`.

```bash
scripts/build_solver.sh
python scripts/download_benchmarks.py
python scripts/make_benchmark.py
python -m c4.eval.cli elo --agents random minimax:d=4 mcts:it=2000 --workers 8
python -m c4.eval.cli benchmark "minimax:d=8" --file data/benchmarks/c4_mixed_test.jsonl
python -m c4.eval.cli report     # rewrites results/README.md
```
