# Conventions

These conventions are shared by every part of the project (Python engine, training, evaluation,
TypeScript engine and web app). They are fixed: changing one means changing every component and
regenerating the shared fixtures in `data/fixtures/`.

1. **Players.** Player 1 moves first, player 2 second. Internally the engine works from the
   perspective of the *side to move* ("current player") and the *opponent*.
2. **Move notation.** A game or position is a string of 1-indexed column digits, e.g. `"4453"`
   (same as Pascal Pons' solver and benchmarks). Used in URLs, the database, fixtures and
   benchmarks. Internally, columns are 0-indexed (`0..6`).
3. **Network input.** `float32` tensor of shape `[N, 2, 6, 7]` laid out `[channel, row, col]`.
   Channel 0 = stones of the side to move, channel 1 = opponent stones.
   **Row 0 is the bottom row**, column 0 is the leftmost column.
4. **Network outputs.** `policy_logits` `[N, 7]` (raw logits; illegal moves are masked after the
   network) and `value` `[N, 1]` (tanh, in `[-1, 1]`, from the perspective of the side to move).
5. **Search values.** Always from the perspective of the side to move at that node and negated
   when backed up to the parent. At a terminal node the side to move has value `-1` if the
   previous mover just won, `0` for a draw.
6. **Solver scores** (Pons convention). Positive = side to move wins, larger = faster win,
   `0` = draw, negative = side to move loses. If the side to move can force a win with its `k`-th
   stone (counting all of its stones on the board), the score is `22 - k`; a forced loss to the
   opponent's `k`-th stone scores `-(22 - k)`.
7. **Randomness.** Every stochastic component takes an explicit seed; training runs log all seeds.
8. **Configuration.** Training/eval/ablation settings live in YAML under `configs/` and are
   validated by pydantic models. No magic numbers in training code.
