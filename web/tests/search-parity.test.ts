import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";

import { Board, WIDTH } from "../src/engine/board";
import { chooseMove, newRoot, runSearch, type AzEvaluator } from "../src/ai/puct";

interface Case {
  moves: string;
  best_move: number;
  visits: number[];
  value: number;
}

const fixture = JSON.parse(
  readFileSync(new URL("../../data/fixtures/search_cases.json", import.meta.url), "utf8"),
) as { sims: number; c_puct: number; fpu: number; cases: Case[] };

const modelPath = fileURLToPath(new URL("../public/models/master.onnx", import.meta.url));

let session: ort.InferenceSession;
let evaluator: AzEvaluator;

function maskedPolicy(logits: ArrayLike<number>, board: Board): number[] {
  const legal = new Set(board.legalMoves());
  const z = new Array<number>(WIDTH).fill(-Infinity);
  let max = -Infinity;
  for (let c = 0; c < WIDTH; c++) {
    if (!legal.has(c)) continue;
    z[c] = Number(logits[c]);
    if (z[c] > max) max = z[c];
  }
  const p = z.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = p.reduce((a, b) => a + b, 0);
  return p.map((v) => v / sum);
}

beforeAll(async () => {
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });
  const cache = new Map<number, { priors: number[]; value: number }>();
  evaluator = {
    async evaluate(board) {
      const key = board.key();
      const hit = cache.get(key);
      if (hit) return hit;
      const input = new ort.Tensor("float32", board.toPlanes(), [1, 2, 6, 7]);
      const out = await session.run({ board: input });
      const res = {
        priors: maskedPolicy(out.policy_logits.data as Float32Array, board),
        value: (out.value.data as Float32Array)[0],
      };
      cache.set(key, res);
      return res;
    },
  };
});

describe("PUCT parity with the Python reference", () => {
  it(`picks the same move on >=98% of ${fixture.cases.length} positions`, async () => {
    let matches = 0;
    const misses: string[] = [];
    for (const c of fixture.cases) {
      const board = Board.fromMoves(c.moves);
      const root = newRoot(board);
      await runSearch(root, board, evaluator, {
        sims: fixture.sims,
        cPuct: fixture.c_puct,
        fpu: fixture.fpu,
      });
      const move = chooseMove(root, 0, Math.random);
      if (move === c.best_move) matches++;
      else misses.push(`${c.moves}: got ${move}, expected ${c.best_move}`);
      expect(root.N.reduce((a, b) => a + b, 0)).toBe(fixture.sims);
    }
    const rate = matches / fixture.cases.length;
    console.log(`PUCT parity: ${matches}/${fixture.cases.length} (${(rate * 100).toFixed(0)}%)`);
    if (misses.length) console.log(`misses:\n${misses.slice(0, 5).join("\n")}`);
    expect(rate).toBeGreaterThanOrEqual(0.98);
  }, 180_000);

  it("values a forced win as positive for the side to move", async () => {
    const board = Board.fromMoves("121212"); // player 1 wins with column 1
    const root = newRoot(board);
    await runSearch(root, board, evaluator, { sims: 200, cPuct: 2, fpu: 0 });
    expect(root.value()).toBeGreaterThan(0.5);
    expect(chooseMove(root, 0, Math.random)).toBe(0);
  }, 30_000);
});
