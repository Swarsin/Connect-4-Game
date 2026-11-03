import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";

import { Board, WIDTH } from "../src/engine/board";

interface Case {
  moves: string;
  policy: number[];
  value: number;
}

const fixture = JSON.parse(
  readFileSync(new URL("../../data/fixtures/net_cases.json", import.meta.url), "utf8"),
) as { model: string; cases: Case[] };

const modelPath = fileURLToPath(new URL("../public/models/master.onnx", import.meta.url));

let session: ort.InferenceSession;

beforeAll(async () => {
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });
});

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

async function run(board: Board): Promise<{ policy: number[]; value: number }> {
  const input = new ort.Tensor("float32", board.toPlanes(), [1, 2, 6, 7]);
  const out = await session.run({ board: input });
  const logits = out.policy_logits.data as Float32Array;
  return {
    policy: maskedPolicy(logits, board),
    value: (out.value.data as Float32Array)[0],
  };
}

describe("ONNX model parity with the Python export", () => {
  it(`matches policy and value on ${fixture.cases.length} fixture positions`, async () => {
    let worstPolicy = 0;
    let worstValue = 0;
    for (const c of fixture.cases) {
      const board = Board.fromMoves(c.moves);
      const got = await run(board);
      expect(got.policy.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
      for (let col = 0; col < WIDTH; col++) {
        if (!board.legalMoves().includes(col)) expect(got.policy[col]).toBe(0);
      }
      worstPolicy = Math.max(
        worstPolicy,
        ...got.policy.map((p, i) => Math.abs(p - c.policy[i])),
      );
      worstValue = Math.max(worstValue, Math.abs(got.value - c.value));
    }
    expect(worstPolicy).toBeLessThan(5e-3);
    expect(worstValue).toBeLessThan(5e-3);
  }, 60_000);

  it("values the empty board for player 1", async () => {
    const { value } = await run(new Board());
    expect(value).toBeGreaterThan(0.5);
  });
});
