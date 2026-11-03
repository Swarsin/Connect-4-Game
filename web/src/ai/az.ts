/**
 * AlphaZero inference in the browser: ONNX Runtime Web (wasm) + PUCT.
 *
 * Models are the static-int8 exports in `public/models` (see results/export.json). The worker
 * passes an absolute `wasmBase` because the site is served from a sub-path on GitHub Pages and
 * ORT needs to fetch its runtime from `public/ort`.
 */

import type { InferenceSession } from "onnxruntime-web/wasm";

import { Board, WIDTH } from "../engine/board";
import { chooseFromPriors, chooseMove, newRoot, runSearch, type AzEvaluator } from "./puct";

export interface AzLevel {
  model: string;
  sims: number;
  temperature: number;
  wasmBase: string;
}

let ortModule: Promise<typeof import("onnxruntime-web/wasm")> | null = null;

async function getOrt(wasmBase: string): Promise<typeof import("onnxruntime-web/wasm")> {
  if (!ortModule) {
    ortModule = import("onnxruntime-web/wasm").then((ort) => {
      ort.env.wasm.wasmPaths = wasmBase;
      // GitHub Pages cannot send COOP/COEP, so no SharedArrayBuffer / threads.
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      return ort;
    });
  }
  return ortModule;
}

const sessions = new Map<string, Promise<InferenceSession>>();
const evalCaches = new Map<string, Map<number, { priors: number[]; value: number }>>();
const MAX_CACHE = 5000;

async function getSession(level: AzLevel): Promise<InferenceSession> {
  let s = sessions.get(level.model);
  if (!s) {
    s = getOrt(level.wasmBase).then((ort) =>
      ort.InferenceSession.create(level.model, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
    );
    sessions.set(level.model, s);
  }
  return s;
}

/** Masked softmax over legal moves, in float64 from the float32 logits. */
function policyFromLogits(logits: ArrayLike<number>, board: Board): number[] {
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
  return sum ? p.map((v) => v / sum) : new Array(WIDTH).fill(1 / WIDTH);
}

async function evaluate(level: AzLevel, board: Board): Promise<{ priors: number[]; value: number }> {
  let cache = evalCaches.get(level.model);
  if (!cache) {
    cache = new Map();
    evalCaches.set(level.model, cache);
  }
  const key = board.key();
  const hit = cache.get(key);
  if (hit) return hit;

  const ort = await getOrt(level.wasmBase);
  const session = await getSession(level);
  const tensor = new ort.Tensor("float32", board.toPlanes(), [1, 2, 6, 7]);
  const out = await session.run({ board: tensor });
  const logits = out.policy_logits.data as Float32Array;
  const value = (out.value.data as Float32Array)[0];
  const res = { priors: policyFromLogits(logits, board), value };
  if (cache.size >= MAX_CACHE) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  cache.set(key, res);
  return res;
}

const evaluatorFor = (level: AzLevel): AzEvaluator => ({
  evaluate: (board) => evaluate(level, board),
});

export interface AzResult {
  move: number;
  analysis: {
    priors: number[];
    value: number;
    visits?: number[];
    sims: number;
    source: "az";
  };
}

/** Pick a move and return the display data for the same search. */
export async function azSearch(
  level: AzLevel,
  board: Board,
  seed: number,
  temperature = level.temperature,
): Promise<AzResult> {
  let s = seed >>> 0;
  const rng = (): number => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  if (level.sims <= 0) {
    const { priors, value } = await evaluate(level, board);
    return {
      move: chooseFromPriors(priors, temperature, rng),
      analysis: { priors, value, sims: 0, source: "az" },
    };
  }

  const root = newRoot(board);
  await runSearch(root, board, evaluatorFor(level), { sims: level.sims, cPuct: 2.0, fpu: 0.0 });
  return {
    move: chooseMove(root, temperature, rng),
    analysis: {
      priors: root.visitDistribution(),
      value: root.value(),
      visits: [...root.N],
      sims: level.sims,
      source: "az",
    },
  };
}

/** Network-only evaluation (used for the bar while the human is thinking). */
export async function azAnalyse(level: AzLevel, board: Board): Promise<AzResult["analysis"]> {
  const { priors, value } = await evaluate(level, board);
  return { priors, value, sims: 0, source: "az" };
}
