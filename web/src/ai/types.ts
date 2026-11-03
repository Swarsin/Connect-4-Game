import type { Board } from "../engine/board";

export interface Agent {
  readonly name: string;
  reset(seed?: number): void;
  selectMove(board: Board): number;
}

export interface AgentAnalysis {
  visits?: number[];
  q?: Array<number | null>;
  priors?: number[];
  value?: number;
  score?: number[];
}

/** Deterministic 32-bit PRNG (mulberry32). */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function pickIndex(n: number, rng: () => number): number {
  return Math.min(n - 1, Math.floor(rng() * n));
}
