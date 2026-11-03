/**
 * Turning search output into display numbers.
 *
 * Minimax scores are heuristic window counts (roughly -20..+20), not centipawns and not
 * probabilities. We map them through a logistic with a scale that matches those units and
 * treat mate scores separately. MCTS visit counts give an empirical rollout score. Neither is
 * a solved probability — the UI labels them as estimates (the AlphaZero value head is the
 * calibrated source is the AlphaZero value head).
 */

export const MATE = 100_000;
const SCORE_SCALE = 5;
const PRIOR_SCALE = 4;

export interface RawAnalysis {
  visits?: number[];
  q?: Array<number | null>;
  score?: Array<number | null>;
  priors?: number[];
  value?: number;
  depth?: number;
  sims?: number;
  source?: "az";
}

export interface Analysis {
  priors: number[];
  value: number; // estimate for the side to move, 0..1
  visits?: number[];
  score?: Array<number | null>;
  source: "minimax" | "mcts" | "az" | "default";
  evalScore?: number;
  depth?: number;
  sims?: number;
  ms?: number;
}

const clamp01 = (x: number): number => Math.max(0, Math.min(1, x));

/** Monotonic map from a Minimax score to a win-probability estimate. */
export function scoreToProbability(score: number): number {
  if (!Number.isFinite(score)) return 0.5;
  if (score >= MATE) return 1;
  if (score <= -MATE) return 0;
  return 1 / (1 + Math.exp(-score / SCORE_SCALE));
}

/** Softmax over per-move scores, with mate scores clamped so they dominate without overflow. */
export function scoresToPriors(scores: Array<number | null>): number[] {
  const finite = scores.map((s) => (s == null || !Number.isFinite(s) ? -Infinity : s));
  const best = Math.max(...finite.filter((s) => Number.isFinite(s)));
  if (!Number.isFinite(best)) return new Array(7).fill(1 / 7);
  const z = finite.map((s) => {
    if (!Number.isFinite(s)) return -Infinity;
    return Math.max(-MATE, Math.min(MATE, s)) / PRIOR_SCALE;
  });
  const max = Math.max(...z.filter((v) => Number.isFinite(v)));
  const e = z.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => (sum ? v / sum : 0));
}

export function parseAgentSpec(spec: string): { name: string; kv: Record<string, string> } {
  const [name, ...rest] = spec.split(":");
  const kv: Record<string, string> = {};
  for (const part of rest.join(":").split(",")) {
    if (!part) continue;
    const [k, v] = part.split("=");
    if (k && v !== undefined) kv[k] = v;
  }
  return { name, kv };
}

export function normaliseAnalysis(spec: string, raw: RawAnalysis | undefined): Analysis {
  const { name, kv } = parseAgentSpec(spec);

  if (raw?.source === "az" && raw.priors) {
    return {
      priors: raw.priors,
      value: clamp01(raw.value ?? 0.5),
      visits: raw.visits,
      source: "az",
      sims: raw.sims,
    };
  }

  if (raw?.visits) {
    const visits = raw.visits.map((v) => (Number.isFinite(v) ? v : 0));
    const total = visits.reduce((a, b) => a + b, 0);
    const priors = total ? visits.map((v) => v / total) : new Array(7).fill(1 / 7);
    let best = 0;
    for (let c = 1; c < 7; c++) if (visits[c] > visits[best]) best = c;
    // UCT stores results from the perspective of the player who made the move into the node.
    // Root children were all played by the side to move, so q[best] is already its score.
    const value = raw.value ?? raw.q?.[best];
    const finalValue = Number.isFinite(value) ? value : 0.5;
    return {
      priors,
      value: clamp01(finalValue ?? 0.5),
      visits,
      source: name === "uct" ? "mcts" : "default",
      sims: Number(kv.it) || undefined,
    };
  }

  if (raw?.score) {
    const scores = raw.score;
    const best = Math.max(...scores.filter((s): s is number => s != null));
    return {
      priors: scoresToPriors(scores),
      value: scoreToProbability(best),
      score: scores,
      source: "minimax",
      evalScore: Number.isFinite(best) ? best : undefined,
      depth: raw.depth,
    };
  }

  return { priors: new Array(7).fill(1 / 7), value: 0.5, source: "default" };
}
