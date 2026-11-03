import { describe, expect, it } from "vitest";

import {
  MATE,
  normaliseAnalysis,
  scoreToProbability,
  scoresToPriors,
} from "../src/ai/analysis";
import { MinimaxAgent } from "../src/ai/minimax";
import { UctAgent } from "../src/ai/uct";
import { Board } from "../src/engine/board";

describe("score mapping", () => {
  it("is 50% at equality and monotonic", () => {
    expect(scoreToProbability(0)).toBeCloseTo(0.5, 6);
    const values = [-12, -6, -2, 0, 2, 5, 10, 20].map(scoreToProbability);
    for (let i = 1; i < values.length; i++) expect(values[i]).toBeGreaterThan(values[i - 1]);
  });

  it("moves away from 50% on ordinary heuristic scores", () => {
    // the old bug: scale 250 kept everything at 49-51%
    expect(scoreToProbability(5)).toBeGreaterThan(0.7);
    expect(scoreToProbability(10)).toBeGreaterThan(0.85);
    expect(scoreToProbability(-6)).toBeLessThan(0.3);
    expect(scoreToProbability(2)).toBeGreaterThan(0.55);
  });

  it("treats mate scores as certain", () => {
    expect(scoreToProbability(MATE + 3)).toBe(1);
    expect(scoreToProbability(-MATE - 3)).toBe(0);
  });

  it("spreads priors instead of flattening them", () => {
    const priors = scoresToPriors([-6, -4, -1, 0, -1, -4, -6]);
    expect(priors.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(priors[3]).toBeGreaterThan(priors[0]);
    expect(priors[3]).toBeGreaterThan(0.2);
    expect(new Set(priors.map((p) => p.toFixed(3))).size).toBeGreaterThan(2);
  });

  it("gives a mate move essentially all the prior mass", () => {
    const priors = scoresToPriors([Math.max(-MATE, -MATE), MATE, -2, -2, -2, -2, -2]);
    expect(priors[1]).toBeGreaterThan(0.99);
  });
});

describe("normaliseAnalysis", () => {
  it("uses minimax scores for value and priors", () => {
    const a = normaliseAnalysis("minimax:d=4", {
      score: [-6, -4, -1, 0, -1, -4, -6],
      depth: 4,
    });
    expect(a.source).toBe("minimax");
    expect(a.value).toBeCloseTo(0.5, 6);
    expect(a.evalScore).toBe(0);
    expect(a.depth).toBe(4);
    expect(a.priors[3]).toBeGreaterThan(a.priors[0]);
  });

  it("reports a forced win as certain", () => {
    const a = normaliseAnalysis("minimax:d=4", {
      score: [MATE + 3, 0, -MATE, -MATE, -MATE, -MATE, -MATE],
      depth: 4,
    });
    expect(a.value).toBe(1);
    expect(a.priors[0]).toBeGreaterThan(0.99);
  });

  it("keeps the MCTS side-to-move perspective (not inverted)", () => {
    const a = normaliseAnalysis("uct:it=2000", {
      visits: [900, 50, 20, 10, 10, 5, 5],
      q: [0.99, 0.4, null, null, null, null, null],
      value: 0.99,
    });
    expect(a.source).toBe("mcts");
    expect(a.value).toBeCloseTo(0.99, 6);
    expect(a.sims).toBe(2000);
    expect(a.priors[0]).toBeGreaterThan(0.8);
  });

  it("falls back gracefully", () => {
    const a = normaliseAnalysis("random", undefined);
    expect(a.source).toBe("default");
    expect(a.value).toBe(0.5);
    expect(a.priors.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 6);
  });
});

describe("real agents feed the display", () => {
  it("minimax analysis is exact, capped and non-uniform on the empty board", () => {
    const agent = new MinimaxAgent(8, 0);
    const a = agent.analyze(new Board());
    expect(a.depth).toBe(6);
    const scores = a.score.filter((s): s is number => s != null);
    expect(new Set(scores).size).toBeGreaterThan(1);
    const n = normaliseAnalysis("minimax:d=8", a);
    expect(n.evalScore).toBe(Math.max(...scores));
    expect(n.priors[3]).toBeGreaterThan(0.2);
    expect(new Set(n.priors.map((p) => p.toFixed(2))).size).toBeGreaterThan(2);
  });

  it("minimax sees an immediate win as a win", () => {
    const b = Board.fromMoves("121212"); // player 1 wins with column 1
    const n = normaliseAnalysis("minimax:d=4", new MinimaxAgent(4, 0).analyze(b));
    expect(n.value).toBe(1);
    expect(n.priors[0]).toBeGreaterThan(0.99);
  });

  it("uct reports a forced win as ~1 for the side to move", () => {
    const b = Board.fromMoves("121212");
    const agent = new UctAgent(600, Math.SQRT2, 0, true);
    const n = normaliseAnalysis("uct:it=600", agent.analyze(b));
    expect(n.value).toBeGreaterThan(0.9);
    expect(n.visits?.[0]).toBeGreaterThan(0);
  });
});
