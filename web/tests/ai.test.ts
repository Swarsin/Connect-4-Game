import { describe, expect, it } from "vitest";

import { Board } from "../src/engine/board";
import { MinimaxAgent } from "../src/ai/minimax";
import { UctAgent } from "../src/ai/uct";
import { makeRng, pickIndex, type Agent } from "../src/ai/types";

class RandomAgent implements Agent {
  readonly name = "random";
  private rng = makeRng(1);
  reset(seed = 1): void {
    this.rng = makeRng(seed);
  }
  selectMove(b: Board): number {
    const moves = b.legalMoves();
    return moves[pickIndex(moves.length, this.rng)];
  }
}

function randomPosition(rng: () => number, minPly = 6): Board {
  for (;;) {
    const b = new Board();
    const target = minPly + Math.floor(rng() * 24);
    while (!b.isTerminal() && b.n < target) {
      const moves = b.legalMoves();
      b.play(moves[pickIndex(moves.length, rng)]);
    }
    if (!b.isTerminal()) return b;
  }
}

function tactical(kind: "win" | "block", count: number, seed: number): Board[] {
  const rng = makeRng(seed);
  const out: Board[] = [];
  let guard = 0;
  while (out.length < count && guard++ < 20_000) {
    const b = randomPosition(rng);
    const mine = b.winningMoves(true);
    const theirs = b.winningMoves(false);
    if (kind === "win" && mine.length) out.push(b);
    if (kind === "block" && mine.length === 0 && theirs.length === 1) out.push(b);
  }
  return out;
}

function play(a: Agent, b: Agent, opening = ""): 0 | 1 | 2 {
  const board = Board.fromMoves(opening);
  const agents: Record<number, Agent> = { 1: a, 2: b };
  while (!board.isTerminal()) board.play(agents[board.playerToMove()].selectMove(board));
  return board.winner() as 0 | 1 | 2;
}

describe("minimax", () => {
  it("always takes an immediate win at depth 2", () => {
    const agent = new MinimaxAgent(2, 0);
    for (const b of tactical("win", 40, 11)) {
      expect(b.winningMoves(true)).toContain(agent.selectMove(b));
    }
  });

  it("always blocks a single immediate threat at depth 2", () => {
    const agent = new MinimaxAgent(2, 0);
    for (const b of tactical("block", 40, 12)) {
      expect(agent.selectMove(b)).toBe(b.winningMoves(false)[0]);
    }
  });

  it("does not return stale mate scores from a reused transposition table", () => {
    const line = ["37611143156552217767", "376111431565522177674", "3761114315655221776745"];
    const tt = new MinimaxAgent(4, 0, true);
    const nt = new MinimaxAgent(4, 0, false);
    for (const moves of [...line].reverse()) {
      const b = Board.fromMoves(moves);
      expect(tt.moveScores(b)).toEqual(nt.moveScores(b));
    }
  });

  it("beats random at depth 4", () => {
    const mm = new MinimaxAgent(4, 0);
    const rnd = new RandomAgent();
    let wins = 0;
    const openings = ["", "4", "3", "5", "2", "6"];
    for (let i = 0; i < 20; i++) {
      const opening = openings[i % openings.length];
      mm.reset(i);
      rnd.reset(i + 100);
      const w = i % 2 === 0 ? play(mm, rnd, opening) : play(rnd, mm, opening);
      if ((i % 2 === 0 && w === 1) || (i % 2 === 1 && w === 2)) wins++;
    }
    expect(wins).toBeGreaterThanOrEqual(18);
  });
});

describe("uct", () => {
  it("takes wins and blocks with 1500 iterations", () => {
    const agent = new UctAgent(1500, Math.SQRT2, 0, true);
    const wins = tactical("win", 15, 21);
    const blocks = tactical("block", 15, 22);
    let okW = 0;
    let okB = 0;
    for (const b of wins) {
      agent.reset(7);
      if (b.winningMoves(true).includes(agent.selectMove(b))) okW++;
    }
    for (const b of blocks) {
      agent.reset(8);
      if (agent.selectMove(b) === b.winningMoves(false)[0]) okB++;
    }
    expect(okW / wins.length).toBeGreaterThanOrEqual(0.9);
    expect(okB / blocks.length).toBeGreaterThanOrEqual(0.9);
  });

  it("beats random with 800 iterations", () => {
    const uct = new UctAgent(800, Math.SQRT2, 0);
    const rnd = new RandomAgent();
    let wins = 0;
    for (let i = 0; i < 4; i++) {
      uct.reset(i);
      rnd.reset(i + 50);
      const w = i % 2 === 0 ? play(uct, rnd, "4") : play(rnd, uct, "4");
      if ((i % 2 === 0 && w === 1) || (i % 2 === 1 && w === 2)) wins++;
    }
    expect(wins).toBe(4);
  });
});
