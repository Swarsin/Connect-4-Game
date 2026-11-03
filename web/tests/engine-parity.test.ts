import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { Board, CELLS, IllegalMoveError } from "../src/engine/board";

interface Case {
  moves: string;
  legal: number[];
  terminal: boolean;
  winner: 0 | 1 | 2 | null;
  to_move: 1 | 2;
  wins_me: number[];
  wins_opp: number[];
  planes: string;
}

const fixture = JSON.parse(
  readFileSync(new URL("../../data/fixtures/engine_cases.json", import.meta.url), "utf8"),
) as { cases: Case[] };

function planesString(b: Board): string {
  return Array.from(b.toPlanes(), (v) => (v ? 1 : 0)).join("");
}

describe("engine parity with Python", () => {
  it(`matches all ${fixture.cases.length} shared positions`, () => {
    for (const c of fixture.cases) {
      const b = Board.fromMoves(c.moves);
      expect(b.toMoves(), c.moves).toBe(c.moves);
      expect(b.legalMoves(), `legal ${c.moves}`).toEqual(c.legal);
      expect(b.isTerminal(), `terminal ${c.moves}`).toBe(c.terminal);
      expect(b.winner(), `winner ${c.moves}`).toBe(c.winner);
      expect(b.playerToMove(), `turn ${c.moves}`).toBe(c.to_move);
      expect(b.winningMoves(true), `win ${c.moves}`).toEqual(c.wins_me);
      expect(b.winningMoves(false), `block ${c.moves}`).toEqual(c.wins_opp);
      expect(planesString(b), `planes ${c.moves}`).toBe(c.planes);
    }
  });
});

describe("engine basics", () => {
  it("detects every win direction and colours", () => {
    expect(Board.fromMoves("1213141").winner()).toBe(1); // vertical
    expect(Board.fromMoves("7676767").winner()).toBe(1); // vertical at the edge
    expect(Board.fromMoves("1122334").winner()).toBe(1); // horizontal
    expect(Board.fromMoves("12233434414").winner()).toBe(1); // diagonal /
    expect(Board.fromMoves("76655454474").winner()).toBe(1); // diagonal \
    expect(Board.fromMoves("17273757").winner()).toBe(2); // vertical, player 2
  });

  it("returns the winning four cells", () => {
    const line = Board.fromMoves("1122334").winningLine();
    expect(line).toEqual([
      [0, 0],
      [0, 1],
      [0, 2],
      [0, 3],
    ]);
  });

  it("detects a full-board draw", () => {
    const b = Board.fromMoves("472424756645422154157737216635626711335331");
    expect(b.isDraw()).toBe(true);
    expect(b.winner()).toBe(0);
    expect(b.isTerminal()).toBe(true);
    expect(b.legalMoves()).toEqual([]);
  });

  it("rejects illegal moves and undoes exactly", () => {
    expect(() => Board.fromMoves("1111111")).toThrow(IllegalMoveError);
    expect(() => new Board().undo()).toThrow(IllegalMoveError);
    const b = Board.fromMoves("4453");
    const snapshot = {
      cells: Array.from(b.cells),
      heights: Array.from(b.heights),
      n: b.n,
      moves: b.toMoves(),
      key: b.key(),
      lastRow: b.lastRow,
      lastCol: b.lastCol,
    };
    b.play(2);
    b.undo();
    expect(Array.from(b.cells)).toEqual(snapshot.cells);
    expect(Array.from(b.heights)).toEqual(snapshot.heights);
    expect(b.n).toBe(snapshot.n);
    expect(b.toMoves()).toBe(snapshot.moves);
    expect(b.key()).toBe(snapshot.key);
    expect(b.lastRow).toBe(snapshot.lastRow);
    expect(b.lastCol).toBe(snapshot.lastCol);
  });

  it("clones independently", () => {
    const a = Board.fromMoves("4453");
    const b = a.clone();
    b.play(3);
    expect(a.toMoves()).toBe("4453");
    expect(b.toMoves()).toBe("44534");
  });

  it("encodes planes with row 0 at the bottom", () => {
    const p = Board.fromMoves("1").toPlanes();
    expect(p.length).toBe(2 * CELLS);
    // after player 1 moves, player 2 is to move: the stone is on the opponent channel
    expect(p[0]).toBe(0);
    expect(p[CELLS]).toBe(1);
    const q = Board.fromMoves("12").toPlanes();
    expect(q[1]).toBe(0); // player 2 stone is now the opponent
    expect(q[CELLS + 1]).toBe(1);
  });

  it("finds immediate wins for both sides", () => {
    const b = Board.fromMoves("121212");
    expect(b.winningMoves(true)).toEqual([0]);
    expect(b.winningMoves(false)).toEqual([1]);
  });
});
