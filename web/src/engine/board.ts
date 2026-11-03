/**
 * Connect 4 rules engine — TypeScript port of src/c4/engine/board.py.
 *
 * Cells are stored row-major with row 0 at the BOTTOM, matching docs/conventions.md.
 * The shared fixture data/fixtures/engine_cases.json must match this implementation exactly.
 */

export const WIDTH = 7;
export const HEIGHT = 6;
export const CELLS = WIDTH * HEIGHT;
export const CENTRE_ORDER = [3, 2, 4, 1, 5, 0, 6] as const;

export type Player = 1 | 2;
export type Cell = 0 | 1 | 2;

export class IllegalMoveError extends Error {}

/** Deterministic 32-bit PRNG so Zobrist keys are stable across sessions. */
function splitmix32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x9e3779b9) >>> 0;
    let t = a;
    t ^= t >>> 16;
    t = Math.imul(t, 0x21f0aaad) >>> 0;
    t ^= t >>> 15;
    t = Math.imul(t, 0x735a2d97) >>> 0;
    return (t ^ (t >>> 15)) >>> 0;
  };
}

const rand = splitmix32(0xc0ffee);
const Z1: number[] = [];
const Z2: number[] = [];
for (let i = 0; i < CELLS; i++) {
  for (let p = 0; p < 3; p++) {
    Z1.push(rand());
    Z2.push(rand());
  }
}

export class Board {
  readonly cells: Int8Array;
  readonly heights: Uint8Array;
  n = 0;
  readonly history: number[] = [];
  lastRow = -1;
  lastCol = -1;
  private h1 = 0;
  private h2 = 0;

  constructor() {
    this.cells = new Int8Array(CELLS);
    this.heights = new Uint8Array(WIDTH);
  }

  static fromMoves(moves: string | readonly number[]): Board {
    const b = new Board();
    const cols = typeof moves === "string" ? [...moves].map((ch) => Number(ch) - 1) : moves;
    for (const c of cols) b.play(c);
    return b;
  }

  clone(): Board {
    const b = new Board();
    b.cells.set(this.cells);
    b.heights.set(this.heights);
    b.n = this.n;
    b.history.push(...this.history);
    b.lastRow = this.lastRow;
    b.lastCol = this.lastCol;
    b.h1 = this.h1;
    b.h2 = this.h2;
    return b;
  }

  toMoves(): string {
    return this.history.map((c) => c + 1).join("");
  }

  playerToMove(): Player {
    return this.n % 2 === 0 ? 1 : 2;
  }

  cell(row: number, col: number): Cell {
    return this.cells[row * WIDTH + col] as Cell;
  }

  height(col: number): number {
    return this.heights[col];
  }

  isLegal(col: number): boolean {
    return Number.isInteger(col) && col >= 0 && col < WIDTH && this.heights[col] < HEIGHT && !this.isTerminal();
  }

  legalMoves(): number[] {
    if (this.isTerminal()) return [];
    const out: number[] = [];
    for (let c = 0; c < WIDTH; c++) if (this.heights[c] < HEIGHT) out.push(c);
    return out;
  }

  play(col: number): void {
    if (!this.isLegal(col)) {
      throw new IllegalMoveError(`illegal move ${col} in position "${this.toMoves()}"`);
    }
    this.playUnchecked(col);
  }

  playUnchecked(col: number): void {
    const row = this.heights[col];
    const p = this.playerToMove();
    const idx = row * WIDTH + col;
    this.cells[idx] = p;
    this.heights[col] = row + 1;
    this.n += 1;
    this.history.push(col);
    this.lastRow = row;
    this.lastCol = col;
    this.h1 = (this.h1 ^ Z1[idx * 3 + p]) >>> 0;
    this.h2 = (this.h2 ^ Z2[idx * 3 + p]) >>> 0;
  }

  undo(): number {
    if (this.n === 0) throw new IllegalMoveError("nothing to undo");
    const col = this.history.pop() as number;
    const row = this.heights[col] - 1;
    const idx = row * WIDTH + col;
    const p = this.cells[idx] as Player;
    this.cells[idx] = 0;
    this.heights[col] = row;
    this.n -= 1;
    this.h1 = (this.h1 ^ Z1[idx * 3 + p]) >>> 0;
    this.h2 = (this.h2 ^ Z2[idx * 3 + p]) >>> 0;
    if (this.n === 0) {
      this.lastRow = -1;
      this.lastCol = -1;
    } else {
      const prev = this.history[this.history.length - 1];
      this.lastRow = this.heights[prev] - 1;
      this.lastCol = prev;
    }
    return col;
  }

  /** The four-cell run through the last move, if any. */
  winningLine(): Array<[number, number]> | null {
    if (this.lastRow < 0) return null;
    const p = this.cells[this.lastRow * WIDTH + this.lastCol];
    if (!p) return null;
    const dirs: Array<[number, number]> = [
      [0, 1],
      [1, 0],
      [1, 1],
      [-1, 1],
    ];
    for (const [dr, dc] of dirs) {
      const line: Array<[number, number]> = [[this.lastRow, this.lastCol]];
      for (const sign of [1, -1]) {
        let r = this.lastRow + dr * sign;
        let c = this.lastCol + dc * sign;
        while (r >= 0 && r < HEIGHT && c >= 0 && c < WIDTH && this.cell(r, c) === p) {
          line.push([r, c]);
          r += dr * sign;
          c += dc * sign;
        }
      }
      if (line.length >= 4) {
        line.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
        return line.slice(0, 4);
      }
    }
    return null;
  }

  lastMoveWon(): boolean {
    if (this.lastRow < 0) return false;
    const p = this.cells[this.lastRow * WIDTH + this.lastCol];
    if (!p) return false;
    const dirs: Array<[number, number]> = [
      [0, 1],
      [1, 0],
      [1, 1],
      [-1, 1],
    ];
    for (const [dr, dc] of dirs) {
      let count = 1;
      for (const sign of [1, -1]) {
        let r = this.lastRow + dr * sign;
        let c = this.lastCol + dc * sign;
        while (r >= 0 && r < HEIGHT && c >= 0 && c < WIDTH && this.cell(r, c) === p) {
          count++;
          r += dr * sign;
          c += dc * sign;
        }
      }
      if (count >= 4) return true;
    }
    return false;
  }

  isDraw(): boolean {
    return this.n === CELLS && !this.lastMoveWon();
  }

  isTerminal(): boolean {
    return this.n === CELLS || this.lastMoveWon();
  }

  /** 1 or 2 for a win, 0 for a draw, null while the game is running. */
  winner(): Player | 0 | null {
    if (this.lastMoveWon()) return this.n % 2 === 1 ? 1 : 2;
    if (this.n === CELLS) return 0;
    return null;
  }

  /** Does a stone for the side to move in `col` complete four? */
  wouldWin(col: number): boolean {
    const row = this.heights[col];
    if (row >= HEIGHT) return false;
    const p = this.playerToMove();
    const get = (r: number, c: number): number =>
      r >= 0 && r < HEIGHT && c >= 0 && c < WIDTH
        ? r === row && c === col
          ? p
          : this.cell(r, c)
        : 0;
    const dirs: Array<[number, number]> = [
      [0, 1],
      [1, 0],
      [1, 1],
      [-1, 1],
    ];
    for (const [dr, dc] of dirs) {
      let count = 1;
      for (const sign of [1, -1]) {
        let r = row + dr * sign;
        let c = col + dc * sign;
        while (get(r, c) === p) {
          count++;
          r += dr * sign;
          c += dc * sign;
        }
      }
      if (count >= 4) return true;
    }
    return false;
  }

  /** Columns where the side to move (or the opponent) would win immediately. */
  winningMoves(forSideToMove = true): number[] {
    if (this.isTerminal()) return [];
    const out: number[] = [];
    for (let c = 0; c < WIDTH; c++) {
      if (this.heights[c] >= HEIGHT) continue;
      if (forSideToMove) {
        if (this.wouldWin(c)) out.push(c);
      } else if (this.opponentWouldWin(c)) out.push(c);
    }
    return out;
  }

  /** Would the opponent win by playing `col` next turn? */
  opponentWouldWin(col: number): boolean {
    const row = this.heights[col];
    if (row >= HEIGHT) return false;
    const opp = this.playerToMove() === 1 ? 2 : 1;
    const get = (r: number, c: number): number =>
      r >= 0 && r < HEIGHT && c >= 0 && c < WIDTH
        ? r === row && c === col
          ? opp
          : this.cell(r, c)
        : 0;
    const dirs: Array<[number, number]> = [
      [0, 1],
      [1, 0],
      [1, 1],
      [-1, 1],
    ];
    for (const [dr, dc] of dirs) {
      let count = 1;
      for (const sign of [1, -1]) {
        let r = row + dr * sign;
        let c = col + dc * sign;
        while (get(r, c) === opp) {
          count++;
          r += dr * sign;
          c += dc * sign;
        }
      }
      if (count >= 4) return true;
    }
    return false;
  }

  /** `[2, 6, 7]` planes flattened: channel 0 = side to move, 1 = opponent, row 0 = bottom. */
  toPlanes(out = new Float32Array(2 * CELLS)): Float32Array {
    const me = this.playerToMove();
    for (let r = 0; r < HEIGHT; r++) {
      for (let c = 0; c < WIDTH; c++) {
        const v = this.cells[r * WIDTH + c];
        const idx = r * WIDTH + c;
        out[idx] = v === me ? 1 : 0;
        out[CELLS + idx] = v !== 0 && v !== me ? 1 : 0;
      }
    }
    return out;
  }

  /** Transposition key (two Zobrist halves packed into a safe integer). */
  key(): number {
    return (this.h1 >>> 0) * 2097152 + (this.h2 & 0x1fffff);
  }

  toString(): string {
    const sym: Record<number, string> = { 0: ".", 1: "X", 2: "O" };
    const rows: string[] = [];
    for (let r = HEIGHT - 1; r >= 0; r--) {
      const row: string[] = [];
      for (let c = 0; c < WIDTH; c++) row.push(sym[this.cell(r, c)]);
      rows.push(row.join(" "));
    }
    rows.push("1 2 3 4 5 6 7");
    return rows.join("\n");
  }
}

export function cellCount(): number {
  return CELLS;
}
