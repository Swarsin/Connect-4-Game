import { Board, CENTRE_ORDER, CELLS, HEIGHT, WIDTH } from "../engine/board";
import { makeRng, pickIndex, type Agent } from "./types";

const WIN = 1_000_000;
const EXACT = 0;
const LOWER = 1;
const UPPER = 2;

interface Entry {
  d: number;
  v: number;
  flag: number;
  move: number;
}

/** All 69 windows of four cells, as row/col pairs. */
const WINDOWS: Array<Array<[number, number]>> = (() => {
  const out: Array<Array<[number, number]>> = [];
  for (let r = 0; r < HEIGHT; r++) {
    for (let c = 0; c < WIDTH; c++) {
      for (const [dr, dc] of [
        [0, 1],
        [1, 0],
        [1, 1],
        [-1, 1],
      ] as const) {
        const cells: Array<[number, number]> = [];
        for (let i = 0; i < 4; i++) {
          const rr = r + i * dr;
          const cc = c + i * dc;
          if (rr < 0 || rr >= HEIGHT || cc < 0 || cc >= WIDTH) break;
          cells.push([rr, cc]);
        }
        if (cells.length === 4) out.push(cells);
      }
    }
  }
  return out;
})();

/** Symmetric static evaluation from the side to move's perspective. */
export function evaluate(b: Board): number {
  const me = b.playerToMove();
  const opp = me === 1 ? 2 : 1;
  let score = 3 * (countInColumn(b, 3, me) - countInColumn(b, 3, opp));
  for (const w of WINDOWS) {
    let a = 0;
    let o = 0;
    for (const [r, c] of w) {
      const v = b.cell(r, c);
      if (v === me) a++;
      else if (v === opp) o++;
    }
    if (a && o) continue;
    if (a === 3) score += 5;
    else if (a === 2) score += 2;
    else if (o === 3) score -= 4;
    else if (o === 2) score -= 2;
  }
  return score;
}

function countInColumn(b: Board, col: number, player: number): number {
  let n = 0;
  for (let r = 0; r < b.height(col); r++) if (b.cell(r, col) === player) n++;
  return n;
}

export class MinimaxAgent implements Agent {
  readonly name: string;
  private rng: () => number;
  private tt = new Map<number, Entry>();
  nodes = 0;

  constructor(
    public depth = 4,
    seed = 0,
    public useTt = true,
  ) {
    this.name = `minimax:d=${depth}`;
    this.rng = makeRng(seed);
  }

  reset(seed = 0): void {
    this.rng = makeRng(seed);
    this.tt.clear();
  }

  private negamax(b: Board, depth: number, alpha: number, beta: number): number {
    this.nodes++;
    if (b.lastMoveWon()) return -(WIN + depth);
    if (b.n === CELLS) return 0;
    if (depth === 0) return evaluate(b);
    if (b.winningMoves(true).length) return WIN + depth - 1;

    const alpha0 = alpha;
    const key = b.key();
    let ttMove = -1;
    if (this.useTt) {
      const e = this.tt.get(key);
      if (e && e.d >= depth) {
        if (e.flag === EXACT) return e.v;
        if (e.flag === LOWER) alpha = Math.max(alpha, e.v);
        else if (e.flag === UPPER) beta = Math.min(beta, e.v);
        if (alpha >= beta) return e.v;
      }
      if (e) ttMove = e.move;
    }

    const moves: number[] = [];
    for (const c of CENTRE_ORDER) if (b.height(c) < HEIGHT) moves.push(c);
    if (ttMove >= 0) {
      const i = moves.indexOf(ttMove);
      if (i > 0) {
        moves.splice(i, 1);
        moves.unshift(ttMove);
      }
    }

    let best = -Infinity;
    let bestMove = moves[0] ?? 0;
    for (const c of moves) {
      b.playUnchecked(c);
      const v = -this.negamax(b, depth - 1, -beta, -alpha);
      b.undo();
      if (v > best) {
        best = v;
        bestMove = c;
      }
      alpha = Math.max(alpha, v);
      if (alpha >= beta) break;
    }

    if (this.useTt) {
      const flag = best <= alpha0 ? UPPER : best >= beta ? LOWER : EXACT;
      this.tt.set(key, { d: depth, v: best, flag, move: bestMove });
    }
    return best;
  }

  moveScores(board: Board): Map<number, number> {
    if (this.tt.size > 500_000) this.tt.clear();
    const b = board.clone();
    const scores = new Map<number, number>();
    let best = -Infinity;
    const moves: number[] = [];
    for (const c of CENTRE_ORDER) if (b.height(c) < HEIGHT) moves.push(c);
    for (const c of moves) {
      b.playUnchecked(c);
      const v = -this.negamax(b, this.depth - 1, -Infinity, -(best - 1));
      b.undo();
      scores.set(c, v);
      best = Math.max(best, v);
    }
    return scores;
  }

  selectMove(board: Board): number {
    const scores = this.moveScores(board);
    let best = -Infinity;
    for (const v of scores.values()) best = Math.max(best, v);
    const ties = [...scores.entries()].filter(([, v]) => v === best).map(([c]) => c).sort((a, b) => a - b);
    return ties[pickIndex(ties.length, this.rng)];
  }

  /**
   * Exact score for every legal move (full window, so no null-window bounds leak into the UI).
   * Capped at depth 6: the bar has to stay responsive even when the bot plays at depth 8.
   */
  analyze(board: Board): { score: Array<number | null>; depth: number } {
    if (this.tt.size > 500_000) this.tt.clear();
    const depth = Math.max(1, Math.min(this.depth, 6));
    const b = board.clone();
    const scores: Array<number | null> = new Array(WIDTH).fill(null);
    for (const c of CENTRE_ORDER) {
      if (b.height(c) >= HEIGHT) continue;
      b.playUnchecked(c);
      scores[c] = -this.negamax(b, depth - 1, -Infinity, Infinity);
      b.undo();
    }
    return { score: scores, depth };
  }
}
