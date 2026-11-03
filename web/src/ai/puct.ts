/**
 * PUCT search — TypeScript port of src/c4/alphazero/mcts.py (single tree, one leaf per
 * simulation). Kept algorithmically identical to the Python reference so the shared
 * `search_cases.json` fixture can be used as a parity test.
 */

import { Board, CELLS, WIDTH } from "../engine/board";

export interface AzEvaluator {
  evaluate(board: Board): Promise<{ priors: ArrayLike<number>; value: number }>;
}

export interface PuctParams {
  sims: number;
  cPuct?: number;
  fpu?: number;
}

export class PuctNode {
  P: number[] = new Array(WIDTH).fill(0);
  N: number[] = new Array(WIDTH).fill(0);
  W: number[] = new Array(WIDTH).fill(0);
  children: Array<PuctNode | null> = new Array(WIDTH).fill(null);
  legal: number[] = [];
  expanded = false;
  terminal = false;
  tv = 0;
  n = 0;

  q(a: number): number {
    return this.N[a] ? this.W[a] / this.N[a] : 0;
  }

  value(): number {
    if (!this.n) return 0;
    let sum = 0;
    for (let a = 0; a < WIDTH; a++) sum += this.W[a];
    return sum / this.n;
  }

  visitDistribution(): number[] {
    const total = this.N.reduce((a, b) => a + b, 0);
    return total ? this.N.map((v) => v / total) : new Array(WIDTH).fill(0);
  }
}

export function newRoot(board: Board): PuctNode {
  const node = new PuctNode();
  if (board.isTerminal()) {
    node.terminal = true;
    node.tv = board.lastMoveWon() ? -1 : 0;
  } else {
    node.legal = board.legalMoves();
  }
  return node;
}

function select(node: PuctNode, cPuct: number, fpu: number): number {
  const sq = cPuct * Math.sqrt(node.n + 1);
  let bestA = -1;
  let bestS = -Infinity;
  for (const a of node.legal) {
    const n = node.N[a];
    const q = n ? node.W[a] / n : fpu;
    const s = q + (sq * node.P[a]) / (1 + n);
    if (s > bestS) {
      bestS = s;
      bestA = a;
    }
  }
  return bestA;
}

function backup(path: Array<[PuctNode, number]>, v: number): void {
  for (let i = path.length - 1; i >= 0; i--) {
    v = -v;
    const [node, a] = path[i];
    node.N[a] += 1;
    node.W[a] += v;
    node.n += 1;
  }
}

export async function runSearch(
  root: PuctNode,
  board: Board,
  evaluator: AzEvaluator,
  params: PuctParams,
): Promise<void> {
  const cPuct = params.cPuct ?? 2.0;
  const fpu = params.fpu ?? 0.0;

  if (!root.expanded && !root.terminal) {
    const { priors } = await evaluator.evaluate(board);
    root.P = Array.from(priors);
    root.expanded = true;
  }

  for (let i = 0; i < params.sims; i++) {
    if (root.terminal) return;
    let node = root;
    const b = board.clone();
    const path: Array<[PuctNode, number]> = [];
    for (;;) {
      const a = select(node, cPuct, fpu);
      path.push([node, a]);
      b.playUnchecked(a);
      let child = node.children[a];
      if (!child) {
        child = new PuctNode();
        node.children[a] = child;
        if (b.lastMoveWon()) {
          child.terminal = true;
          child.tv = -1;
        } else if (b.n === CELLS) {
          child.terminal = true;
          child.tv = 0;
        } else {
          child.legal = b.legalMoves();
        }
      }
      node = child;
      if (node.terminal || !node.expanded) break;
    }

    if (node.terminal) {
      backup(path, node.tv);
    } else {
      const { priors, value } = await evaluator.evaluate(b);
      node.P = Array.from(priors);
      node.expanded = true;
      backup(path, value);
    }
  }
}

export function chooseMove(root: PuctNode, temperature: number, rng: () => number): number {
  let counts = root.N;
  if (counts.reduce((a, b) => a + b, 0) === 0) counts = root.P;
  if (temperature <= 1e-6) {
    let best = 0;
    for (let c = 1; c < WIDTH; c++) if (counts[c] > counts[best]) best = c;
    return best;
  }
  const p = counts.map((c) => Math.pow(c, 1 / temperature));
  const total = p.reduce((a, b) => a + b, 0);
  let r = rng() * total;
  for (let c = 0; c < WIDTH; c++) {
    r -= p[c];
    if (r <= 0) return c;
  }
  return WIDTH - 1;
}

export function chooseFromPriors(priors: ArrayLike<number>, temperature: number, rng: () => number): number {
  if (temperature <= 1e-6) {
    let best = 0;
    for (let c = 1; c < WIDTH; c++) if (priors[c] > priors[best]) best = c;
    return best;
  }
  const p = Array.from(priors, (v) => Math.pow(v, 1 / temperature));
  const total = p.reduce((a, b) => a + b, 0);
  let r = rng() * total;
  for (let c = 0; c < WIDTH; c++) {
    r -= p[c];
    if (r <= 0) return c;
  }
  return WIDTH - 1;
}
