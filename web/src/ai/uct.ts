import { Board } from "../engine/board";
import { makeRng, pickIndex, type Agent } from "./types";

class Node {
  children: Node[] = [];
  untried: number[];
  wins = 0;
  visits = 0;
  readonly justMoved: number;

  constructor(
    public readonly move: number | null,
    public parent: Node | null,
    board: Board,
  ) {
    this.untried = board.legalMoves();
    this.justMoved = 3 - board.playerToMove();
  }

  uctChild(c: number): Node {
    const logN = Math.log(this.visits);
    let best = this.children[0];
    let bestScore = -Infinity;
    for (const ch of this.children) {
      const score = ch.wins / ch.visits + c * Math.sqrt(logN / ch.visits);
      if (score > bestScore) {
        bestScore = score;
        best = ch;
      }
    }
    return best;
  }
}

export class UctAgent implements Agent {
  readonly name: string;
  private rng: () => number;
  private root: Node | null = null;
  private rootMoves = "";

  constructor(
    public iterations = 2000,
    public c = Math.SQRT2,
    seed = 0,
    public smartRollouts = false,
    public reuseTree = true,
  ) {
    this.name = `mcts:it=${iterations}`;
    this.rng = makeRng(seed);
  }

  reset(seed = 0): void {
    this.rng = makeRng(seed);
    this.root = null;
    this.rootMoves = "";
  }

  private rollout(b: Board): number {
    while (!b.isTerminal()) {
      const moves = b.legalMoves();
      if (this.smartRollouts) {
        const win = b.winningMoves(true);
        if (win.length) {
          b.playUnchecked(win[0]);
          break;
        }
        const block = b.winningMoves(false);
        if (block.length) {
          b.playUnchecked(block[0]);
          continue;
        }
      }
      b.playUnchecked(moves[pickIndex(moves.length, this.rng)]);
    }
    return b.winner() as number;
  }

  private getRoot(board: Board): Node {
    const moves = board.toMoves();
    let root = this.root;
    if (this.reuseTree && root && moves.startsWith(this.rootMoves)) {
      const extra = moves.slice(this.rootMoves.length);
      for (const ch of extra) {
        if (!root) break;
        const col = Number(ch) - 1;
        const next: Node | undefined = root.children.find((n: Node) => n.move === col);
        if (!next) {
          root = null;
          break;
        }
        next.parent = null;
        root = next;
      }
      if (root) {
        this.root = root;
        this.rootMoves = moves;
        return root;
      }
    }
    this.root = new Node(null, null, board);
    this.rootMoves = moves;
    return this.root;
  }

  search(board: Board): Node {
    const root = this.getRoot(board);
    for (let i = 0; i < this.iterations; i++) {
      let node = root;
      const b = board.clone();
      while (node.untried.length === 0 && node.children.length > 0) {
        node = node.uctChild(this.c);
        b.playUnchecked(node.move as number);
      }
      if (node.untried.length > 0) {
        const idx = pickIndex(node.untried.length, this.rng);
        const col = node.untried[idx];
        node.untried[idx] = node.untried[node.untried.length - 1];
        node.untried.pop();
        b.playUnchecked(col);
        const child = new Node(col, node, b);
        node.children.push(child);
        node = child;
      }
      const winner = this.rollout(b);
      let up: Node | null = node;
      while (up) {
        up.visits++;
        if (winner === 0) up.wins += 0.5;
        else if (winner === up.justMoved) up.wins += 1;
        up = up.parent;
      }
    }
    return root;
  }

  selectMove(board: Board): number {
    const root = this.search(board);
    let most = 0;
    for (const ch of root.children) most = Math.max(most, ch.visits);
    const ties = root.children.filter((ch) => ch.visits === most).map((ch) => ch.move as number);
    ties.sort((a, b) => a - b);
    return ties[pickIndex(ties.length, this.rng)];
  }

  analyze(board: Board): {
    visits: number[];
    q: Array<number | null>;
    value: number;
  } {
    const root = this.search(board);
    const visits = new Array(7).fill(0);
    const q = new Array(7).fill(null);
    let best = 0;
    for (const ch of root.children) {
      const col = ch.move as number;
      visits[col] = ch.visits;
      q[col] = ch.wins / ch.visits;
      if (visits[col] > visits[best]) best = col;
    }
    // root children were played by the side to move at the root, so q is its score.
    // Report the score of the move the search would actually play, not an average over
    // exploratory moves that will never be chosen.
    return { visits, q, value: q[best] ?? 0.5 };
  }
}
