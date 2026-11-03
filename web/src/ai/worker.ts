import { Board } from "../engine/board";
import { normaliseAnalysis, type Analysis, type RawAnalysis } from "./analysis";
import { azAnalyse, azSearch, type AzLevel } from "./az";
import { MinimaxAgent } from "./minimax";
import { UctAgent } from "./uct";
import { makeRng, pickIndex, type Agent } from "./types";

export interface AiRequest {
  id: number;
  kind: "move" | "analyze";
  moves: string;
  agent: string;
  seed: number;
  /** Present for AlphaZero levels; resolved to absolute URLs by the UI. */
  level?: AzLevel;
}

export interface AiResponse {
  id: number;
  move?: number;
  analysis?: Analysis;
  ms?: number;
  error?: string;
}

class RandomAgent implements Agent {
  readonly name = "random";
  private rng: () => number;
  constructor(seed: number) {
    this.rng = makeRng(seed);
  }
  reset(seed = 0): void {
    this.rng = makeRng(seed);
  }
  selectMove(b: Board): number {
    const moves = b.legalMoves();
    return moves[pickIndex(moves.length, this.rng)];
  }
}

function build(spec: string, seed: number): Agent {
  const [name, ...rest] = spec.split(":");
  const kv: Record<string, string> = {};
  for (const part of rest.join(":").split(",")) {
    if (!part) continue;
    const [k, v] = part.split("=");
    if (k && v !== undefined) kv[k] = v;
  }
  if (name === "random") return new RandomAgent(seed);
  if (name === "minimax") return new MinimaxAgent(Number(kv.d ?? 4), seed);
  if (name === "uct") return new UctAgent(Number(kv.it ?? 2000), Math.SQRT2, seed);
  throw new Error(`unknown agent "${spec}"`);
}

function send(res: AiResponse): void {
  self.postMessage(res);
}

self.onmessage = async (event: MessageEvent<AiRequest>) => {
  const req = event.data;
  const started = performance.now();
  try {
    const board = Board.fromMoves(req.moves);

    if (req.kind === "move") {
      if (req.level) {
        const { move } = await azSearch(req.level, board, req.seed);
        send({ id: req.id, move, ms: performance.now() - started });
      } else {
        const move = build(req.agent, req.seed).selectMove(board);
        send({ id: req.id, move, ms: performance.now() - started });
      }
      return;
    }

    let analysis: Analysis;
    if (req.level) {
      // Fast calibrated value for the bar: the network head, not an 800-simulation search.
      analysis = normaliseAnalysis(req.agent, (await azAnalyse(req.level, board)) as RawAnalysis);
    } else {
      const agent = build(req.agent, req.seed);
      const analyzer = agent as unknown as { analyze?: (b: Board) => RawAnalysis };
      analysis = normaliseAnalysis(req.agent, analyzer.analyze?.(board));
    }
    analysis.ms = performance.now() - started;
    send({ id: req.id, analysis, ms: analysis.ms });
  } catch (err) {
    send({ id: req.id, error: err instanceof Error ? err.message : String(err) });
  }
};
