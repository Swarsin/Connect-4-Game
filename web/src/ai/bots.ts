export interface BotOption {
  id: string;
  label: string;
  elo: number;
  family: "classic" | "az";
}

export interface AzLevel {
  id: string;
  label: string;
  model: string;
  sims: number;
  temperature: number;
  elo: number;
  checkpoint?: string;
}

export function azOptions(levels: AzLevel[]): BotOption[] {
  return levels.map((l) => ({
    id: `az:${l.id}`,
    label: `AlphaZero · ${l.label}`,
    elo: l.elo,
    family: "az",
  }));
}

/** Measured in the Python evaluation pool — see results/elo/ratings.md. */
export const BOTS: BotOption[] = [
  { id: "minimax:d=2", label: "Minimax · depth 2", elo: 837, family: "classic" },
  { id: "minimax:d=4", label: "Minimax · depth 4", elo: 940, family: "classic" },
  { id: "minimax:d=6", label: "Minimax · depth 6", elo: 1010, family: "classic" },
  { id: "minimax:d=8", label: "Minimax · depth 8", elo: 1024, family: "classic" },
  { id: "uct:it=500", label: "MCTS · 500 sims", elo: 798, family: "classic" },
  { id: "uct:it=2000", label: "MCTS · 2,000 sims", elo: 936, family: "classic" },
];

export function botById(id: string): BotOption {
  return BOTS.find((b) => b.id === id) ?? BOTS[1];
}
