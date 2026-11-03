export interface Settings {
  mode: "ai" | "local";
  bot: string;
  human: 1 | 2;
}

export const DEFAULT_BOT = "minimax:d=4";

export function parseSettings(params: URLSearchParams): Settings {
  return {
    mode: params.get("mode") === "local" ? "local" : "ai",
    bot: params.get("bot") ?? DEFAULT_BOT,
    human: params.get("human") === "2" ? 2 : 1,
  };
}
