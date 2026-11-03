import { MATE, type Analysis } from "../ai/analysis";
import type { BotOption } from "../ai/bots";
import type { Settings } from "../state/settings";

interface Props {
  settings: Settings;
  bots: BotOption[];
  turn: 1 | 2;
  humanTurn: boolean;
  thinking: boolean;
  terminal: boolean;
  result: string | null;
  analysis: Analysis | null;
  humanProb: number | null;
  moves: string;
  copied: boolean;
  error: string | null;
  onSettings: (s: Settings) => void;
  onNew: () => void;
  onUndo: () => void;
  onShare: () => void;
}

function sourceNote(a: Analysis | null): string {
  if (!a) return "";
  if (a.source === "minimax") {
    const bits = [a.depth ? `minimax d${a.depth}` : "minimax"];
    if (a.evalScore != null && Math.abs(a.evalScore) >= MATE) {
      bits.push("mate");
    } else if (a.evalScore != null && a.evalScore !== 0) {
      bits.push(`${a.evalScore > 0 ? "+" : ""}${Math.round(a.evalScore)}`);
    }
    return bits.join(" · ");
  }
  if (a.source === "mcts") {
    return `mcts · ${a.sims ? a.sims.toLocaleString() : "?"} rollouts`;
  }
  if (a.source === "az") return "AZ · network value";
  return "estimate";
}

function TopMoves({ analysis }: { analysis: Analysis | null }) {
  if (!analysis) {
    return <div className="skeleton" aria-label="Analysing" />;
  }
  const max = Math.max(...analysis.priors);
  const ranked = analysis.priors
    .map((p, col) => ({ p, col }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3);
  return (
    <div className="moves">
      {ranked.map(({ p, col }) => (
        <div className="move-row" key={col}>
          <span className="k">{col + 1}</span>
          <span className="move-bar" style={{ width: `${max ? (p / max) * 100 : 0}%` }} />
          <span className="v">{Math.round(p * 100)}%</span>
        </div>
      ))}
    </div>
  );
}

export function Sidebar(props: Props) {
  const { settings, turn, humanTurn, thinking, terminal, result, analysis, humanProb, moves } = props;

  const status = terminal
    ? result
    : settings.mode === "local"
      ? `Player ${turn} to move`
      : humanTurn
        ? "Your move"
        : "AI is thinking";

  return (
    <aside className="side">
      <section className="card setup">
        <h2 className="label">Game</h2>
        <div className="field">
          <label htmlFor="mode">Mode</label>
          <select
            id="mode"
            className="select"
            value={settings.mode}
            onChange={(e) =>
              props.onSettings({ ...settings, mode: e.target.value as Settings["mode"] })
            }
          >
            <option value="ai">Play the AI</option>
            <option value="local">Two players (same device)</option>
          </select>
        </div>
        {settings.mode === "ai" ? (
          <>
            <div className="field">
              <label htmlFor="bot">Opponent</label>
              <select
                id="bot"
                className="select"
                value={settings.bot}
                onChange={(e) => props.onSettings({ ...settings, bot: e.target.value })}
              >
                {(["classic", "az"] as const).map((family) => {
                  const list = props.bots.filter((b) => b.family === family);
                  if (!list.length) return null;
                  return (
                    <optgroup
                      key={family}
                      label={family === "classic" ? "Classic search" : "AlphaZero"}
                    >
                      {list.map((b) => (
                        <option value={b.id} key={b.id}>
                          {b.label} · {b.elo} Elo
                        </option>
                      ))}
                    </optgroup>
                  );
                })}
              </select>
            </div>
            <div className="field">
              <label htmlFor="human">You play</label>
              <select
                id="human"
                className="select"
                value={settings.human}
                onChange={(e) =>
                  props.onSettings({ ...settings, human: Number(e.target.value) as 1 | 2 })
                }
              >
                <option value={1}>First · teal</option>
                <option value={2}>Second · amber</option>
              </select>
            </div>
          </>
        ) : null}
        <div className="actions">
          <button className="btn primary" onClick={props.onNew} disabled={thinking}>
            New game
          </button>
          <button className="btn" onClick={props.onUndo} disabled={!moves || thinking}>
            Undo
          </button>
        </div>
        {props.error ? (
          <p className="toast" role="alert">
            {props.error}
          </p>
        ) : null}
      </section>

      <section className="card" aria-live="polite">
        <h2 className="label">Status</h2>
        <p className="status-line">
          <span>{status}</span>
          {thinking ? (
            <span className="thinking" aria-label="AI is thinking">
              <i />
              <i />
              <i />
            </span>
          ) : null}
        </p>
        {terminal ? (
          <p className="muted small">Game over · {moves.length} moves</p>
        ) : (
          <>
            <div className="eval">
              <span className="eval-num">
                {humanProb == null ? "—" : `${Math.round(humanProb * 100)}%`}
              </span>
              <div
                className="eval-bar"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={humanProb == null ? 50 : Math.round(humanProb * 100)}
              >
                <div
                  className={`eval-fill${
                    settings.mode === "ai" && settings.human === 2 ? " p2" : ""
                  }`}
                  style={{ width: `${humanProb == null ? 50 : humanProb * 100}%` }}
                />
              </div>
            </div>
            <p className="muted small">
              {settings.mode === "local"
                ? `Player ${turn} to move`
                : humanTurn
                  ? "your estimated chance"
                  : "AI's estimated chance"}
              {analysis
                ? ` · ${sourceNote(analysis)}`
                : settings.bot.startsWith("az:")
                  ? " · loading model…"
                  : ""}
              {analysis?.ms ? ` · ${Math.round(analysis.ms)} ms` : ""}
            </p>
          </>
        )}
      </section>

      {terminal ? null : (
        <section className="card">
          <h2 className="label">Top moves</h2>
          <TopMoves analysis={analysis} />
        </section>
      )}

      <section className="card">
        <h2 className="label">Moves</h2>
        {moves ? (
          <div className="history">
            {[...moves].map((ch, i) => (
              <span key={i} className={i === moves.length - 1 ? "last" : ""}>
                {ch}
              </span>
            ))}
          </div>
        ) : (
          <p className="muted small">No moves yet.</p>
        )}
      </section>

      <button className="btn ghost" onClick={props.onShare}>
        {props.copied ? "Link copied" : "Share position"}
      </button>
    </aside>
  );
}
