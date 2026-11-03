import { useEffect, useMemo, useRef, useState } from "react";

import { AiClient, type Analysis } from "./ai/client";
import { azOptions, BOTS, type AzLevel, type BotOption } from "./ai/bots";
import { Board } from "./engine/board";
import { DEFAULT_BOT, parseSettings, type Settings } from "./state/settings";
import { BoardView } from "./ui/Board";
import { Sidebar } from "./ui/Sidebar";

import "./styles/app.css";

function readLocation(): { path: string; params: URLSearchParams } {
  const raw = window.location.hash.replace(/^#/, "") || "/play";
  const [path, qs] = raw.split("?");
  return { path: path || "/play", params: new URLSearchParams(qs ?? "") };
}

function safeMoves(m: string | null): string {
  if (!m) return "";
  try {
    Board.fromMoves(m);
    return m;
  } catch {
    return "";
  }
}

export default function App() {
  const initial = useMemo(readLocation, []);
  const [route, setRoute] = useState(initial.path);
  const [settings, setSettings] = useState<Settings>(() => parseSettings(initial.params));
  const [moves, setMoves] = useState(() => safeMoves(initial.params.get("m")));
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [gameSeed, setGameSeed] = useState(1);
  const [theme, setTheme] = useState<"light" | "dark">(
    () => (localStorage.getItem("c4-theme") as "light" | "dark") || "light",
  );
  const [azLevels, setAzLevels] = useState<AzLevel[]>([]);

  const aiRef = useRef<AiClient | null>(null);
  const aiKey = useRef<string | null>(null);
  const analysisKey = useRef<string | null>(null);
  const baseUrl = useMemo(() => new URL(import.meta.env.BASE_URL, window.location.href), []);

  const board = useMemo(() => Board.fromMoves(moves), [moves]);
  const turn = board.playerToMove();
  const terminal = board.isTerminal();
  const winner = board.winner();
  const humanTurn = settings.mode === "local" || turn === settings.human;
  const botOptions: BotOption[] = useMemo(
    () => [...BOTS, ...azOptions(azLevels)],
    [azLevels],
  );
  const bot = botOptions.find((b) => b.id === settings.bot) ?? BOTS[1];
  const levelDef = azLevels.find((l) => `az:${l.id}` === settings.bot);

  const result =
    winner == null
      ? null
      : winner === 0
        ? "Draw"
        : settings.mode === "local"
          ? `Player ${winner} wins`
          : winner === settings.human
            ? "You win"
            : "AI wins";

  // AlphaZero level manifest (shipped next to the models)
  useEffect(() => {
    fetch(new URL("models/levels.json", baseUrl).href)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: { levels?: AzLevel[] }) => {
        const levels = d.levels ?? [];
        setAzLevels(levels);
        // a stale share link may name a level that no longer exists
        setSettings((s) =>
          s.bot.startsWith("az:") && !levels.some((l) => `az:${l.id}` === s.bot)
            ? { ...s, bot: DEFAULT_BOT }
            : s,
        );
      })
      .catch(() => {
        setAzLevels([]);
        setSettings((s) => (s.bot.startsWith("az:") ? { ...s, bot: DEFAULT_BOT } : s));
      });
  }, [baseUrl]);

  // hash routing
  useEffect(() => {
    const onHash = () => {
      const { path, params } = readLocation();
      setRoute(path);
      if (path === "/play") {
        setSettings(parseSettings(params));
        setMoves(safeMoves(params.get("m")));
        setAnalysis(null);
        setError(null);
        setThinking(false);
        aiKey.current = null;
        analysisKey.current = null;
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // theme
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("c4-theme", theme);
  }, [theme]);

  // keep the URL shareable
  useEffect(() => {
    if (route !== "/play") return;
    const p = new URLSearchParams();
    if (moves) p.set("m", moves);
    p.set("bot", settings.bot);
    p.set("mode", settings.mode);
    p.set("human", String(settings.human));
    const next = `#/play?${p.toString()}`;
    if (window.location.hash !== next) window.history.replaceState(null, "", next);
  }, [moves, settings, route]);

  // AI turn + position analysis
  useEffect(() => {
    if (route !== "/play") return;
    const ai = (aiRef.current ??= new AiClient());
    const b = Board.fromMoves(moves);
    if (b.isTerminal()) {
      setThinking(false);
      setAnalysis(null);
      return;
    }
    if (settings.bot.startsWith("az:") && !levelDef) return; // manifest still loading
    const level = levelDef
      ? {
          model: new URL(levelDef.model, baseUrl).href,
          sims: levelDef.sims,
          temperature: levelDef.temperature,
          wasmBase: new URL("ort/", baseUrl).href,
        }
      : undefined;
    const humanToMove = settings.mode === "local" || b.playerToMove() === settings.human;

    if (humanToMove) {
      setThinking(false);
      if (analysisKey.current === moves) return;
      analysisKey.current = moves;
      const requested = moves;
      ai.request({ kind: "analyze", moves, agent: settings.bot, seed: gameSeed, level })
        .then((res) => {
          if (analysisKey.current !== requested) return;
          if (!res.error) setAnalysis(res.analysis ?? null);
        })
        .catch(() => undefined);
      return;
    }

    setAnalysis(null);
    if (aiKey.current === moves) return;
    aiKey.current = moves;
    const requested = moves;
    setThinking(true);
    ai.request({ kind: "move", moves, agent: settings.bot, seed: gameSeed, level })
      .then((res) => {
        if (res.error || res.move == null) {
          setError(res.error ?? "The AI failed to return a move.");
          return;
        }
        setMoves((prev) => {
          if (prev !== requested) return prev;
          const nb = Board.fromMoves(prev);
          if (!nb.isLegal(res.move as number)) {
            setError("The AI returned an illegal move.");
            return prev;
          }
          nb.play(res.move as number);
          return nb.toMoves();
        });
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setThinking(false));
  }, [moves, settings, route, gameSeed, levelDef, baseUrl]);

  useEffect(
    () => () => {
      aiRef.current?.dispose();
      aiRef.current = null;
      // StrictMode unmounts and remounts once; drop the dedupe keys so the fresh worker
      // actually receives the first move/analysis request.
      aiKey.current = null;
      analysisKey.current = null;
    },
    [],
  );

  function restart(next: Settings = settings) {
    setSettings(next);
    setMoves("");
    setAnalysis(null);
    setError(null);
    setThinking(false);
    setGameSeed((s) => s + 1);
    aiKey.current = null;
    analysisKey.current = null;
  }

  function handlePlay(col: number) {
    if (thinking || !humanTurn || terminal) return;
    const nb = board.clone();
    if (!nb.isLegal(col)) return;
    nb.play(col);
    setMoves(nb.toMoves());
    setAnalysis(null);
    setError(null);
  }

  function handleUndo() {
    if (!moves || thinking) return;
    let m = moves.slice(0, -1);
    if (settings.mode === "ai") {
      const b = Board.fromMoves(m);
      if (!b.isTerminal() && b.playerToMove() !== settings.human) m = m.slice(0, -1);
    }
    setMoves(m);
    setAnalysis(null);
    setError(null);
    aiKey.current = null;
    analysisKey.current = null;
  }

  async function share() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setError("Could not copy the link.");
    }
  }

  return (
    <div className="app">
      <header className="header">
        <a className="brand" href="#/play" aria-label="Connect Four home">
          <span className="dot" />
          Connect Four
        </a>
        <div className="header-right">
          <span className="pill">
            {settings.mode === "ai" ? `${bot.label} · ${bot.elo} Elo` : "Two players"}
          </span>
          <button
            className="icon-btn"
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
          >
            {theme === "light" ? "Dark" : "Light"}
          </button>
          <a className="icon-btn" href="#/about">
            About
          </a>
        </div>
      </header>

      <main className="main">
        {route === "/about" ? (
          <article className="prose">
            <h1>About</h1>
            <p>
              Connect 4 with two kinds of opponent: fixed-depth Minimax and Monte Carlo tree
              search, and an AlphaZero-style network trained from self-play. Every bot is rated
              against perfect play, and the numbers below come from the project&rsquo;s evaluation
              harness, not from vibes.
            </p>
            <h2>How strong?</h2>
            <table>
              <thead>
                <tr>
                  <th>Opponent</th>
                  <th>Elo</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>AlphaZero, 800 simulations</td>
                  <td>1400</td>
                </tr>
                <tr>
                  <td>AlphaZero, 200 simulations</td>
                  <td>1313</td>
                </tr>
                <tr>
                  <td>MCTS, 10,000 simulations</td>
                  <td>1052</td>
                </tr>
                <tr>
                  <td>Minimax, depth 8</td>
                  <td>1024</td>
                </tr>
                <tr>
                  <td>Random</td>
                  <td>0</td>
                </tr>
              </tbody>
            </table>
            <p className="muted">
              Elo is a Bradley–Terry fit over the project&rsquo;s match pool, anchored at random =
              0. AlphaZero picks a game-theoretically optimal move in 87.7% of held-out positions
              at 800 simulations and wins 30/30 games as first player against a perfect solver.
            </p>
            <h2>How it works</h2>
            <p>
              The rules engine, Minimax, UCT and PUCT search are implemented from scratch. The
              AlphaZero network is a small residual policy/value network trained on games it played
              against itself; the browser version runs an int8 export with ONNX Runtime Web.
            </p>
            <p>
              <a href="https://github.com/Swarsin/Connect-4-Game">Source on GitHub</a> ·{' '}
              <a href="https://github.com/Swarsin/Connect-4-Game/blob/main/docs/report.md">
                Full technical report
              </a>
            </p>
            <p>
              <a href="#/play">Back to the board</a>
            </p>
          </article>
        ) : (
          <div className="layout">
            <div className="board-column">
              {terminal ? (
                <div className={`banner${winner === 0 ? " draw" : " over"}`} role="status">
                  <div>
                    <strong>{result}</strong>
                    <span className="muted small">
                      {settings.mode === "ai" ? `vs ${bot.label}` : `${moves.length} moves`}
                    </span>
                  </div>
                  <button className="btn primary" onClick={() => restart()}>
                    Play again
                  </button>
                </div>
              ) : null}
              <BoardView
                moves={moves}
                onPlay={handlePlay}
                disabled={thinking || !humanTurn || terminal}
              />
              <p className="hint muted small">
                Click a column, or press <kbd>1</kbd>–<kbd>7</kbd>.
              </p>
            </div>
            <Sidebar
              settings={settings}
              bots={botOptions}
              turn={turn}
              humanTurn={humanTurn}
              thinking={thinking}
              terminal={terminal}
              result={result}
              analysis={analysis}
              humanProb={analysis ? analysis.value : null}
              moves={moves}
              copied={copied}
              error={error}
              onSettings={(s) => restart(s)}
              onNew={() => restart()}
              onUndo={handleUndo}
              onShare={share}
            />
          </div>
        )}
      </main>

      <footer className="footer">
        <span>AlphaZero + classic search</span>
        <span aria-hidden="true">·</span>
        <a href="https://github.com/Swarsin/Connect-4-Game">GitHub</a>
      </footer>
    </div>
  );
}
