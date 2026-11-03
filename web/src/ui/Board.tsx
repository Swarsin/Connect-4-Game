import { useMemo, useState } from "react";

import { Board, HEIGHT, WIDTH } from "../engine/board";

interface Props {
  moves: string;
  onPlay: (col: number) => void;
  disabled: boolean;
}

export function BoardView({ moves, onPlay, disabled }: Props) {
  const board = useMemo(() => Board.fromMoves(moves), [moves]);
  const [hover, setHover] = useState<number | null>(null);
  const [focusCol, setFocusCol] = useState(3);
  const [focused, setFocused] = useState(false);

  const win = board.winningLine();
  const winSet = new Set((win ?? []).map(([r, c]) => `${r},${c}`));
  const turn = board.playerToMove();

  const legal = new Set(board.legalMoves());
  const canPlay = !disabled && !board.isTerminal();
  const ghostCol = canPlay && hover != null && legal.has(hover) ? hover : null;
  const ghostRow = ghostCol != null ? board.height(ghostCol) : -1;

  function colFromEvent(event: React.MouseEvent<HTMLDivElement>): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const pad = rect.width * 0.04;
    const inner = rect.width - pad * 2;
    const col = Math.floor((event.clientX - rect.left - pad) / (inner / WIDTH));
    return Math.max(0, Math.min(WIDTH - 1, col));
  }

  function handleClick(event: React.MouseEvent<HTMLDivElement>) {
    if (!canPlay) return;
    const col = colFromEvent(event);
    if (legal.has(col)) onPlay(col);
  }

  function handleKey(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key >= "1" && event.key <= "7") {
      const col = Number(event.key) - 1;
      if (legal.has(col)) {
        event.preventDefault();
        onPlay(col);
      }
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const next = focusCol + (event.key === "ArrowLeft" ? -1 : 1);
      setFocusCol(Math.max(0, Math.min(WIDTH - 1, next)));
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (legal.has(focusCol)) onPlay(focusCol);
    }
  }

  return (
    <div className="board-shell">
      <div className="col-hint" aria-hidden="true">
        {Array.from({ length: WIDTH }, (_, c) => (
          <span key={c} className={ghostCol === c || (focused && canPlay && focusCol === c) ? "on" : ""} />
        ))}
      </div>
      <div
        className="board"
        role="grid"
        aria-label="Connect 4 board. Use keys 1 to 7 or arrow keys and Enter."
        aria-rowcount={HEIGHT}
        aria-colcount={WIDTH}
        tabIndex={0}
        onMouseMove={(e) => canPlay && setHover(colFromEvent(e))}
        onMouseLeave={() => setHover(null)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onClick={handleClick}
        onKeyDown={handleKey}
      >
        {Array.from({ length: HEIGHT }, (_, i) => HEIGHT - 1 - i).map((row) =>
          Array.from({ length: WIDTH }, (_, col) => {
            const owner = board.cell(row, col);
            const isGhost = ghostCol === col && ghostRow === row;
            const isWin = winSet.has(`${row},${col}`);
            return (
              <div
                key={`${row}-${col}`}
                role="gridcell"
                aria-label={`Row ${row + 1}, column ${col + 1}: ${
                  owner === 0 ? "empty" : owner === 1 ? "player 1" : "player 2"
                }`}
                className={`cell${isWin ? " win" : ""}`}
              >
                {owner !== 0 && (
                  <span
                    className={`disc p${owner}${
                      board.lastRow === row && board.lastCol === col ? " just-dropped" : ""
                    }`}
                  />
                )}
                {isGhost && <span className={`disc ghost p${turn}`} />}
              </div>
            );
          }),
        )}
      </div>
    </div>
  );
}
