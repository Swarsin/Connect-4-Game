"""Wrapper around Pascal Pons' ``c4solver`` binary (built by ``scripts/build_solver.sh``).

The solver is AGPL-3.0 licensed; it is only invoked as an external process for evaluation and
is never bundled with this project.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from c4.engine import Board

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "third_party" / "connect4-solver"
INVALID = -1000


def solver_dir() -> Path:
    return Path(os.environ.get("C4_SOLVER_DIR", DEFAULT_DIR))


def solver_available() -> bool:
    return (solver_dir() / "c4solver").exists()


class SolverError(RuntimeError):
    pass


class Solver:
    """A long-running ``c4solver -a`` subprocess with an in-memory + optional on-disk cache.

    ``analyze(moves)`` returns a list of 7 scores (Pons convention, side to move's perspective),
    with ``None`` for illegal columns.
    """

    def __init__(self, cache_path: str | Path | None = None, weak: bool = False) -> None:
        d = solver_dir()
        exe = d / "c4solver"
        if not exe.exists():
            raise SolverError(f"solver not found at {exe}; run scripts/build_solver.sh")
        args = [str(exe), "-a", "-b", str(d / "7x6.book")]
        if weak:
            args.append("-w")
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._mem: dict[str, list[int | None]] = {}
        self._db: sqlite3.Connection | None = None
        if cache_path is not None:
            self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS a (moves TEXT PRIMARY KEY, scores TEXT)")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()  # type: ignore[union-attr]
            self._proc.wait(timeout=10)
        if self._db is not None:
            self._db.commit()
            self._db.close()

    def __enter__(self) -> Solver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _query(self, moves: str) -> list[int | None]:
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(moves + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise SolverError(f"solver died on '{moves}'")
        toks = line.split()
        scores = [int(t) for t in toks[-7:]]
        pos = toks[0] if len(toks) == 8 else ""
        if pos != moves:
            raise SolverError(f"solver output mismatch for '{moves}': {line!r}")
        return [None if s == INVALID else s for s in scores]

    def analyze(self, moves: str) -> list[int | None]:
        if moves in self._mem:
            return self._mem[moves]
        b = Board.from_moves(moves)
        if b.is_terminal():
            raise SolverError(f"cannot analyse terminal position '{moves}'")
        with self._lock:
            if self._db is not None:
                row = self._db.execute("SELECT scores FROM a WHERE moves=?", (moves,)).fetchone()
                if row:
                    res = [None if t == "x" else int(t) for t in row[0].split(",")]
                    self._mem[moves] = res
                    return res
            res = self._query(moves)
            if self._db is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO a VALUES (?, ?)",
                    (moves, ",".join("x" if s is None else str(s) for s in res)),
                )
        self._mem[moves] = res
        return res

    def score(self, moves: str) -> int:
        """Score of the position for the side to move."""
        return max(s for s in self.analyze(moves) if s is not None)


def analyze_many(
    positions: Sequence[str], workers: int = 8, cache_path: str | Path | None = None
) -> list[list[int | None]]:
    """Analyse many positions in parallel with ``workers`` solver processes."""
    solvers = [Solver(cache_path if i == 0 else None) for i in range(workers)]
    try:
        chunks = [list(range(i, len(positions), workers)) for i in range(workers)]
        out: list[list[int | None] | None] = [None] * len(positions)

        def run(k: int) -> None:
            for idx in chunks[k]:
                out[idx] = solvers[k].analyze(positions[idx])

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(run, range(workers)))
        return out  # type: ignore[return-value]
    finally:
        for s in solvers:
            s.close()


def optimal_moves(scores: Sequence[int | None]) -> list[int]:
    best = max(s for s in scores if s is not None)
    return [c for c, s in enumerate(scores) if s == best]


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


def outcome_preserving_moves(scores: Sequence[int | None]) -> list[int]:
    best = sign(max(s for s in scores if s is not None))
    return [c for c, s in enumerate(scores) if s is not None and sign(s) == best]
