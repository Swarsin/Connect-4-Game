"""Play games and matches between agents.

Matches take agent *spec strings* (see :mod:`c4.agents.registry`) so games can be distributed
across worker processes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from c4.agents import Agent, make_agent
from c4.engine import Board
from c4.eval.openings import get_openings


@dataclass
class GameRecord:
    first: str  # spec of the agent that played first (player 1)
    second: str
    opening: str
    moves: str
    winner: int  # 0 draw, 1 first player, 2 second player
    seconds: float = 0.0


@dataclass
class MatchResult:
    a: str
    b: str
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    games: list[GameRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    @property
    def a_score(self) -> float:
        return (self.a_wins + 0.5 * self.draws) / max(1, self.n)

    def summary(self) -> str:
        return (
            f"{self.a} vs {self.b}: +{self.a_wins} ={self.draws} -{self.b_wins} "
            f"(score {self.a_score:.3f} over {self.n} games)"
        )


def play_game(first: Agent, second: Agent, opening: str = "") -> GameRecord:
    """Play one game from ``opening``; ``first`` is player 1."""
    t = time.perf_counter()
    board = Board.from_moves(opening)
    agents = {1: first, 2: second}
    while not board.is_terminal():
        board.play(agents[board.player_to_move()].select_move(board))
    winner = board.winner()
    assert winner is not None
    return GameRecord(
        getattr(first, "name", "?"),
        getattr(second, "name", "?"),
        opening,
        board.to_moves(),
        winner,
        time.perf_counter() - t,
    )


def _worker(args: tuple[str, str, list[tuple[str, bool]], int]) -> list[GameRecord]:
    spec_a, spec_b, jobs, seed = args
    a, b = make_agent(spec_a, seed=seed), make_agent(spec_b, seed=seed + 1)
    out = []
    for i, (opening, a_first) in enumerate(jobs):
        a.reset(seed + 2 * i)
        b.reset(seed + 2 * i + 1)
        out.append(play_game(a, b, opening) if a_first else play_game(b, a, opening))
    return out


def play_match(
    spec_a: str,
    spec_b: str,
    openings: Sequence[str] | str = "two",
    n_games: int | None = None,
    seed: int = 0,
    workers: int = 1,
) -> MatchResult:
    """Each opening is played twice with colours swapped. ``n_games`` truncates the schedule."""
    ops = get_openings(openings) if isinstance(openings, str) else list(openings)
    jobs = [(o, af) for o in ops for af in (True, False)]
    if n_games is not None:
        while len(jobs) < n_games:  # repeat schedule (useful for stochastic agents)
            jobs = jobs + jobs
        jobs = jobs[:n_games]
    workers = max(1, min(workers, len(jobs)))
    chunks = [jobs[i::workers] for i in range(workers)]
    tasks = [(spec_a, spec_b, ch, seed + 10_000 * i) for i, ch in enumerate(chunks)]
    if workers == 1:
        records = _worker(tasks[0])
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            records = [r for rs in pool.map(_worker, tasks) for r in rs]

    res = MatchResult(spec_a, spec_b, games=records)
    for g in records:
        if g.winner == 0:
            res.draws += 1
        elif (g.winner == 1) == (g.first == spec_a):
            res.a_wins += 1
        else:
            res.b_wins += 1
    return res


def save_games(records: Sequence[GameRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for g in records:
            f.write(json.dumps(asdict(g)) + "\n")


def load_games(path: str | Path) -> list[GameRecord]:
    path = Path(path)
    if not path.exists():
        return []
    return [
        GameRecord(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()
    ]
