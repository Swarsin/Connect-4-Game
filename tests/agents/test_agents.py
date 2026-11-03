from __future__ import annotations

import random

import pytest

from c4.agents import make_agent, parse_spec
from c4.agents.heuristic import evaluate
from c4.agents.minimax import MinimaxAgent
from c4.agents.random_agent import RandomAgent
from c4.engine import Board
from c4.eval.arena import play_match


def tactical_positions(kind: str, n: int, seed: int = 0) -> list[Board]:
    """Random positions where the side to move has a win ("win") or must block one ("block")."""
    rng = random.Random(seed)
    out: list[Board] = []
    while len(out) < n:
        b = Board()
        target = rng.randrange(6, 30)
        while not b.is_terminal() and b.n_moves < target:
            b.play(rng.choice(b.legal_moves()))
        if b.is_terminal():
            continue
        mine, theirs = b.winning_moves(True), b.winning_moves(False)
        if kind == "win" and mine or kind == "block" and not mine and len(theirs) == 1:
            out.append(b)
    return out


def test_parse_spec() -> None:
    assert parse_spec("minimax:d=4") == ("minimax", {"depth": 4})
    assert parse_spec("mcts:it=100,c=1.5") == ("mcts", {"iterations": 100, "c": 1.5})
    assert parse_spec("random") == ("random", {})
    assert make_agent("minimax:d=3").name == "minimax:d=3"
    with pytest.raises(ValueError):
        make_agent("nope")


def test_heuristic_symmetric_sign() -> None:
    b = Board.from_moves("4")  # player 1 has a centre stone, player 2 to move
    assert evaluate(b) < 0


@pytest.mark.parametrize("depth", [2, 4])
def test_minimax_takes_wins_and_blocks(depth: int) -> None:
    agent = MinimaxAgent(depth=depth, seed=0)
    for b in tactical_positions("win", 60, seed=1):
        assert agent.select_move(b) in b.winning_moves(True), repr(b)
    for b in tactical_positions("block", 60, seed=2):
        assert agent.select_move(b) == b.winning_moves(False)[0], repr(b)


def test_mcts_takes_wins_and_blocks() -> None:
    agent = make_agent("mcts:it=2000", seed=0)
    wins = tactical_positions("win", 40, seed=3)
    blocks = tactical_positions("block", 40, seed=4)
    ok_w = sum(agent.select_move(b) in b.winning_moves(True) for b in wins)
    ok_b = sum(agent.select_move(b) == b.winning_moves(False)[0] for b in blocks)
    assert ok_w / len(wins) >= 0.95
    assert ok_b / len(blocks) >= 0.95


def test_minimax_ties_are_random_but_seeded() -> None:
    a1, a2 = MinimaxAgent(depth=1, seed=5), MinimaxAgent(depth=1, seed=5)
    b = Board()
    assert [a1.select_move(b) for _ in range(5)] == [a2.select_move(b) for _ in range(5)]


def test_random_agent_legal() -> None:
    agent = RandomAgent(seed=0)
    b = Board.from_moves("111111")
    for _ in range(20):
        assert agent.select_move(b) != 0


def test_minimax_beats_random() -> None:
    res = play_match("minimax:d=4", "random", openings="one", n_games=100, seed=0)
    assert res.a_score >= 0.95, res.summary()


def test_mcts_beats_random() -> None:
    res = play_match("mcts:it=2000", "random", openings="one", n_games=100, seed=0, workers=4)
    assert res.a_score >= 0.95, res.summary()


def test_mcts_tree_reuse_consistent() -> None:
    agent = make_agent("mcts:it=300", seed=0)
    b = Board()
    for _ in range(6):
        b.play(agent.select_move(b))
    assert not b.is_terminal() or b.winner() is not None
