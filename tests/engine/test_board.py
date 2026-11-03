"""Tests for the bitboard engine against a naive 2D-list reference implementation."""

from __future__ import annotations

import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from c4.engine import Board, IllegalMoveError, to_planes
from c4.engine.notation import is_valid_game, parse_moves
from c4.engine.symmetry import mirror_moves, mirror_planes

ROWS, COLS = 6, 7


class RefBoard:
    """Deliberately simple reference implementation."""

    def __init__(self) -> None:
        self.grid = [[0] * COLS for _ in range(ROWS)]  # grid[row][col], row 0 bottom
        self.n = 0
        self.won_by: int | None = None

    def to_move(self) -> int:
        return 1 if self.n % 2 == 0 else 2

    def legal(self) -> list[int]:
        if self.won_by or self.n == 42:
            return []
        return [c for c in range(COLS) if self.grid[ROWS - 1][c] == 0]

    def play(self, c: int) -> None:
        r = next(r for r in range(ROWS) if self.grid[r][c] == 0)
        p = self.to_move()
        self.grid[r][c] = p
        self.n += 1
        if self._four(p):
            self.won_by = p

    def _four(self, p: int) -> bool:
        g = self.grid
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + i * dr, c + i * dc) for i in range(4)]
                    if all(
                        0 <= rr < ROWS and 0 <= cc < COLS and g[rr][cc] == p for rr, cc in cells
                    ):
                        return True
        return False

    def winner(self) -> int | None:
        if self.won_by:
            return self.won_by
        return 0 if self.n == 42 else None

    def planes(self) -> np.ndarray:
        me = self.to_move()
        out = np.zeros((2, ROWS, COLS), dtype=np.float32)
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == me:
                    out[0, r, c] = 1
                elif self.grid[r][c]:
                    out[1, r, c] = 1
        return out

    def immediate_wins(self, p: int) -> list[int]:
        res = []
        for c in range(COLS):
            if self.grid[ROWS - 1][c]:
                continue
            r = next(r for r in range(ROWS) if self.grid[r][c] == 0)
            self.grid[r][c] = p
            if self._four(p):
                res.append(c)
            self.grid[r][c] = 0
        return res


def random_game(rng: random.Random) -> list[int]:
    ref = RefBoard()
    moves = []
    while ref.legal():
        c = rng.choice(ref.legal())
        ref.play(c)
        moves.append(c)
    return moves


def check_against_ref(moves: list[int]) -> None:
    b, ref = Board(), RefBoard()
    for c in moves:
        assert b.legal_moves() == ref.legal()
        assert b.player_to_move() == ref.to_move()
        assert b.winner() == ref.winner()
        assert np.array_equal(to_planes(b), ref.planes())
        if not b.is_terminal():
            assert b.winning_moves(True) == ref.immediate_wins(ref.to_move())
            assert b.winning_moves(False) == ref.immediate_wins(3 - ref.to_move())
        b.play(c)
        ref.play(c)
    assert b.winner() == ref.winner()
    assert b.legal_moves() == ref.legal()
    assert b.grid() == ref.grid
    assert np.array_equal(to_planes(b), ref.planes())


# ---------------------------------------------------------------- hand-written cases
@pytest.mark.parametrize(
    "moves, winner",
    [
        ("1213141", 1),  # vertical, column 1
        ("7676767", 1),  # vertical, edge column 7
        ("1122334", 1),  # horizontal bottom-left
        ("4455667", 1),  # horizontal bottom-right edge
        ("12233434414", 1),  # diagonal /
        ("76655454474", 1),  # diagonal \
        ("17273757", 2),  # vertical win for player 2
    ],
)
def test_known_wins(moves: str, winner: int) -> None:
    b = Board.from_moves(moves)
    assert b.winner() == winner, str(b)
    assert b.is_terminal()
    assert b.legal_moves() == []


def test_full_board_draw() -> None:
    # Known 42-move draw
    moves = "472424756645422154157737216635626711335331"
    # verify our sequence is actually a legal draw using the reference
    ref = RefBoard()
    for ch in moves:
        assert int(ch) - 1 in ref.legal()
        ref.play(int(ch) - 1)
    assert ref.winner() == 0
    b = Board.from_moves(moves)
    assert b.is_draw() and b.winner() == 0 and b.is_terminal()


def test_illegal_moves() -> None:
    b = Board.from_moves("111111")
    with pytest.raises(IllegalMoveError):
        b.play(0)
    with pytest.raises(IllegalMoveError):
        b.play(7)
    with pytest.raises(IllegalMoveError):
        Board().undo()
    won = Board.from_moves("1213141")
    with pytest.raises(IllegalMoveError):
        won.play(3)


def test_play_undo_roundtrip() -> None:
    rng = random.Random(0)
    for _ in range(50):
        moves = random_game(rng)
        b = Board()
        snapshots = []
        for c in moves:
            snapshots.append((b.current, b.mask, b.n_moves, list(b.history)))
            b.play(c)
        for snap in reversed(snapshots):
            b.undo()
            assert (b.current, b.mask, b.n_moves, b.history) == snap


def test_notation() -> None:
    assert parse_moves("4453") == [3, 3, 4, 2]
    assert Board.from_moves("4453").to_moves() == "4453"
    assert is_valid_game("4453")
    assert not is_valid_game("1111111")
    assert not is_valid_game("48")
    with pytest.raises(ValueError):
        parse_moves("0")


def test_key_unique_and_mirror() -> None:
    a = Board.from_moves("12")
    b = Board.from_moves("21")
    assert a.key() != b.key()
    assert Board.from_moves("1234").key() == Board.from_moves("3214").key()  # transposition
    assert Board.from_moves("12").mirror_key() == Board.from_moves("76").mirror_key()


def test_cell_and_str() -> None:
    b = Board.from_moves("44")
    assert b.cell(0, 3) == 1 and b.cell(1, 3) == 2 and b.cell(0, 0) == 0
    assert "X" in str(b) and "O" in str(b)


# ---------------------------------------------------------------- random / property tests
def test_random_games_match_reference() -> None:
    rng = random.Random(1234)
    for _ in range(300):
        check_against_ref(random_game(rng))


@settings(max_examples=10_000, deadline=None)
@given(st.lists(st.integers(0, 6), max_size=60))
def test_hypothesis_matches_reference(cols: list[int]) -> None:
    b, ref = Board(), RefBoard()
    for c in cols:
        if c not in ref.legal():
            assert not b.is_legal(c)
            continue
        b.play(c)
        ref.play(c)
        assert b.winner() == ref.winner()
        assert b.legal_moves() == ref.legal()
    assert np.array_equal(to_planes(b), ref.planes())


def test_mirror_preserves_winner() -> None:
    rng = random.Random(7)
    for _ in range(200):
        moves = "".join(str(c + 1) for c in random_game(rng))
        a, m = Board.from_moves(moves), Board.from_moves(mirror_moves(moves))
        assert a.winner() == m.winner()
        assert np.array_equal(mirror_planes(to_planes(a)), to_planes(m))
