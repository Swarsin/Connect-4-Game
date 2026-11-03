"""Bitboard Connect 4 engine.

Bit layout (Pascal Pons style): each column uses ``HEIGHT + 1 = 7`` bits, the extra top bit is a
sentinel that is never set, which stops shifted patterns from wrapping between columns.
Bit index of (row, col) is ``col * 7 + row`` with row 0 at the bottom.

``current`` holds the stones of the side to move, ``mask`` holds all stones.
"""

from __future__ import annotations

from collections.abc import Iterable

WIDTH = 7
HEIGHT = 6
H1 = HEIGHT + 1
N_CELLS = WIDTH * HEIGHT
CENTRE_ORDER = (3, 2, 4, 1, 5, 0, 6)


def _bottom(col: int) -> int:
    return 1 << (col * H1)


def _top(col: int) -> int:
    return 1 << (col * H1 + HEIGHT - 1)


def _column(col: int) -> int:
    return ((1 << HEIGHT) - 1) << (col * H1)


BOTTOM = tuple(_bottom(c) for c in range(WIDTH))
TOP = tuple(_top(c) for c in range(WIDTH))
COLUMN = tuple(_column(c) for c in range(WIDTH))
BOTTOM_MASK = sum(BOTTOM)
BOARD_MASK = BOTTOM_MASK * ((1 << HEIGHT) - 1)


def has_four(pos: int) -> bool:
    """True if the bitboard ``pos`` contains four aligned stones."""
    m = pos & (pos >> 1)  # vertical
    if m & (m >> 2):
        return True
    m = pos & (pos >> 7)  # horizontal
    if m & (m >> 14):
        return True
    m = pos & (pos >> 6)  # diagonal \
    if m & (m >> 12):
        return True
    m = pos & (pos >> 8)  # diagonal /
    return bool(m & (m >> 16))


def winning_cells(pos: int, mask: int) -> int:
    """Empty cells (bitboard) that would complete a four for the stones in ``pos``."""
    r = (pos << 1) & (pos << 2) & (pos << 3)
    for s in (H1, H1 - 1, H1 + 1):
        p = (pos << s) & (pos << (2 * s))
        r |= p & (pos << (3 * s))
        r |= p & (pos >> s)
        p = (pos >> s) & (pos >> (2 * s))
        r |= p & (pos << s)
        r |= p & (pos >> (3 * s))
    return r & (BOARD_MASK ^ mask)


class IllegalMoveError(ValueError):
    pass


class Board:
    """A Connect 4 position. Mutable: use :meth:`copy` before speculative play."""

    __slots__ = ("current", "mask", "n_moves", "history")

    def __init__(self) -> None:
        self.current = 0
        self.mask = 0
        self.n_moves = 0
        self.history: list[int] = []

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_moves(cls, moves: str | Iterable[int]) -> Board:
        """Build a board from a 1-indexed move string (``"4453"``) or 0-indexed column ints."""
        b = cls()
        cols = [int(ch) - 1 for ch in moves] if isinstance(moves, str) else list(moves)
        for c in cols:
            b.play(c)
        return b

    def to_moves(self) -> str:
        return "".join(str(c + 1) for c in self.history)

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b.current = self.current
        b.mask = self.mask
        b.n_moves = self.n_moves
        b.history = self.history.copy()
        return b

    # ------------------------------------------------------------------ queries
    def player_to_move(self) -> int:
        return 1 if self.n_moves % 2 == 0 else 2

    def is_legal(self, col: int) -> bool:
        return 0 <= col < WIDTH and not (self.mask & TOP[col]) and not self.is_terminal()

    def legal_moves(self) -> list[int]:
        if self.is_terminal():
            return []
        m = self.mask
        return [c for c in range(WIDTH) if not m & TOP[c]]

    def legal_mask(self) -> list[bool]:
        legal = set(self.legal_moves())
        return [c in legal for c in range(WIDTH)]

    def last_move_won(self) -> bool:
        return self.n_moves > 0 and has_four(self.current ^ self.mask)

    def is_draw(self) -> bool:
        return self.n_moves == N_CELLS and not self.last_move_won()

    def is_terminal(self) -> bool:
        return self.n_moves == N_CELLS or self.last_move_won()

    def winner(self) -> int | None:
        """1 or 2 for a win, 0 for a draw, ``None`` if the game is still going."""
        if self.last_move_won():
            return 1 if self.n_moves % 2 == 1 else 2
        if self.n_moves == N_CELLS:
            return 0
        return None

    def height(self, col: int) -> int:
        return (self.mask & COLUMN[col]).bit_count()

    def cell(self, row: int, col: int) -> int:
        """Absolute owner of a cell: 0 empty, 1 or 2."""
        bit = 1 << (col * H1 + row)
        if not self.mask & bit:
            return 0
        me = self.player_to_move()
        return me if self.current & bit else 3 - me

    def grid(self) -> list[list[int]]:
        """6x7 nested list of absolute owners, row 0 = bottom."""
        return [[self.cell(r, c) for c in range(WIDTH)] for r in range(HEIGHT)]

    def winning_moves(self, for_side_to_move: bool = True) -> list[int]:
        """Columns where a stone would immediately complete four (ignoring whose turn it is
        for the opponent case: these are the opponent's threats it could play next turn)."""
        if self.is_terminal():
            return []
        pos = self.current if for_side_to_move else self.current ^ self.mask
        playable = (self.mask + BOTTOM_MASK) & BOARD_MASK
        w = winning_cells(pos, self.mask) & playable
        return [c for c in range(WIDTH) if w & COLUMN[c]]

    def key(self) -> int:
        """Unique integer key of the position (side to move is implied by the stone count)."""
        return self.current + self.mask

    def mirror_key(self) -> int:
        """Key that is identical for a position and its left-right mirror image."""
        return min(self.key(), Board.from_moves([WIDTH - 1 - c for c in self.history]).key())

    # ------------------------------------------------------------------ mutation
    def play(self, col: int) -> None:
        if not self.is_legal(col):
            raise IllegalMoveError(f"illegal move {col} in position '{self.to_moves()}'")
        self.current ^= self.mask
        self.mask |= self.mask + BOTTOM[col]
        self.n_moves += 1
        self.history.append(col)

    def play_unchecked(self, col: int) -> None:
        """Like :meth:`play` without legality checks, for hot search loops."""
        self.current ^= self.mask
        self.mask |= self.mask + BOTTOM[col]
        self.n_moves += 1
        self.history.append(col)

    def undo(self) -> int:
        if not self.history:
            raise IllegalMoveError("nothing to undo")
        col = self.history.pop()
        top_bit = 1 << ((self.mask & COLUMN[col]).bit_length() - 1)
        self.mask ^= top_bit
        self.current ^= self.mask
        self.n_moves -= 1
        return col

    # ------------------------------------------------------------------ display
    def __str__(self) -> str:
        sym = {0: ".", 1: "X", 2: "O"}
        rows = [
            " ".join(sym[self.cell(r, c)] for c in range(WIDTH)) for r in reversed(range(HEIGHT))
        ]
        return "\n".join(rows + ["1 2 3 4 5 6 7"])

    def __repr__(self) -> str:
        return f"Board('{self.to_moves()}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())
