"""Move-string notation helpers (1-indexed column digits, e.g. ``"4453"``)."""

from __future__ import annotations

from c4.engine.board import WIDTH, Board, IllegalMoveError


def parse_moves(moves: str) -> list[int]:
    """Parse a 1-indexed move string into 0-indexed columns, validating the characters."""
    cols = []
    for ch in moves.strip():
        if not ch.isdigit() or not 1 <= int(ch) <= WIDTH:
            raise ValueError(f"invalid move character {ch!r} in {moves!r}")
        cols.append(int(ch) - 1)
    return cols


def format_moves(cols: list[int]) -> str:
    return "".join(str(c + 1) for c in cols)


def is_valid_game(moves: str) -> bool:
    """True if ``moves`` is a legal sequence (it may end in a finished game)."""
    try:
        Board.from_moves(parse_moves(moves))
    except (ValueError, IllegalMoveError):
        return False
    return True
