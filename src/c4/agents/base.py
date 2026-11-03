"""Common agent interface."""

from __future__ import annotations

import random
from typing import Any, Protocol, runtime_checkable

from c4.engine import Board


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def select_move(self, board: Board) -> int: ...

    def analyze(self, board: Board) -> dict[str, Any] | None: ...


class BaseAgent:
    """Convenience base class: holds a seeded RNG and a default no-op ``analyze``."""

    name = "base"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_move(self, board: Board) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def analyze(self, board: Board) -> dict[str, Any] | None:
        return None

    def __repr__(self) -> str:
        return f"<{self.name}>"
