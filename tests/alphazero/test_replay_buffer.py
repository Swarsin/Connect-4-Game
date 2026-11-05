from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import c4.alphazero.replay_buffer as replay_buffer
from c4.alphazero.replay_buffer import ReplayBuffer


def filled_buffer(capacity: int = 8) -> ReplayBuffer:
    """A wrapped buffer whose retained samples are identifiable by z."""
    buf = ReplayBuffer(capacity)
    n = 10
    planes = np.zeros((n, 2, 6, 7), dtype=np.uint8)
    planes[:, 0, 0, 0] = np.arange(1, n + 1, dtype=np.uint8)
    pi = np.full((n, 7), 1 / 7, dtype=np.float16)
    z = np.arange(n, dtype=np.int8)
    q = np.arange(n, dtype=np.float16)
    buf.add(planes[:6], pi[:6], z[:6], q[:6])
    buf.add(planes[6:], pi[6:], z[6:], q[6:])
    return buf


def record_closes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap np.load so every returned archive reports its close() calls."""
    closed: list[int] = []
    real_load = replay_buffer.np.load

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        archive = real_load(*args, **kwargs)
        real_close = archive.close

        def close() -> None:
            closed.append(1)
            real_close()

        archive.close = close
        return archive

    monkeypatch.setattr(replay_buffer.np, "load", recording_load)
    return closed


@pytest.mark.parametrize("capacity", [8, 4, 16])
def test_buffer_load_closes_archive(tmp_path, monkeypatch, capacity: int) -> None:
    path = tmp_path / "buffer.npz"
    saved = filled_buffer()
    saved.save(path)
    closed = record_closes(monkeypatch)

    loaded = ReplayBuffer.load(path, capacity=capacity)

    assert closed == [1]
    assert loaded.total_added == saved.total_added == 10
    if capacity == saved.capacity:
        assert np.array_equal(loaded.z, saved.z)
        assert loaded.ptr == saved.ptr
    elif capacity < saved.capacity:
        assert loaded.z.tolist() == [6, 7, 8, 9]
        assert loaded.ptr == 0
    else:
        assert loaded.z[:8].tolist() == list(range(2, 10))
        assert loaded.ptr == 8
    assert loaded.size == min(capacity, saved.size)


def test_buffer_load_closes_archive_on_error(tmp_path, monkeypatch) -> None:
    path = tmp_path / "corrupt.npz"
    np.savez(path, meta=np.array([8, 0, 0, 0]))
    closed = record_closes(monkeypatch)

    with pytest.raises(KeyError):
        ReplayBuffer.load(path)

    assert closed == [1]
