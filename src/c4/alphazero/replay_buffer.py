"""Fixed-capacity circular replay buffer of self-play positions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.planes = np.zeros((capacity, 2, 6, 7), dtype=np.uint8)
        self.pi = np.zeros((capacity, 7), dtype=np.float16)
        self.z = np.zeros(capacity, dtype=np.int8)
        self.q = np.zeros(capacity, dtype=np.float16)  # root search value (for zq ablation)
        self.ptr = 0
        self.size = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self, planes: np.ndarray, pi: np.ndarray, z: np.ndarray, q: np.ndarray | None = None
    ) -> None:
        n = len(planes)
        if n == 0:
            return
        if n > self.capacity:
            planes, pi, z = planes[-self.capacity :], pi[-self.capacity :], z[-self.capacity :]
            q = None if q is None else q[-self.capacity :]
            n = self.capacity
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.planes[idx] = planes
        self.pi[idx] = pi
        self.z[idx] = z
        self.q[idx] = 0 if q is None else q
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)
        self.total_added += n

    def sample(
        self, batch: int, rng: np.random.Generator, augment_mirror: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch)
        x = self.planes[idx].astype(np.float32)
        pi = self.pi[idx].astype(np.float32)
        z = self.z[idx].astype(np.float32)
        q = self.q[idx].astype(np.float32)
        if augment_mirror:
            flip = rng.random(batch) < 0.5
            x[flip] = x[flip][..., ::-1]
            pi[flip] = pi[flip][:, ::-1]
        return x, pi, z, q

    def state(self) -> dict[str, np.ndarray]:
        n = self.size
        if n < self.capacity:
            sl = slice(0, n)
            return {
                "planes": self.planes[sl], "pi": self.pi[sl], "z": self.z[sl], "q": self.q[sl],
                "meta": np.array([self.capacity, self.ptr, self.size, self.total_added]),
            }  # fmt: skip
        return {
            "planes": self.planes, "pi": self.pi, "z": self.z, "q": self.q,
            "meta": np.array([self.capacity, self.ptr, self.size, self.total_added]),
        }  # fmt: skip

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **self.state())  # type: ignore[arg-type]  # numpy stubs mistype **kwargs values
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, capacity: int | None = None) -> ReplayBuffer:
        with np.load(path) as d:
            cap, ptr, size, total = (int(v) for v in d["meta"])
            buf = cls(capacity or cap)
            if buf.capacity == cap:
                buf.planes[:size] = d["planes"][:size]
                buf.pi[:size] = d["pi"][:size]
                buf.z[:size] = d["z"][:size]
                buf.q[:size] = d["q"][:size]
                buf.ptr, buf.size = ptr, size
            else:  # capacity changed: re-add in chronological order
                order = np.r_[ptr:size, 0:ptr] if size == cap else np.arange(size)
                buf.add(d["planes"][order], d["pi"][order], d["z"][order], d["q"][order])
            buf.total_added = total
        return buf
