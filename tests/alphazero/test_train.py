from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from c4.alphazero.config import load_config  # noqa: E402
from c4.alphazero.train import train  # noqa: E402

pytestmark = pytest.mark.torch


def tiny_cfg(tmp_path, workers: int = 1):
    return load_config(
        "configs/smoke.yaml",
        [
            f"runtime.out_dir={tmp_path}",
            "run_name=resume-test",
            "net.channels=8",
            "net.blocks=1",
            "mcts.sims=4",
            f"selfplay.workers={workers}",
            "selfplay.games_per_iter=8",
            "selfplay.parallel_games_per_worker=4",
            "train.batch_size=32",
            "train.min_buffer=50",
            "eval.every_iters=2",
            "eval.arena=[random]",
            "eval.benchmark_limit=20",
            "eval.arena_sims=2",
        ],
    )


def test_train_kill_and_resume(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    run_dir = train(cfg, resume="none", max_iters=2)
    state1 = (run_dir / "state.pt").stat().st_mtime
    train(cfg, resume="auto", max_iters=2)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [m["iteration"] for m in lines] == [0, 1, 2, 3]
    games = [m["total/games"] for m in lines]
    assert games == [8, 16, 24, 32]
    sizes = [m["buffer/size"] for m in lines]
    assert sizes == sorted(sizes) and sizes[2] > sizes[1]
    assert (run_dir / "state.pt").stat().st_mtime >= state1
    assert any("arena/random" in m for m in lines)
    assert sorted(p.name for p in (run_dir / "models").glob("*.pt"))[-1] == "model_iter_0003.pt"


@pytest.mark.slow
def test_train_multiprocess(tmp_path) -> None:
    run_dir = train(tiny_cfg(tmp_path, workers=2), resume="none", max_iters=1)
    assert (run_dir / "state.pt").exists()
