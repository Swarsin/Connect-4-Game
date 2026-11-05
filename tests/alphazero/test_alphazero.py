from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c4.alphazero.config import MCTSConfig, NetConfig, TrainConfig, load_config  # noqa: E402
from c4.alphazero.mcts import (  # noqa: E402
    NetEvaluator,
    SearchParams,
    UniformEvaluator,
    choose_from_priors,
    choose_move,
    new_root,
    run_search,
)
from c4.alphazero.network import C4Net, load_model, masked_softmax, save_model  # noqa: E402
from c4.alphazero.replay_buffer import ReplayBuffer  # noqa: E402
from c4.alphazero.selfplay import play_games  # noqa: E402
from c4.alphazero.trainer import Trainer, lr_at  # noqa: E402
from c4.engine import Board, to_planes  # noqa: E402
from tests.agents.test_agents import tactical_positions  # noqa: E402

pytestmark = pytest.mark.torch


def test_network_shapes_and_masking() -> None:
    net = C4Net(NetConfig(channels=16, blocks=2)).eval()
    x = torch.from_numpy(np.stack([to_planes(Board()), to_planes(Board.from_moves("111111"))]))
    logits, v = net(x)
    assert logits.shape == (2, 7) and v.shape == (2, 1)
    assert torch.all(v.abs() <= 1)
    legal = torch.tensor([[True] * 7, [False] + [True] * 6])
    p = masked_softmax(logits, legal)
    assert torch.allclose(p.sum(-1), torch.ones(2))
    assert p[1, 0] == 0


def test_save_load_model(tmp_path) -> None:
    net = C4Net(NetConfig(channels=8, blocks=1)).eval()
    save_model(net, tmp_path / "m.pt")
    net2 = load_model(tmp_path / "m.pt")
    x = torch.from_numpy(to_planes(Board.from_moves("4453"))[None])
    assert torch.allclose(net(x)[0], net2(x)[0])


def test_fused_inference_matches_network() -> None:
    from c4.alphazero.inference import FastInference

    torch.manual_seed(0)
    net = C4Net(NetConfig(channels=16, blocks=2))
    # give BatchNorm non-trivial statistics
    net.train()
    with torch.no_grad():
        for _ in range(5):
            net(torch.rand(32, 2, 6, 7))
    net.eval()
    boards = tactical_positions("win", 10, seed=3)
    x = torch.from_numpy(np.stack([to_planes(b) for b in boards]))
    logits, v = net(x)
    ref = torch.cat([logits, v], 1)
    fast = FastInference(net, torch.device("cpu"))
    assert torch.allclose(fast(x), ref, atol=1e-4)
    if torch.cuda.is_available():
        gpu = FastInference(net, torch.device("cuda"), half=True)
        for n in (1, 3, 10):
            assert torch.allclose(gpu(x[:n]), ref[:n], atol=2e-2)


def test_mcts_uniform_finds_wins_and_blocks() -> None:
    params = SearchParams(sims=200, c_puct=2.0)
    rng = np.random.default_rng(0)
    wins = tactical_positions("win", 40, seed=11)
    roots = [new_root(b) for b in wins]
    run_search(roots, wins, UniformEvaluator(), params)
    assert all(
        choose_move(r, 0, rng) in b.winning_moves(True) for r, b in zip(roots, wins, strict=True)
    )
    blocks = tactical_positions("block", 40, seed=12)
    roots = [new_root(b) for b in blocks]
    run_search(roots, blocks, UniformEvaluator(), params)
    ok = sum(
        choose_move(r, 0, rng) == b.winning_moves(False)[0]
        for r, b in zip(roots, blocks, strict=True)
    )
    assert ok >= 38


def test_mcts_value_sign_in_won_position() -> None:
    b = Board.from_moves("121212")  # player 1 to move, wins with column 1
    root = new_root(b)
    run_search([root], [b], UniformEvaluator(), SearchParams(sims=100))
    assert root.value() > 0.5
    b2 = Board.from_moves("37475")  # player 2 to move, player 1 threatens columns 2 and 6
    root2 = new_root(b2)
    run_search([root2], [b2], UniformEvaluator(), SearchParams(sims=300))
    assert root2.value() < -0.5


def test_mirror_augmentation_consistent() -> None:
    buf = ReplayBuffer(10)
    b = Board.from_moves("1")
    pi = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    buf.add(to_planes(b, np.uint8)[None], pi[None], np.array([1]))
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, p, _, _ = buf.sample(1, rng, augment_mirror=True)
        # the stone and the policy mass must be on the same side
        stone_col = int(np.argwhere(x[0, 1] == 1)[0][1])
        assert int(np.argmax(p[0])) == stone_col


def test_buffer_roundtrip_and_wraparound(tmp_path) -> None:
    buf = ReplayBuffer(50)
    for k in range(4):
        n = 20
        buf.add(
            np.full((n, 2, 6, 7), k, np.uint8),
            np.full((n, 7), k / 10, np.float32),
            np.full(n, k % 2, np.int8),
        )
    assert len(buf) == 50 and buf.total_added == 80
    buf.save(tmp_path / "b.npz")
    b2 = ReplayBuffer.load(tmp_path / "b.npz")
    assert np.array_equal(buf.planes, b2.planes) and buf.ptr == b2.ptr and b2.total_added == 80
    b3 = ReplayBuffer.load(tmp_path / "b.npz", capacity=100)
    assert len(b3) == 50


def test_trainer_overfits_small_batch() -> None:
    torch.manual_seed(0)
    net = C4Net(NetConfig(channels=32, blocks=2))
    cfg = TrainConfig(
        batch_size=64, lr=0.01, optimizer="adamw", weight_decay=0, amp=False, augment_mirror=False
    )
    tr = Trainer(net, cfg, torch.device("cpu"))
    buf = ReplayBuffer(64)
    rng = np.random.default_rng(0)
    boards = []
    for _ in range(64):
        b = Board()
        for _ in range(rng.integers(0, 20)):
            if b.is_terminal():
                break
            b.play(int(rng.choice(b.legal_moves())))
        if b.is_terminal():
            b.undo()
        boards.append(b)
    planes = np.stack([to_planes(b, np.uint8) for b in boards])
    pi = rng.dirichlet(np.ones(7), size=64).astype(np.float32)
    z = rng.choice([-1, 0, 1], size=64).astype(np.int8)
    buf.add(planes, pi, z)
    first = tr.train_steps(buf, 1, rng)
    for _ in range(300):
        last = tr.train_steps(buf, 1, rng)
    assert last["loss/value"] < 0.1 * first["loss/value"]
    assert last["policy/kl"] < 0.3 * first["policy/kl"]


def test_selfplay_produces_consistent_samples() -> None:
    net = C4Net(NetConfig(channels=8, blocks=1)).eval()
    ev = NetEvaluator(net, torch.device("cpu"), amp=False, cache_size=1000)
    res = play_games(ev, 6, 3, MCTSConfig(sims=8, temp_moves=4), np.random.default_rng(0))
    assert res.n_games == 6
    assert len(res.z) == sum(res.lengths)
    assert np.allclose(res.pi.sum(1), 1, atol=1e-5)
    # replay the recorded games and check the samples' planes and outcomes
    i = 0
    for g, w in zip(res.games, res.winners, strict=True):
        b = Board()
        for ch in g:
            assert np.array_equal(res.planes[i], to_planes(b, np.uint8))
            p = b.player_to_move()
            assert res.z[i] == (0 if w == 0 else 1 if p == w else -1)
            assert res.pi[i][int(ch) - 1] > 0
            b.play(int(ch) - 1)
            i += 1
        assert b.winner() == w


def test_config_extends_and_overrides() -> None:
    cfg = load_config("configs/smoke.yaml", ["mcts.sims=7", "train.lr=0.5"])
    assert cfg.net.channels == 32
    assert cfg.mcts.sims == 7 and cfg.train.lr == 0.5
    assert cfg.mcts.c_puct == 2.0  # inherited from main.yaml
    assert lr_at(load_config("configs/main.yaml").train, 130) == pytest.approx(0.02 * 0.01)


def test_choose_move_terminal_root_raises() -> None:
    root = new_root(Board.from_moves("1212121"))
    assert root.terminal and not root.legal
    with pytest.raises(ValueError, match="no search information"):
        choose_move(root, 1.0, np.random.default_rng(0))


def test_choose_from_priors_all_zero_raises() -> None:
    with pytest.raises(ValueError, match="no search information"):
        choose_from_priors(np.zeros(7), 1.0, np.random.default_rng(0))
