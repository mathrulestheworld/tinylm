"""Checks of the Week 5 pretraining code (tinylm/pretrain.py): batches, schedules, optimizers, the loop, and
the base model in checkpoints/."""
import math
import pathlib

import torch

from tinylm.pretrain import (BASE_MODEL, load_lm, fast_attention, pack, get_batch, lr_at, newton_schulz, Muon,
                             make_optimizer, pretrain, stories_loss, token_losses, sample)
from tinylm.transformer import TransformerLM, attention_solution, count_parameters

CKPT = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"


def test_pack_and_batches():
    stream = pack([[5, 6, 7], [8, 9]], eot=0)
    assert stream.tolist() == [0, 5, 6, 7, 0, 8, 9]
    long = torch.arange(1000)
    x, y = get_batch(long, 4, 16, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(y[:, :-1], x[:, 1:]) and torch.equal(y, x + 1)       # targets: the next token


def test_schedules():
    assert math.isclose(lr_at(50, 1000, 1.0, 100), 0.5)                     # warmup
    assert math.isclose(lr_at(100, 1000, 1.0, 100), 1.0)
    assert math.isclose(lr_at(1000, 1000, 1.0, 100), 0.1)                    # cosine to a tenth
    assert math.isclose(lr_at(550, 1000, 1.0, 100), 0.55)                    # halfway: the middle
    assert lr_at(700, 1000, 1.0, 100, "constant") == 1.0
    assert lr_at(800, 1000, 1.0, 100, "wsd") == 1.0 and math.isclose(lr_at(1000, 1000, 1.0, 100, "wsd"), 0.1)


def test_newton_schulz_flattens_the_singular_values():
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(32, 32))
    V, _ = torch.linalg.qr(torch.randn(16, 16))
    G = U[:, :16] @ torch.diag(torch.logspace(-2, 0, 16)) @ V.T             # singular values from 0.01 to 1
    s = torch.linalg.svdvals(newton_schulz(G))
    assert s.min() > 0.6 and s.max() < 1.25
    assert torch.allclose(torch.linalg.svdvals(newton_schulz(G.T)), s, atol=1e-4)


def test_adamw_decays_matrices_only():
    model = TransformerLM(20, d_model=8, n_layers=1, n_heads=2, max_len=8)
    opt = make_optimizer(model, "adamw", 1e-3, weight_decay=0.1)
    for g in opt.param_groups:
        assert all(p.dim() >= 2 for p in g["params"]) == (g["weight_decay"] > 0)
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == count_parameters(model)


def test_every_optimizer_learns_a_repeating_stream():
    stream = torch.tensor([1, 2, 3, 4, 5, 6, 7] * 200)
    for kind, lr in [("sgd", 1.0), ("momentum", 0.3), ("adam", 0.01), ("adamw", 0.01), ("muon", 0.02)]:
        torch.manual_seed(0)
        model = TransformerLM(10, d_model=16, n_layers=1, n_heads=2, max_len=16)
        model.set_attention(fast_attention)
        curve = pretrain(model, stream, 60, batch_size=8, length=16, optimizer=kind, peak=lr, warmup=5, log=None)
        assert curve[-1][1] < 0.5 * curve[0][1], (kind, curve[0], curve[-1])


def test_fast_attention_equals_the_written_one():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 9, 4), torch.randn(2, 3, 9, 4), torch.randn(2, 3, 9, 4)
    assert torch.allclose(fast_attention(q, k, v)[0], attention_solution(q, k, v)[0], atol=1e-5)


def test_stories_loss_reads_long_stories_in_windows():
    torch.manual_seed(0)
    model = TransformerLM(30, d_model=16, n_layers=1, n_heads=2, max_len=64)
    short = TransformerLM(30, d_model=16, n_layers=1, n_heads=2, max_len=16)
    short.load_state_dict(model.state_dict(), strict=False)
    story = torch.randint(1, 30, (12,)).tolist()                 # fits in both contexts
    assert math.isclose(stories_loss(model, [story], 0)[0], stories_loss(short, [story], 0)[0], rel_tol=1e-5)
    total, count = stories_loss(short, [torch.randint(1, 30, (40,)).tolist()], 0)
    assert count == 41                                            # every token and the end, predicted once


def test_the_base_model():
    model = load_lm(CKPT / "week05_base.pt")
    assert count_parameters(model) == 3_456_000
    assert model.unembed.weight is model.tok.weight
    ids = [4088] + [1001, 266, 271]
    assert token_losses(model, ids).shape == (3,)
    out = sample(model, ids, 5, temperature=0)
    assert out == sample(model, ids, 5, temperature=0)            # greedy is deterministic
    assert len(sample(model, ids, 50, temperature=1.0, stop=tuple(range(4096)))) == 1
