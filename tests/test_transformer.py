"""Checks of the Week 4 Transformer (no data or checkpoints needed)."""
import math

import torch

from tinylm.transformer import (TransformerLM, attention_solution, leaky_attention, count_parameters,
                                log_probs, prefix_log_probs)


def hand_example():
    """The head of the Lecture 4 notes, §3 ("One head by hand"), on "lily saw tom . she"."""
    X = torch.tensor([[1., 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 1]])
    W_Q = torch.tensor([[0., 0], [0, 0], [2, 0], [0, 0], [2, 2]])
    W_K = torch.tensor([[1., 0], [0, 1], [0, 0], [0, 0], [0, 0]])
    W_V = torch.tensor([[1., -1], [0, 2], [0, 0], [0, 0], [0, -2]])
    return (X @ W_Q)[None, None], (X @ W_K)[None, None], (X @ W_V)[None, None]


def test_hand_example():
    q, k, v = hand_example()
    read, weights = attention_solution(q, k, v)
    she = weights[0, 0, 4]
    assert torch.allclose(she, torch.tensor([0.62, 0.04, 0.15, 0.04, 0.15]), atol=0.006)
    assert torch.allclose(read[0, 0, 4], torch.tensor([0.77, 0.47]), atol=0.006)


def test_weights_are_causal_distributions():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 7, 4), torch.randn(2, 3, 7, 4), torch.randn(2, 3, 7, 4)
    read, weights = attention_solution(q, k, v)
    assert read.shape == (2, 3, 7, 4) and weights.shape == (2, 3, 7, 7)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 3, 7), atol=1e-6)
    assert (torch.triu(weights, diagonal=1) == 0).all()
    assert torch.allclose(read[:, :, 0], v[:, :, 0], atol=1e-6)              # position 0 reads only itself


def test_changing_the_future_changes_nothing_earlier():
    torch.manual_seed(0)
    model = TransformerLM(50, d_model=16, n_layers=2, n_heads=2, max_len=12)
    model.eval()
    x = torch.randint(0, 50, (1, 12))
    x2 = x.clone()
    x2[0, 7] = (x[0, 7] + 1) % 50
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert torch.allclose(a[0, :7], b[0, :7], atol=1e-6)
    assert not torch.allclose(a[0, 7], b[0, 7], atol=1e-3)


def test_one_pass_equals_one_prefix_at_a_time():
    torch.manual_seed(0)
    model = TransformerLM(50, d_model=16, n_layers=2, n_heads=2, max_len=12)
    model.eval()
    x = torch.randint(0, 50, (12,))
    y = torch.randint(0, 50, (12,))
    assert torch.allclose(log_probs(model, x, y), prefix_log_probs(model, x, y), atol=1e-5)
    model.set_attention(leaky_attention)
    assert not torch.allclose(log_probs(model, x, y), prefix_log_probs(model, x, y), atol=1e-3)


def test_parameter_counts():
    model = TransformerLM(8682, d_model=128, n_layers=4, n_heads=4, max_len=256, no_output=8681)
    assert count_parameters(model) == 1904640
    # GPT-2 small, built without allocating its weights
    with torch.device("meta"):
        gpt2 = TransformerLM(50257, d_model=768, n_layers=12, n_heads=12, max_len=1024, positions="learned")
    assert count_parameters(gpt2) == 124439808
    assert 12 * 12 * 768 ** 2 == 84934656


def test_no_output_and_shapes():
    model = TransformerLM(30, d_model=8, n_layers=1, n_heads=2, max_len=5, no_output=29)
    logits = model(torch.randint(0, 30, (3, 5)))
    assert logits.shape == (3, 5, 30)
    assert torch.isinf(logits[..., 29]).all()
    assert math.isclose(torch.softmax(logits, -1)[..., :29].sum(-1).mean().item(), 1.0, abs_tol=1e-6)
