"""Checks of the Week 3 translator's attention read (no data or checkpoints needed)."""
import torch
import torch.nn as nn

from tinylm.translator import Translator, attend_solution, PAD


def worked_example():
    """The read of the Lecture 3 notes, §6 ("One read by hand")."""
    W_a = torch.tensor([[1.0, 0.5], [0.0, -0.75]])
    U_a = torch.tensor([[0.5, 0.5], [-1.0, 1.0]])
    W = nn.Linear(2, 2, bias=False)
    v = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        W.weight.copy_(W_a.T)                       # nn.Linear computes x A^T, so A = W_a^T gives x W_a
        v.weight.copy_(torch.tensor([[2.0, 1.0]]))
    s = torch.tensor([[1.0, 2.0]])
    h = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    return s, h @ U_a, h, W, v


def test_worked_example():
    s, keys, h, W, v = worked_example()
    with torch.no_grad():
        context, weights = attend_solution(s, keys, h, torch.ones(1, 3, dtype=torch.bool), W, v)
    assert torch.allclose(weights[0], torch.tensor([0.579, 0.084, 0.337]), atol=1e-3)
    assert torch.allclose(context[0], torch.tensor([1.495, 0.421]), atol=1e-3)


def test_weights_are_distributions_and_padding_is_ignored():
    torch.manual_seed(0)
    B, n, H = 3, 6, 8
    W, v = nn.Linear(H, H, bias=False), nn.Linear(H, 1, bias=False)
    query, keys, values = torch.randn(B, H), torch.randn(B, n, H), torch.randn(B, n, H)
    lengths = torch.tensor([6, 4, 1])
    mask = torch.arange(n)[None, :] < lengths[:, None]
    with torch.no_grad():
        context, weights = attend_solution(query, keys, values, mask, W, v)
        assert context.shape == (B, H) and weights.shape == (B, n)
        assert torch.allclose(weights.sum(-1), torch.ones(B), atol=1e-6)
        assert (weights[~mask] == 0).all()
        assert torch.allclose(context[2], values[2, 0], atol=1e-6)          # one real token: weight 1 on it
        # changing the values at padded positions changes nothing
        values2 = values.clone()
        values2[~mask] = 100.0
        context2, _ = attend_solution(query, keys, values2, mask, W, v)
        assert torch.allclose(context2, context, atol=1e-6)


def test_translator_shapes_and_padding():
    torch.manual_seed(0)
    model = Translator(20, 30, attention=True, E=8, H=16)
    model.eval()
    src = torch.tensor([[5, 6, 7, 8], [9, 10, PAD, PAD]])
    tgt = torch.tensor([[4, 5, 6], [7, 8, 9]])
    with torch.no_grad():
        scores, weights = model(src, tgt)
        assert scores.shape == (2, 3, 30) and weights.shape == (2, 3, 4)
        assert (weights[1, :, 2:] == 0).all()
        # the second sentence alone, without padding, gets the same scores
        alone, _ = model(src[1:, :2], tgt[1:])
        assert torch.allclose(alone[0], scores[1], atol=1e-5)
