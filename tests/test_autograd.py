"""Checks for the Week 1 autograd engine.

Every operation is compared with central finite differences, including cases
where broadcasting stretches an input and where one tensor is used twice. If
PyTorch is installed, gradients are also compared with ``torch.autograd``.
"""
import numpy as np
import pytest

from tinylm.autograd import Tensor, binary_cross_entropy_with_logits, cross_entropy, gradcheck
from tinylm.nn import MLP, Linear
from tinylm.optim import SGD, Adam

rng = np.random.default_rng(0)
TOL = 1e-6


def t(*shape, positive=False):
    x = rng.normal(size=shape)
    return Tensor(np.abs(x) + 0.5 if positive else x, requires_grad=True)


UNARY = {
    "exp": lambda a: a.exp().sum(),
    "log": lambda a: a.log().sum(),
    "tanh": lambda a: a.tanh().sum(),
    "sigmoid": lambda a: a.sigmoid().sum(),
    "relu": lambda a: (a.relu() * a).sum(),
    "pow": lambda a: (a ** 3).sum(),
    "sqrt": lambda a: (a ** 0.5).sum(),
    "neg": lambda a: (-a * a).sum(),
    "sum_axis": lambda a: (a.sum(axis=0) ** 2).sum(),
    "mean_keepdims": lambda a: (a.mean(axis=1, keepdims=True) * a).sum(),
    "reshape": lambda a: (a.reshape(-1) * a.reshape(-1)).sum(),
    "transpose": lambda a: (a.transpose() @ a).sum(),
    "index": lambda a: (a[np.array([0, 0, 1])] ** 2).sum(),      # repeated index must accumulate
    "logsumexp": lambda a: a.logsumexp(axis=-1).sum(),
    "log_softmax": lambda a: (a.log_softmax(axis=-1) * a).sum(),
    "softmax": lambda a: (a.softmax(axis=0) ** 2).sum(),
    "reused": lambda a: (a * a * a + a).sum(),                   # one tensor used several times
}


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary(name):
    a = t(3, 4, positive=name in {"log", "sqrt"})
    assert gradcheck(lambda: UNARY[name](a), [a]) < TOL


BINARY = {
    "add": lambda a, b: ((a + b) ** 2).sum(),
    "sub": lambda a, b: ((a - b) ** 2).sum(),
    "mul": lambda a, b: (a * b).sum(),
    "div": lambda a, b: (a / b).sum(),
}


@pytest.mark.parametrize("name", sorted(BINARY))
@pytest.mark.parametrize("shapes", [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 1))])
def test_binary_broadcasting(name, shapes):
    a = t(*shapes[0])
    b = t(*shapes[1], positive=name == "div")
    assert gradcheck(lambda: BINARY[name](a, b), [a, b]) < TOL


@pytest.mark.parametrize("shapes", [((3, 4), (4, 5)), ((1, 4), (4, 1)), ((5, 1), (1, 3))])
def test_matmul(shapes):
    a, b = t(*shapes[0]), t(*shapes[1])
    assert gradcheck(lambda: ((a @ b).tanh()).sum(), [a, b]) < TOL


def test_matmul_needs_matrices():
    with pytest.raises(ValueError):
        t(3) @ t(3, 2)


def test_scalar_operands():
    a = t(3)
    assert gradcheck(lambda: (2.0 * a + 1.0 - a / 3.0 + 1.0 / (a * a + 1.0)).sum(), [a]) < TOL


def test_cross_entropy():
    logits = t(5, 3)
    y = np.array([0, 2, 1, 1, 0])
    assert gradcheck(lambda: cross_entropy(logits, y), [logits]) < TOL
    p = np.exp(logits.data) / np.exp(logits.data).sum(axis=1, keepdims=True)
    assert np.isclose(cross_entropy(logits, y).item(), -np.log(p[np.arange(5), y]).mean())


def test_binary_cross_entropy_is_stable_and_correct():
    z = Tensor(np.array([-800.0, -2.0, 0.0, 3.0, 800.0]), requires_grad=True)
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    loss = binary_cross_entropy_with_logits(z, y)
    assert np.isfinite(loss.item())
    zz = Tensor(np.array([-2.0, 0.5, 3.0]), requires_grad=True)
    yy = np.array([0.0, 1.0, 1.0])
    s = 1 / (1 + np.exp(-zz.data))
    expected = -(yy * np.log(s) + (1 - yy) * np.log(1 - s)).mean()
    assert np.isclose(binary_cross_entropy_with_logits(zz, yy).item(), expected)
    assert gradcheck(lambda: binary_cross_entropy_with_logits(zz, yy), [zz]) < TOL


def test_gradients_accumulate_across_backward_calls():
    a = t(3)
    (a * 2).sum().backward()
    (a * 3).sum().backward()
    assert np.allclose(a.grad, 5.0)


def test_only_leaves_keep_gradients():
    a = t(3)
    h = a * 2
    loss = (h * h).sum()
    loss.backward()
    assert h.grad is None and loss.grad is None
    assert np.allclose(a.grad, 8 * a.data)
    loss.backward()                                    # a second pass over the same graph adds again
    assert np.allclose(a.grad, 16 * a.data)


def test_backward_needs_a_tensor_that_requires_grad():
    with pytest.raises(RuntimeError):
        (Tensor([1.0, 2.0]) * 2).sum().backward()


def test_non_scalar_backward_needs_a_gradient():
    a = t(3)
    with pytest.raises(RuntimeError):
        (a * 2).backward()


def test_a_long_chain_of_operations():
    a = Tensor(np.array(0.5), requires_grad=True)
    x = a
    for _ in range(500):
        x = x * 1.001
    x.backward()
    assert np.isclose(a.grad, 1.001 ** 500)


def test_mlp_learns_xor():
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float))
    y = np.array([0, 1, 1, 0])
    model = MLP([2, 8, 2], rng=np.random.default_rng(1))
    opt = Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        loss = cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert (model(X).data.argmax(axis=1) == y).all()


def test_sgd_step_and_parameter_count():
    layer = Linear(3, 2)
    assert layer.num_parameters() == 3 * 2 + 2
    x = Tensor(np.ones((1, 3)))
    loss = (layer(x) ** 2).sum()
    loss.backward()
    before = layer.W.data.copy()
    SGD(layer.parameters(), lr=0.1).step()
    assert np.allclose(layer.W.data, before - 0.1 * layer.W.grad)


torch = pytest.importorskip("torch", reason="PyTorch comparison skipped: torch is not installed")


def test_matches_pytorch_on_an_mlp_loss():
    X = rng.normal(size=(16, 5))
    y = rng.integers(0, 3, size=16)
    W1, b1 = rng.normal(size=(5, 7)), rng.normal(size=7)
    W2, b2 = rng.normal(size=(7, 3)), rng.normal(size=3)

    ours = [Tensor(w, requires_grad=True) for w in (W1, b1, W2, b2)]
    h = (Tensor(X) @ ours[0] + ours[1]).tanh()
    cross_entropy(h @ ours[2] + ours[3], y).backward()

    theirs = [torch.tensor(w, requires_grad=True) for w in (W1, b1, W2, b2)]
    th = torch.tanh(torch.tensor(X) @ theirs[0] + theirs[1])
    torch.nn.functional.cross_entropy(th @ theirs[2] + theirs[3], torch.tensor(y)).backward()

    for a, b in zip(ours, theirs):
        assert np.allclose(a.grad, b.grad.numpy(), atol=1e-10)
