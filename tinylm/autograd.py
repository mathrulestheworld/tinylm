"""A small reverse-mode automatic differentiation engine on NumPy arrays.

A :class:`Tensor` holds an array and, after :meth:`Tensor.backward`, the
gradient of a scalar output with respect to it. Every operation records its
inputs and a *vector-Jacobian product*: a function that maps the gradient of
the operation's output to the gradients of its inputs. ``backward`` visits the
recorded graph once, in reverse topological order, applying these functions
and adding up contributions when a tensor is used more than once.

This is the whole idea behind the autograd of PyTorch or JAX. From Week 2 on
the course uses PyTorch, whose interface this engine imitates.

Example::

    >>> from tinylm.autograd import Tensor
    >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = (x * x).sum()          # y = sum of squares
    >>> y.backward()
    >>> x.grad                     # dy/dx = 2x
    array([2., 4., 6.])
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

ArrayLike = "Tensor | np.ndarray | float | int | list"


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` over the axes that broadcasting added or stretched, so it has ``shape``.

    If ``a`` of shape (3, 1) is added to ``b`` of shape (4,), both are broadcast
    to (3, 4). The gradient for ``a`` must be summed back to (3, 1): each entry
    of ``a`` was used four times.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """An array that remembers how it was computed.

    Attributes:
        data: the value, a float64 NumPy array.
        grad: after ``backward()``, the gradient of the output with respect to
            this tensor (same shape as ``data``), or ``None``.
        requires_grad: whether gradients flow to this tensor.
    """

    def __init__(self, data, requires_grad: bool = False, _parents: tuple = (), _op: str = ""):
        self.data = np.asarray(data.data if isinstance(data, Tensor) else data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._vjp: Callable[[np.ndarray], tuple] | None = None
        self._op = _op

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return float(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4)}{grad})"

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _lift(x) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x)

    @staticmethod
    def _make(data, parents: tuple, vjp: Callable, op: str) -> "Tensor":
        """Create the output of an operation and record how to differentiate it."""
        out = Tensor(data, requires_grad=any(p.requires_grad for p in parents), _parents=parents, _op=op)
        if out.requires_grad:
            out._vjp = vjp
        return out

    # -------------------------------------------------------------- arithmetic
    def __add__(self, other) -> "Tensor":
        other = self._lift(other)
        return self._make(self.data + other.data, (self, other),
                          lambda g: (_unbroadcast(g, self.shape), _unbroadcast(g, other.shape)), "+")

    def __mul__(self, other) -> "Tensor":
        other = self._lift(other)
        return self._make(self.data * other.data, (self, other),
                          lambda g: (_unbroadcast(g * other.data, self.shape),
                                     _unbroadcast(g * self.data, other.shape)), "*")

    def __truediv__(self, other) -> "Tensor":
        other = self._lift(other)
        return self._make(self.data / other.data, (self, other),
                          lambda g: (_unbroadcast(g / other.data, self.shape),
                                     _unbroadcast(-g * self.data / other.data ** 2, other.shape)), "/")

    def __pow__(self, exponent: float) -> "Tensor":
        if isinstance(exponent, Tensor):
            raise TypeError("only constant exponents are supported; use exp(b * log(a)) otherwise")
        return self._make(self.data ** exponent, (self,),
                          lambda g: (g * exponent * self.data ** (exponent - 1),), f"**{exponent}")

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-self._lift(other))

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __rsub__(self, other) -> "Tensor":
        return self._lift(other) - self

    def __rtruediv__(self, other) -> "Tensor":
        return self._lift(other) / self

    def __matmul__(self, other) -> "Tensor":
        """Matrix product of tensors with at least two dimensions (batched over leading ones)."""
        other = self._lift(other)
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul needs tensors with at least two dimensions; reshape vectors to (1, n) or (n, 1)")

        def vjp(g):
            ga = g @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ g
            return _unbroadcast(ga, self.shape), _unbroadcast(gb, other.shape)
        return self._make(self.data @ other.data, (self, other), vjp, "@")

    # -------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        def vjp(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, self.shape).copy(),)
        return self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), vjp, "sum")

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        count = self.data.size if axis is None else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) / float(count)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        """Maximum; the gradient is shared equally among tied maximal entries."""
        out = self.data.max(axis=axis, keepdims=True)

        def vjp(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            mask = self.data == out
            return (mask * g / mask.sum(axis=axis, keepdims=True),)
        result = out if keepdims else (out.squeeze(axis) if axis is not None else out.reshape(()))
        return self._make(result, (self,), vjp, "max")

    # ------------------------------------------------------- elementwise maps
    def exp(self) -> "Tensor":
        out = np.exp(self.data)
        return self._make(out, (self,), lambda g: (g * out,), "exp")

    def log(self) -> "Tensor":
        return self._make(np.log(self.data), (self,), lambda g: (g / self.data,), "log")

    def tanh(self) -> "Tensor":
        out = np.tanh(self.data)
        return self._make(out, (self,), lambda g: (g * (1.0 - out ** 2),), "tanh")

    def sigmoid(self) -> "Tensor":
        out = 0.5 * (1.0 + np.tanh(0.5 * self.data))       # stable form of 1 / (1 + e^{-x})
        return self._make(out, (self,), lambda g: (g * out * (1.0 - out),), "sigmoid")

    def relu(self) -> "Tensor":
        return self._make(np.maximum(self.data, 0.0), (self,), lambda g: (g * (self.data > 0),), "relu")

    # ------------------------------------------------------------------ shapes
    def reshape(self, *shape) -> "Tensor":
        return self._make(self.data.reshape(*shape), (self,), lambda g: (g.reshape(self.shape),), "reshape")

    @property
    def T(self) -> "Tensor":
        """Transpose of the last two axes."""
        return self._make(np.swapaxes(self.data, -1, -2), (self,), lambda g: (np.swapaxes(g, -1, -2),), "T")

    def __getitem__(self, index) -> "Tensor":
        index = index.data.astype(int) if isinstance(index, Tensor) else index

        def vjp(g):
            full = np.zeros_like(self.data)
            np.add.at(full, index, g)        # add.at, not +=, so that repeated indices accumulate
            return (full,)
        return self._make(self.data[index], (self,), vjp, "index")

    # ------------------------------------------------------ softmax and losses
    def logsumexp(self, axis: int = -1, keepdims: bool = False) -> "Tensor":
        """log(sum(exp(x))) along ``axis``, computed stably by subtracting the maximum."""
        m = self.data.max(axis=axis, keepdims=True)
        out = m + np.log(np.exp(self.data - m).sum(axis=axis, keepdims=True))
        softmax = np.exp(self.data - out)

        def vjp(g):
            if not keepdims:
                g = np.expand_dims(g, axis)
            return (g * softmax,)
        return self._make(out if keepdims else out.squeeze(axis), (self,), vjp, "logsumexp")

    def log_softmax(self, axis: int = -1) -> "Tensor":
        return self - self.logsumexp(axis=axis, keepdims=True)

    def softmax(self, axis: int = -1) -> "Tensor":
        return self.log_softmax(axis=axis).exp()

    # ---------------------------------------------------------------- backward
    def backward(self, grad=None) -> None:
        """Compute gradients of this tensor with respect to every tensor it depends on.

        ``grad`` is the gradient flowing in from above; it defaults to 1 and may
        be omitted only when this tensor is a scalar. Gradients are *added* to
        ``.grad``, as in PyTorch, so call ``zero_grad`` between steps.
        """
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() needs an explicit gradient for a non-scalar tensor")
            grad = np.ones_like(self.data)
        order = self._topological_order()
        grads = {id(self): np.asarray(grad, dtype=np.float64)}
        for node in order:                       # outputs before inputs
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if not node._parents:                # a leaf: store the gradient
                node.grad = g if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._vjp(g)):
                if parent.requires_grad:
                    grads[id(parent)] = grads[id(parent)] + pg if id(parent) in grads else pg

    def _topological_order(self) -> list["Tensor"]:
        """Tensors reachable from self, each after every tensor that uses it (iterative DFS)."""
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen or not node.requires_grad:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents)
        return order[::-1]

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())


# ---------------------------------------------------------------------- helpers
def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean negative log-likelihood of integer ``targets`` under softmax(``logits``).

    ``logits`` has shape (batch, classes); ``targets`` holds class indices.
    """
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets, dtype=int)
    logp = logits.log_softmax(axis=-1)
    return -logp[np.arange(len(targets)), targets].mean()


def binary_cross_entropy_with_logits(logits: Tensor, targets) -> Tensor:
    """Mean of -[y log sigma(z) + (1-y) log(1 - sigma(z))], computed stably from the logits z."""
    y = Tensor(targets)
    # log sigma(z) = -softplus(-z) and log(1 - sigma(z)) = -softplus(z); softplus(z) = logsumexp([0, z])
    zero = Tensor(np.zeros_like(logits.data))
    stacked_pos = _stack_last(zero, logits)            # [0, z]
    stacked_neg = _stack_last(zero, -logits)           # [0, -z]
    return (y * stacked_neg.logsumexp(axis=-1) + (1 - y) * stacked_pos.logsumexp(axis=-1)).mean()


def _stack_last(a: Tensor, b: Tensor) -> Tensor:
    """Stack two tensors of the same shape along a new last axis."""
    def vjp(g):
        return g[..., 0], g[..., 1]
    return Tensor._make(np.stack([a.data, b.data], axis=-1), (a, b), vjp, "stack")


def numerical_grad(f: Callable[[], Tensor], x: Tensor, eps: float = 1e-6) -> np.ndarray:
    """Central-difference estimate of d f() / d x, perturbing one entry of ``x.data`` at a time."""
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x.data[i]
        x.data[i] = old + eps
        up = f().item()
        x.data[i] = old - eps
        down = f().item()
        x.data[i] = old
        grad[i] = (up - down) / (2 * eps)
    return grad


def gradcheck(f: Callable[[], Tensor], inputs: Iterable[Tensor], eps: float = 1e-6) -> float:
    """Largest relative error between autograd and finite-difference gradients of scalar ``f``."""
    inputs = list(inputs)
    for x in inputs:
        x.zero_grad()
    f().backward()
    worst = 0.0
    for x in inputs:
        num = numerical_grad(f, x, eps)
        auto = x.grad if x.grad is not None else np.zeros_like(x.data)   # None: f does not depend on x
        err = np.abs(auto - num) / np.maximum(1.0, np.abs(num))
        worst = max(worst, float(err.max()))
    return worst
