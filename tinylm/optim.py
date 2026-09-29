"""Gradient-based optimizers for lists of tensors (the ``torch.optim`` interface, in miniature)."""
from __future__ import annotations

import numpy as np

from .autograd import Tensor


class SGD:
    """Plain gradient descent: p <- p - lr * grad."""

    def __init__(self, params: list[Tensor], lr: float = 0.1):
        self.params, self.lr = list(params), lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad


class Adam(SGD):
    """Adam (Kingma and Ba, 2015): per-coordinate steps scaled by running moment estimates."""

    def __init__(self, params: list[Tensor], lr: float = 1e-2, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)
