"""Optimizers: they update the parameters using the gradients that backward() computed.

The interface copies torch.optim: zero_grad(), then loss.backward(), then step().
"""
import numpy as np


class SGD:
    """Gradient descent: p <- p - lr * p.grad, for every parameter p."""

    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad


class Adam:
    """Adam (Kingma and Ba, 2015).

    For each parameter it keeps running averages of the gradient (m) and of the
    squared gradient (v), and moves each coordinate by about lr in the direction
    m / sqrt(v): large, consistent gradients and small, noisy ones get steps of
    similar size.
    """

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros(p.shape) for p in self.params]
        self.v = [np.zeros(p.shape) for p in self.params]
        self.t = 0                                          # number of steps taken

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            # the averages start at 0, so early on they are too small; these divisions correct that
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
