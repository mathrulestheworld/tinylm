"""Layers built on the autograd engine: just enough for logistic regression and an MLP.

The interface follows ``torch.nn`` on purpose, so that moving to PyTorch in
Week 2 changes little besides the import.
"""
from __future__ import annotations

import numpy as np

from .autograd import Tensor


class Module:
    """Base class: collects the trainable tensors of a model and its sub-modules."""

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for value in vars(self).values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    """x @ W + b, with W of shape (in_features, out_features).

    Weights start at scale 1/sqrt(in_features), so that each output has roughly
    unit variance when the inputs do.
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.W = Tensor(rng.normal(0.0, 1.0 / np.sqrt(in_features), (in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class MLP(Module):
    """Linear layers separated by a nonlinearity: sizes = [d_in, h_1, ..., d_out]."""

    def __init__(self, sizes: list[int], activation: str = "tanh", rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.tanh() if self.activation == "tanh" else x.relu()
        return x
