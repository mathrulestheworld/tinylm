"""Layers built on the autograd engine: enough for logistic regression and a small network.

The interface copies torch.nn, so moving to PyTorch in Week 2 changes little
besides the import.
"""
import numpy as np

from .autograd import Tensor


class Module:
    """A model, or a piece of one. parameters() finds every trainable tensor inside it."""

    def parameters(self):
        params = []
        for value in vars(self).values():                   # the attributes of this module
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_parameters(self):
        return sum(p.data.size for p in self.parameters())

    def __call__(self, x):
        return self.forward(x)                              # model(x) runs model.forward(x), as in PyTorch

    def forward(self, x):
        raise NotImplementedError("each layer or model defines its own forward")


class Linear(Module):
    """x @ W + b, where W has shape (in_features, out_features) and b has shape (out_features,)."""

    def __init__(self, in_features, out_features, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        scale = 1 / np.sqrt(in_features)                    # keeps outputs at about unit variance
        self.W = Tensor(rng.normal(0, scale, (in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.b


class MLP(Module):
    """Linear layers with a nonlinearity (tanh or relu) between them.

    sizes = [inputs, hidden_1, ..., outputs]; for example MLP([2, 16, 1]).
    """

    def __init__(self, sizes, activation="tanh", rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.layers.append(Linear(n_in, n_out, rng))
        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            is_last = i == len(self.layers) - 1
            if not is_last:
                x = x.tanh() if self.activation == "tanh" else x.relu()
        return x
