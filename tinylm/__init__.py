"""tinylm: a small language model built from scratch, one component per week.

Week 1 provides the autograd engine (tinylm/autograd.py), layers (nn.py), and
optimizers (optim.py). Later weeks add tokenizers, n-gram models, a
Transformer, training, fine-tuning, and inference.
"""
from .autograd import Tensor, cross_entropy, binary_cross_entropy_with_logits, gradcheck

__version__ = "0.1.0"
