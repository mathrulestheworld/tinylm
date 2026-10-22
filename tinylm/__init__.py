"""tinylm: a small language model built from scratch, one component per week.

Week 1 provides the autograd engine (tinylm/autograd.py), layers (nn.py), and
optimizers (optim.py). Week 2 adds text data (data.py), n-gram models and the
evaluation harness (ngram.py), word embeddings (embeddings.py), and neural
language models in PyTorch (neural.py). Week 3 adds an English-to-French
translator with attention (translator.py). Week 4 adds a decoder-only
Transformer language model (transformer.py). Later weeks add tokenizers,
pretraining, fine-tuning, and inference.
"""
from .autograd import Tensor, cross_entropy, binary_cross_entropy_with_logits, gradcheck

__version__ = "0.1.0"
