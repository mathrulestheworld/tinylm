"""Checks for the embedding code: PMI by hand, SVD, similarity search, and skip-gram."""
import numpy as np
import torch

from tinylm.embeddings import (cooccurrence_counts, pmi, ppmi, svd_vectors, nearest, analogy, top_k,
                               SkipGram, skipgram_pairs, frequent_words)


def test_cooccurrence_and_pmi():
    seqs = [["a", "b", "c"], ["a", "b"]]
    n = cooccurrence_counts(seqs, ["a", "b", "c"], window=1)
    assert n.tolist() == [[0, 2, 0], [2, 0, 1], [0, 1, 0]]
    m = pmi(n)
    total = n.sum()                                                     # 6
    assert np.isclose(m[0, 1], np.log((2 / total) / ((2 / total) * (3 / total))))
    assert np.isneginf(m[0, 2]) and ppmi(n)[0, 2] == 0


def test_svd_vectors_reconstruct():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 6))              # rank 2
    V = svd_vectors(A @ A.T, 2)                                         # symmetric PSD: V V^T = A A^T
    assert np.allclose(V @ V.T, A @ A.T)


def test_nearest_analogy_and_top_k():
    words = ["king", "queen", "man", "woman", "apple"]
    vecs = np.array([[1, 1, 0], [1, -1, 0], [0, 1, 0.1], [0, -1, 0.1], [0, 0, 1.0]])
    assert nearest(vecs, words, "king", 1)[0][0] in ("man", "queen")
    assert analogy(vecs, words, "man", "woman", "king", 1)[0][0] == "queen"
    M = np.eye(4)
    assert list(top_k(M, np.array([0.1, 0.9, 0.5, 0.0]), 2)) == [1, 2]


def test_skipgram_loss_and_pairs():
    torch.manual_seed(0)
    model = SkipGram(10, 4)
    centers = torch.tensor([0, 1]); contexts = torch.tensor([2, 3]); neg = torch.tensor([[4, 5], [6, 7]])
    # context vectors start at zero, so every score is 0 and each term is log(1/2)
    assert torch.isclose(model.loss(centers, contexts, neg), torch.tensor(3 * np.log(2.0), dtype=torch.float))
    words, counts = frequent_words([["a", "b", "a", "c"]])
    index = {w: i for i, w in enumerate(words)}
    pairs = skipgram_pairs([["a", "b", "c"]], index, np.ones(len(words)), np.random.default_rng(0), max_window=1)
    assert sorted(map(tuple, pairs.tolist())) == sorted([(index["a"], index["b"]), (index["b"], index["a"]),
                                                         (index["b"], index["c"]), (index["c"], index["b"])])
