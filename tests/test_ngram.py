"""Checks for the n-gram models: the worked example of the Lecture 2 notes, and normalization."""
import math
from fractions import Fraction

import numpy as np

from tinylm.data import END
from tinylm.ngram import (BOS, NGramModel, KneserNeyModel, ngram_events, log_loss, perplexity,
                          unseen_fraction, good_turing_unseen, sample)

CORPUS = [s.split() for s in ["the cat sleeps", "the cat eats fish", "the dog sleeps", "a dog eats"]]
VOCAB = ["the", "a", "cat", "dog", "sleeps", "eats", "fish", END]


def sentence_prob(model, sentence):
    p = Fraction(1)
    for h, w in ngram_events(sentence.split(), model.n):
        p *= model.prob(h, w)
    return p


def test_events_pad_and_end():
    assert ngram_events(["a", "b"], 3) == [((BOS, BOS), "a"), ((BOS, "a"), "b"), (("a", "b"), END)]
    assert ngram_events(["a"], 1) == [((), "a"), ((), END)]


def test_maximum_likelihood_worked_example():
    m = NGramModel(CORPUS, 2, VOCAB, alpha=Fraction(0))
    assert sentence_prob(m, "the cat sleeps") == Fraction(1, 4)
    assert sentence_prob(m, "the dog eats") == Fraction(1, 16)
    assert sentence_prob(m, "the cat eats fish") == Fraction(1, 8)
    assert sentence_prob(m, "a cat sleeps") == 0


def test_add_one_worked_example():
    m = NGramModel(CORPUS, 2, VOCAB, alpha=Fraction(1))
    assert sentence_prob(m, "a cat sleeps") == Fraction(1, 900)
    assert sentence_prob(m, "the cat sleeps") == Fraction(3, 550)


def test_distributions_sum_to_one():
    rng = np.random.default_rng(0)
    words = ["a", "b", "c", "d"]
    seqs = [list(rng.choice(words, size=rng.integers(1, 8))) for _ in range(50)]
    vocab = words + [END]
    for model in [NGramModel(seqs, 3, vocab, alpha=0.1), KneserNeyModel(seqs, 3, vocab),
                  KneserNeyModel(seqs, 2, vocab, discount=0.5)]:
        for history in [(BOS, BOS), (BOS, "a"), ("c", "d"), ("d", "d"), ("x", "y")]:
            assert abs(sum(model.distribution(history).values()) - 1) < 1e-12


def test_kneser_ney_prefers_versatile_words():
    # "upon" is frequent but follows only "once"; "toy" is rarer but follows many words.
    seqs = [["once", "upon"]] * 20 + [[w, "toy"] for w in ["a", "the", "my", "his", "her", "one"]]
    vocab = ["once", "upon", "toy", "a", "the", "my", "his", "her", "one", "new", END]
    m = KneserNeyModel(seqs, 2, vocab)
    assert m.prob(("new",), "toy") > m.prob(("new",), "upon")      # "new" never seen as a history


def test_perplexity_of_uniform_model_is_vocabulary_size():
    vocab = ["a", "b", "c", END]
    m = NGramModel([], 2, vocab)                                   # no data: every history unseen
    assert abs(perplexity(m, [["a", "b"], ["c"]]) - 4) < 1e-12


def test_zero_probability_gives_infinite_loss():
    m = NGramModel(CORPUS, 2, VOCAB)
    assert log_loss(m, [["a", "cat", "sleeps"]]) == math.inf


def test_unseen_and_good_turing():
    assert unseen_fraction(CORPUS, CORPUS, 2) == 0
    assert unseen_fraction(CORPUS, [["a", "cat", "sleeps"]], 2) == 0.25        # "a cat" is new
    # bigrams in CORPUS: 17 events; seen once: (BOS,a), (the,dog), (a,dog), (cat,sleeps), (cat,eats),
    # (dog,sleeps), (dog,eats), (eats,fish), (eats,END), (fish,END) -> 10
    assert good_turing_unseen(CORPUS, 2) == 10 / 17


def test_sample_ends_and_stays_in_support():
    m = NGramModel(CORPUS, 2, VOCAB)
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = sample(m, rng)
        assert 2 <= len(s) <= 4 and s[0] in ("the", "a")


def test_tree_of_complete_sentences():
    # The worked example's tree (notes, Figure 2a): nine complete sentences at two depths, summing to one.
    m = NGramModel(CORPUS, 2, VOCAB, alpha=Fraction(0))

    def walk(prefix, p):
        for w, q in m.distribution((BOS,) + prefix).items():
            if q == 0:
                continue
            if w == END:
                yield prefix, p * q
            else:
                yield from walk(prefix + (w,), p * q)

    leaves = dict(walk((), Fraction(1)))
    assert len(leaves) == 9 and sum(leaves.values()) == 1
    assert {len(s) for s in leaves} == {3, 4}
    assert leaves[("the", "cat", "eats", "fish")] == Fraction(1, 8)
