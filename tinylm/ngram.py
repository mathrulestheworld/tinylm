"""n-gram language models estimated by counting (Week 2).

A sequence is a list of tokens (words or characters). An n-gram model predicts each token
from the previous n-1 tokens, its history; the start of a sequence is padded with BOS and
the end-of-sequence token END is predicted after the last token, so that

    q(x_1 ... x_T, END) = prod_t q(x_t | previous n-1 tokens).

Two estimators:

  NGramModel       relative frequencies with add-alpha smoothing (alpha = 0 is maximum likelihood)
  KneserNeyModel   interpolated Kneser-Ney with one discount D

and an evaluation harness: log_loss (nats per predicted token, END included) and perplexity.
"""
import math

from .data import END

BOS = "<s>"        # padding before the first token; a context, never predicted


def ngram_events(seq, n):
    """The prediction events of one sequence: a list of (history, next token) pairs.

    history is a tuple of the previous n-1 tokens (BOS-padded at the start); the last
    event predicts END. A sequence of T tokens gives T + 1 events.
    """
    padded = [BOS] * (n - 1) + list(seq) + [END]
    events = []
    for i in range(n - 1, len(padded)):
        history = tuple(padded[i - n + 1:i])
        events.append((history, padded[i]))
    return events


def last_tokens(history, k):
    """The last k tokens of history, as a tuple (empty for k = 0)."""
    if k == 0:
        return ()
    return tuple(history[len(history) - k:])


def count_events(sequences, n):
    """counts[(history, token)] = number of times token follows history."""
    counts = {}
    for seq in sequences:
        for event in ngram_events(seq, n):
            counts[event] = counts.get(event, 0) + 1
    return counts


class NGramModel:
    """An n-gram model with add-alpha smoothing:

        q(w | h) = (c(h, w) + alpha) / (c(h) + alpha |V|),

    where c(h) is the number of events with history h and V the vocabulary of predicted tokens
    (it contains END but not BOS). With alpha = 0 this is the maximum-likelihood estimate, the
    relative frequency c(h, w) / c(h). A history never seen in training has no evidence at all;
    the model then predicts uniformly.
    """

    def __init__(self, sequences, n, vocab, alpha=0.0):
        self.n = n
        self.vocab = list(vocab)
        self.alpha = alpha
        self.counts = count_events(sequences, n)
        self.history_counts = {}
        for (h, w), c in self.counts.items():
            self.history_counts[h] = self.history_counts.get(h, 0) + c

    def prob(self, history, token):
        h = last_tokens(history, self.n - 1)
        c_h = self.history_counts.get(h, 0)
        if c_h == 0:
            return 1.0 / len(self.vocab)
        c_hw = self.counts.get((h, token), 0)
        return (c_hw + self.alpha) / (c_h + self.alpha * len(self.vocab))

    def distribution(self, history):
        """The whole next-token distribution after history, as a dict token -> probability."""
        dist = {}
        for w in self.vocab:
            dist[w] = self.prob(history, w)
        return dist


class KneserNeyModel:
    """Interpolated Kneser-Ney smoothing (Kneser and Ney, 1995) with a single discount D.

    At the highest order, every positive count is reduced by D and the freed mass,
    D * (number of distinct tokens seen after h) / c(h), goes to the next lower order:

        q(w | h) = max(c(h, w) - D, 0) / c(h) + D * N(h .) / c(h) * q_lower(w | shorter h).

    The lower orders use the same formula with *continuation counts* in place of counts:
    N(. h w) = the number of distinct tokens v for which (v h w) occurs. A word that follows
    many different contexts ("toy") is a better guess in a new context than one that is
    frequent only after one context ("upon", after "once"). Below the unigram level the
    model is uniform over the vocabulary. A history with no counts passes all its mass down.
    """

    def __init__(self, sequences, n, vocab, discount=0.75):
        if n < 2:
            raise ValueError("Kneser-Ney needs n >= 2")
        self.n = n
        self.vocab = list(vocab)
        self.discount = discount
        raw = {}
        for k in range(2, n + 1):
            raw[k] = count_events(sequences, k)
        self.tables = {n: self.summarize(raw[n])}
        for k in range(1, n):
            continuation = {}
            for (h, w) in raw[k + 1]:
                key = (h[1:], w)                       # drop the earliest token of the history
                continuation[key] = continuation.get(key, 0) + 1
            self.tables[k] = self.summarize(continuation)

    def summarize(self, counts):
        """(counts, totals[h] = sum of counts after h, types[h] = number of distinct tokens after h)."""
        totals = {}
        types = {}
        for (h, w), c in counts.items():
            totals[h] = totals.get(h, 0) + c
            types[h] = types.get(h, 0) + 1
        return counts, totals, types

    def prob(self, history, token):
        p = 1.0 / len(self.vocab)                      # order 0: uniform
        for k in range(1, self.n + 1):                 # build up from the unigram level
            counts, totals, types = self.tables[k]
            h = last_tokens(history, k - 1)
            t = totals.get(h, 0)
            if t > 0:
                c = counts.get((h, token), 0)
                p = max(c - self.discount, 0) / t + self.discount * types[h] / t * p
        return p

    def distribution(self, history):
        dist = {}
        for w in self.vocab:
            dist[w] = self.prob(history, w)
        return dist


def log_loss(model, sequences):
    """Average negative log probability, in nats per predicted token (END included).

    Returns infinity if any event gets probability zero.
    """
    total = 0.0
    events = 0
    for seq in sequences:
        for history, token in ngram_events(seq, model.n):
            p = model.prob(history, token)
            if p <= 0.0:
                return math.inf
            total -= math.log(p)
            events += 1
    return total / events


def perplexity(model, sequences):
    """exp(log loss): the size of a uniform choice that would be as uncertain as the model."""
    return math.exp(log_loss(model, sequences))


def bits_per_token(model, sequences):
    """Log loss in bits. For a character model this is bits per character."""
    return log_loss(model, sequences) / math.log(2)


def zero_fraction(model, sequences):
    """The fraction of events to which the model assigns probability zero."""
    zeros = 0
    events = 0
    for seq in sequences:
        for history, token in ngram_events(seq, model.n):
            if model.prob(history, token) == 0.0:
                zeros += 1
            events += 1
    return zeros / events


def unseen_fraction(train_sequences, test_sequences, n):
    """The fraction of test n-grams (history, token) that never occur in the training sequences."""
    seen = count_events(train_sequences, n)
    unseen = 0
    events = 0
    for seq in test_sequences:
        for event in ngram_events(seq, n):
            if event not in seen:
                unseen += 1
            events += 1
    return unseen / events


def good_turing_unseen(sequences, n):
    """Good-Turing's estimate of the probability that the next n-gram is new: N_1 / N,
    the fraction of all n-gram occurrences that belong to n-grams seen exactly once."""
    counts = count_events(sequences, n)
    singletons = 0
    total = 0
    for c in counts.values():
        total += c
        if c == 1:
            singletons += 1
    return singletons / total


def sample(model, rng, max_tokens=60, exclude=()):
    """Generate one sequence: sample a token from the model's distribution after the current
    history, append it, and repeat until END (or max_tokens). rng is a numpy Generator."""
    history = [BOS] * (model.n - 1)
    out = []
    for _ in range(max_tokens):
        dist = model.distribution(history)
        tokens = []
        weights = []
        for w, p in dist.items():
            if w not in exclude:
                tokens.append(w)
                weights.append(p)
        total = sum(weights)
        r = rng.random() * total
        for w, p in zip(tokens, weights):
            r -= p
            if r <= 0:
                break
        if w == END:
            break
        out.append(w)
        history.append(w)
    return out
