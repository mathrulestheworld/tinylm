"""Word embeddings (Week 2): vectors from counts (PMI and SVD) and from prediction (word2vec).

Both routes start from which words occur near which. Counting builds the word-context
co-occurrence matrix, reweights it by pointwise mutual information, and compresses it with a
singular value decomposition. Prediction (skip-gram with negative sampling, Mikolov et al.,
2013) trains vectors so that a logistic regression can tell real (word, neighbor) pairs from
random ones. Levy and Goldberg (2014) showed that the second implicitly factorizes a shifted
version of the matrix the first factorizes explicitly.
"""
import numpy as np
import torch
import torch.nn as nn

PUNCTUATION = set('.,!?;:"')


def content_words(sequences):
    """The sequences with punctuation and <unk> removed: the text the embeddings are built from."""
    result = []
    for seq in sequences:
        kept = []
        for w in seq:
            if w not in PUNCTUATION and w != "<unk>":
                kept.append(w)
        result.append(kept)
    return result


def frequent_words(sequences, how_many=None, min_count=1):
    """Words sorted by frequency (most frequent first), with their counts."""
    counts = {}
    for seq in sequences:
        for w in seq:
            counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=count_then_word)
    words = []
    for w, c in ranked:
        if c >= min_count:
            words.append(w)
    if how_many is not None:
        words = words[:how_many]
    return words, counts


def count_then_word(item):
    """Sort key: higher count first, then alphabetical."""
    word, count = item
    return (-count, word)


# ----------------------------------------------------------------------------- vectors from counts

def cooccurrence_counts(sequences, words, window=2):
    """n[i, j] = how often words[j] occurs within `window` positions of words[i]."""
    index = {}
    for i, w in enumerate(words):
        index[w] = i
    n = np.zeros((len(words), len(words)))
    for seq in sequences:
        for a, w in enumerate(seq):
            if w not in index:
                continue
            for b in range(max(0, a - window), min(len(seq), a + window + 1)):
                if b != a and seq[b] in index:
                    n[index[w], index[seq[b]]] += 1
    return n


def pmi(counts):
    """PMI(w, c) = log p(w, c) / (p(w) p(c)), with probabilities estimated from the counts.
    Pairs never seen together get -inf."""
    total = counts.sum()
    p_w = counts.sum(axis=1) / total
    p_c = counts.sum(axis=0) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log((counts / total) / np.outer(p_w, p_c))


def ppmi(counts):
    """Positive PMI: negative and undefined entries set to zero."""
    m = pmi(counts)
    m[~np.isfinite(m)] = 0.0
    return np.maximum(m, 0.0)


def svd_vectors(matrix, dim):
    """Dense vectors from a truncated SVD, M ~ U_d S_d V_d^T: row i of U_d S_d^(1/2)."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :dim] * np.sqrt(S[:dim])


# ----------------------------------------------------------------------------- similarity

def unit_rows(vectors):
    """Each row divided by its length, so that dot products are cosine similarities."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def nearest(vectors, words, query, k=5):
    """The k words whose vectors have the highest cosine similarity to the query word's."""
    E = unit_rows(vectors)
    i = words.index(query)
    sims = E @ E[i]
    order = np.argsort(-sims)
    result = []
    for j in order:
        if j != i:
            result.append((words[j], round(float(sims[j]), 2)))
        if len(result) == k:
            break
    return result


def analogy(vectors, words, a, b, c, k=3):
    """a is to b as c is to ?: the words nearest to v_b - v_a + v_c (excluding a, b, c)."""
    E = unit_rows(vectors)
    target = E[words.index(b)] - E[words.index(a)] + E[words.index(c)]
    sims = E @ (target / np.linalg.norm(target))
    order = np.argsort(-sims)
    result = []
    for j in order:
        if words[j] not in (a, b, c):
            result.append((words[j], round(float(sims[j]), 2)))
        if len(result) == k:
            break
    return result


def top_k(matrix, query, k=5):
    """Exact nearest-neighbor search: the indices of the k rows of `matrix` with the largest dot
    product with `query`. It reads every row, so it costs O(N d) per query; vector databases
    answer the same question approximately in far less time."""
    scores = matrix @ query
    best = np.argpartition(-scores, k)[:k]
    return best[np.argsort(-scores[best])]


# ----------------------------------------------------------------------------- word2vec

class SkipGram(nn.Module):
    """Skip-gram with negative sampling.

    Every word w has a vector v_w (the embedding we keep) and a context vector u_w. For an
    observed pair (w, c) and k negative contexts c'_1..c'_k drawn from a noise distribution,
    the loss is
        -log sigmoid(v_w . u_c) - sum_i log sigmoid(-v_w . u_{c'_i}),
    one logistic regression per pair: is this pair real or made up?
    """

    def __init__(self, n_words, dim):
        super().__init__()
        self.v = nn.Embedding(n_words, dim)
        self.u = nn.Embedding(n_words, dim)
        nn.init.uniform_(self.v.weight, -0.5 / dim, 0.5 / dim)    # small random word vectors
        nn.init.zeros_(self.u.weight)                            # context vectors start at zero

    def loss(self, centers, contexts, negatives):
        """centers, contexts: (batch,) word ids; negatives: (batch, k) word ids."""
        v = self.v(centers)                                                  # (batch, dim)
        positive = nn.functional.logsigmoid((v * self.u(contexts)).sum(-1))  # log sigmoid(v.u_c)
        scores = (self.u(negatives) @ v[:, :, None]).squeeze(-1)             # (batch, k): v.u_c'
        negative = nn.functional.logsigmoid(-scores).sum(-1)
        return -(positive + negative).mean()


def skipgram_pairs(sequences, index, keep_prob, rng, max_window=2):
    """(word, context) id pairs for one pass over the text.

    Frequent words are first dropped at random (word w is kept with probability keep_prob[w]),
    then each remaining word is paired with its neighbors within a window whose size is drawn
    from 1..max_window, so that nearer words are paired more often.
    """
    pairs = []
    for seq in sequences:
        ids = []
        for w in seq:
            if w in index:
                i = index[w]
                if rng.random() < keep_prob[i]:
                    ids.append(i)
        for a, i in enumerate(ids):
            width = rng.integers(1, max_window + 1)
            for b in range(max(0, a - width), min(len(ids), a + width + 1)):
                if b != a:
                    pairs.append((i, ids[b]))
    return torch.tensor(pairs)


def train_skipgram(sequences, words, counts, dim=100, negatives=5, passes=3, lr=0.003,
                   batch_size=8192, subsample=1e-4, seed=0, log=print):
    """Train skip-gram with negative sampling and return (model, info).

    words: the vocabulary (each with a count); counts: word -> count. Noise distribution:
    unigram frequencies raised to the power 3/4. The learning rate decreases linearly over the
    passes. info holds the vectors before training and the pairs of the last pass.
    """
    index = {}
    for i, w in enumerate(words):
        index[w] = i
    freq = np.array([counts[w] for w in words], dtype=float)
    p = freq / freq.sum()
    keep_prob = np.minimum(1.0, np.sqrt(subsample / p))
    noise = torch.tensor(freq ** 0.75 / np.sum(freq ** 0.75), dtype=torch.float)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = SkipGram(len(words), dim)
    before = model.v.weight.detach().clone().numpy()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(passes):
        for group in opt.param_groups:
            group["lr"] = lr * (1 - epoch / passes) + 1e-4
        pairs = skipgram_pairs(sequences, index, keep_prob, rng)
        pairs = pairs[torch.randperm(len(pairs))]
        total = 0.0
        batches = 0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            neg = torch.multinomial(noise, len(batch) * negatives, replacement=True).view(len(batch), negatives)
            loss = model.loss(batch[:, 0], batch[:, 1], neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        log(f"pass {epoch + 1}: {len(pairs):,} pairs, average loss {total / batches:.3f}")
    info = {"before": before, "pairs": pairs, "noise": noise.numpy(), "negatives": negatives}
    return model, info


def shifted_pmi_check(model, info, min_count=20):
    """Compare the learned v_w . u_c with what counting predicts for the optimum,
    log n(w, c) - log(k n(w) p_n(c)), on the pairs of the last pass seen at least min_count times.
    Returns (predicted, learned) as two arrays."""
    pairs = info["pairs"].numpy()
    pair_counts = {}
    word_counts = {}
    for w, c in pairs:
        pair_counts[(w, c)] = pair_counts.get((w, c), 0) + 1
        word_counts[w] = word_counts.get(w, 0) + 1
    v = model.v.weight.detach().numpy()
    u = model.u.weight.detach().numpy()
    k = info["negatives"]
    noise = info["noise"]
    predicted = []
    learned = []
    for (w, c), n in pair_counts.items():
        if n >= min_count:
            predicted.append(np.log(n) - np.log(k * word_counts[w] * noise[c]))
            learned.append(float(v[w] @ u[c]))
    return np.array(predicted), np.array(learned)
