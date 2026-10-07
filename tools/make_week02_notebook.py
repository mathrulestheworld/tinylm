"""Write notebooks/week02_language_models.ipynb from the cells below (run, then execute the notebook)."""
from pathlib import Path

import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Week 2 hands-on: language models from counts to neural networks

*Generative AI from First Principles* · [course page](https://mathrulestheworld.github.io/genai-first-principles/)

This notebook follows Lecture 2, *Language Modeling: From Counts to Neural Networks*, and its lecture notes. Each part is a stop in the talk: the slides give the idea, this notebook shows it working on real text.

| Part | What happens | Lecture notes |
|---|---|---|
| 1. Counting | A bigram model of four sentences, by hand: its table, its tree of complete sentences, why counting is maximum likelihood, and a zero; Shannon's experiment on TinyStories | §§1–2 |
| 2. Smoothing and evaluation | Sparsity, perplexity, smoothing, honest validation and test splits | §§2–4 |
| 3. Word vectors from counts | Co-occurrence counts, PMI, and an SVD | §5 |
| 4. word2vec | Skip-gram with negative sampling, trained live; the vectors in 2-D and 3-D; nearest-neighbor search; why it agrees with counting | §5 |
| 5. Neural language models | The bigram model as a network; word vectors; the model of Bengio et al. (2003): build it, train it, look inside | §6 |
| 6. Take home: recurrent networks | Character-level RNN and LSTM: why the plain RNN forgets (saturated units, a memory test), a 512-unit LSTM that beats every n-gram model, reading, sampling, temperature, a quotation cell; a word-level LSTM | §§7–8 |

**Data.** [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023): short stories in simple English. We use its validation file (about 22,000 stories, 19 MB; downloaded on first run into `data/`) and split it by story into 80% training, 10% validation, and 10% test, with a fixed seed, so the numbers match the lecture notes.

**Code.** Everything used here is in the repository: [`tinylm/ngram.py`](../tinylm/ngram.py) (count models and the evaluation harness), [`tinylm/embeddings.py`](../tinylm/embeddings.py) (count vectors and word2vec), [`tinylm/neural.py`](../tinylm/neural.py) (the neural models), and [`tinylm/data.py`](../tinylm/data.py). From this week on we use PyTorch for tensors and gradients; it does what the Week 1 engine does, faster.

**Time.** With `QUICK = True` every cell runs in under a minute on a laptop (the evaluations use 500 stories instead of all of them, so a few numbers differ slightly from the notes). Trained neural models are loaded from `checkpoints/`; each training cell says how to retrain from scratch.
""")

code(r"""
import sys, time, math, pathlib
ROOT = pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT))                 # use the repository's tinylm without installing it

import numpy as np
import torch
import matplotlib.pyplot as plt

from tinylm.data import tinystories_words, clean_chars, END, UNK
from tinylm import ngram, embeddings, neural

QUICK = False                                 # True: evaluate on 500 stories (faster; numbers differ slightly from the notes)
torch.set_num_threads(max(1, torch.get_num_threads()))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

data = tinystories_words(ROOT / "data")       # downloads TinyStories-valid.txt on the first run
train, val, test, vocab = data["train"], data["val"], data["test"], data["vocab"]
if QUICK:
    val, test = val[:500], test[:500]
print(f"{len(train):,} training, {len(val):,} validation, {len(test):,} test stories")
print(f"{sum(len(s) + 1 for s in train):,} training tokens (with one END per story); vocabulary of {len(vocab):,} words")
print("first training story:", " ".join(train[0][:40]), "...")
""")

# ============================================================================= Part 1
md(r"""
# Part 1: Counting

## 1.1 A bigram model of four sentences

A **bigram model** predicts each word from the one before it. Its maximum-likelihood estimate is a table of relative frequencies,
$$q(w \mid h) = \frac{c(h, w)}{c(h)},$$
where $c(h, w)$ counts how often $w$ follows $h$. Every sentence starts with the context BOS (written `<s>`), which is never predicted, and ends by predicting END, so the probability of a complete sentence includes the probability of stopping.

*In class:* before running the next cell, guess the word after *the*.
""")

code(r"""
corpus = [s.split() for s in ["the cat sleeps", "the cat eats fish", "the dog sleeps", "a dog eats"]]
toy_vocab = ["the", "a", "cat", "dog", "sleeps", "eats", "fish", END]
bigram = ngram.NGramModel(corpus, 2, toy_vocab)        # alpha = 0: maximum likelihood

print(f"{'history':>9s} " + " ".join(f"{w:>7s}" for w in toy_vocab))
for h in [ngram.BOS, "the", "a", "cat", "dog", "sleeps", "eats", "fish"]:
    print(f"{h:>9s} " + " ".join(f"{bigram.prob((h,), w):7.2f}" for w in toy_vocab))
""")

code(r"""
def sentence_probability(model, sentence):
    p = 1.0
    for history, word in ngram.ngram_events(sentence.split(), model.n):
        p *= model.prob(history, word)
    return p

for s in ["the cat sleeps", "the dog eats", "a cat sleeps"]:
    print(f"q({s}, END) = {sentence_probability(bigram, s):.4f}")

rng = np.random.default_rng(0)
print("\nsamples:", [" ".join(ngram.sample(bigram, rng)) for _ in range(6)])
""")

md(r"""
*the dog eats* never occurred, but each of its transitions did, so it gets probability 1/16. *a cat sleeps* is a fine sentence, but *a cat* never occurred, so it gets probability **zero**. A table of counts can recombine pieces it has seen; it can say nothing about a piece it has not.

### The model as a tree

Every complete sentence is a path from BOS to an END leaf, and its probability is the product of the branch probabilities along the path (Figure 2a of the notes). The cell below walks this model's tree of prefixes and lists every complete sentence it can generate. The sentences end at different depths, and their probabilities add up to one. *a cat sleeps* is not among them.
""")

code(r"""
def complete_sentences(model, prefix=(), p=1.0, max_words=8):
    # walk the tree of prefixes: each branch is one more token, labeled with its probability
    for word, q in model.distribution((ngram.BOS,) + prefix).items():
        if q == 0:
            continue
        if word == END:
            yield " ".join(prefix), p * q                    # an END leaf: a complete sentence
        elif len(prefix) < max_words:
            yield from complete_sentences(model, prefix + (word,), p * q, max_words)

leaves = sorted(complete_sentences(bigram), key=lambda leaf: -leaf[1])
for sentence, p in leaves:
    print(f"{p:.4f}  {sentence}")
print(f"\n{len(leaves)} complete sentences; total probability {sum(p for _, p in leaves):.4f}")
""")

md(r"""
### Counting is maximum likelihood

Why relative frequencies? They are the probabilities that make the training text most probable. In the training sentences *cat* is followed once by *sleeps* and once by *eats*, so the log-likelihood of the training text depends on $p = q(\text{sleeps}\mid\text{cat})$ through $\log p + \log(1-p)$. It is largest at $p = 1/2$, the relative frequency. (The same argument with a Lagrange multiplier for each row gives $c(h,w)/c(h)$ in general.)
""")

code(r"""
for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f"q(sleeps | cat) = {p:.1f}:  log-likelihood of the cat row = {math.log(p) + math.log(1 - p):.4f}")
""")

md(r"""
## 1.2 Shannon's experiment, repeated

In 1948 Shannon generated text from approximations to English that used more and more context. Here are character models of TinyStories that choose each character from the previous $n-1$, with probabilities equal to relative frequencies in 6,000 training stories, and then word models.
""")

code(r"""
char_train = [clean_chars(s) for s in data["stories"][0][:6000]]
char_test = [clean_chars(s) for s in data["stories"][2][:500]]
alphabet = sorted(set("".join(char_train))) + [END]
print(f"{len(alphabet) - 1} characters: {''.join(alphabet[:-1])!r}\n")

rng = np.random.default_rng(3)
for n in [1, 2, 3, 5, 7]:
    model = ngram.NGramModel(char_train, n, alphabet)
    print(f"n={n}: " + "".join(ngram.sample(model, rng, max_tokens=110)))
""")

code(r"""
rng = np.random.default_rng(11)
for n in [1, 2, 3]:
    model = ngram.NGramModel(train, n, vocab)
    print(f"n={n}: " + neural.detokenize(ngram.sample(model, rng, max_tokens=30, exclude=(UNK,))))
""")

# ============================================================================= Part 2
md(r"""
# Part 2: Smoothing and evaluation

## 2.1 Sparsity

How many n-grams of new text were never seen in training? Good–Turing's estimate of the probability that the next n-gram is new is $N_1/N$, the fraction of training occurrences that belong to n-grams seen exactly once. It can be checked against the test stories.
""")

code(r"""
for n in [2, 3, 4]:
    observed = ngram.unseen_fraction(train, test, n)
    line = f"{n}-grams: {100 * observed:5.1f}% of test n-grams never occur in training"
    if n < 4:
        line += f"   (Good-Turing predicts {100 * ngram.good_turing_unseen(train, n):.1f}%)"
    print(line)
""")

md(r"""
## 2.2 Smoothing, measured by perplexity

The evaluation harness scores a model by its average log loss on held-out stories, in nats per predicted token (END included), and reports **perplexity** $= e^{\text{loss}}$: a model that spreads probability uniformly over $K$ tokens has perplexity $K$. Lower is better.

- **Add-$\alpha$**: $q(w\mid h) = (c(h,w)+\alpha)/(c(h)+\alpha|V|)$, with $\alpha$ chosen on the validation stories.
- **Interpolated Kneser–Ney**: discount every count by $D = 0.75$ and give the freed mass to a lower-order model built from *continuation counts* (how many different words a word follows).
""")

code(r"""
start = time.time()
results = {}
for n in [1, 2, 3]:
    best_alpha, best_loss = None, math.inf
    for alpha in [1, 0.1, 0.01, 0.003, 0.001]:
        loss = ngram.log_loss(ngram.NGramModel(train, n, vocab, alpha=alpha), val)
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    results[("add-one", n)] = ngram.perplexity(ngram.NGramModel(train, n, vocab, alpha=1), test)
    results[("add-alpha", n)] = ngram.perplexity(ngram.NGramModel(train, n, vocab, alpha=best_alpha), test)
    print(f"n={n}: best alpha on validation = {best_alpha}")
kn = {}
for n in [2, 3, 4]:
    kn[n] = ngram.KneserNeyModel(train, n, vocab)
    results[("Kneser-Ney", n)] = ngram.perplexity(kn[n], test)

print(f"\n{'test perplexity':22s}" + "".join(f"{n:>9d}-gram" for n in [1, 2, 3, 4]))
for name in ["add-one", "add-alpha", "Kneser-Ney"]:
    row = "".join(f"{results[(name, n)]:14.1f}" if (name, n) in results else f"{'':14s}" for n in [1, 2, 3, 4])
    print(f"{name:22s}{row}")
print(f"\n({time.time() - start:.0f} s)")
""")

md(r"""
Add-one smoothing makes the trigram model *worse* than the bigram model: each trigram history is seen only a few times, and 8,681 pseudo-counts swamp the real ones. With the same counts, how probability is shared out decides whether longer context helps.

## 2.3 Choose on validation, report on test

The four-sentence corpus again, now with held-out sentences: validation {*a cat sleeps*, *a cat eats*} and test {*the dog eats*, *a dog sleeps*}. We choose $\alpha$ on validation, then report test once.
""")

code(r"""
toy_val = [s.split() for s in ["a cat sleeps", "a cat eats"]]
toy_test = [s.split() for s in ["the dog eats", "a dog sleeps"]]
print(f"{'alpha':>6s} {'validation loss':>16s} {'test loss':>10s}")
for alpha in [0, 0.01, 0.1, 0.5, 1]:
    m = ngram.NGramModel(corpus, 2, toy_vocab, alpha=alpha)
    print(f"{alpha:6} {ngram.log_loss(m, toy_val):16.3f} {ngram.log_loss(m, toy_test):10.3f}")
""")

md(r"""
Validation picks $\alpha = 0.1$; its test loss, 0.842 nats per token (perplexity 2.32), is the number to report. Choosing on the test set would have picked $\alpha = 0$, because every transition of the two test sentences happens to occur in training; that model gives the validation sentences probability zero.

## 2.4 How much does more context help?

Character models of increasing order $n$, smoothed with Kneser–Ney: bits per character on the training text and on held-out stories, and the fraction of held-out characters whose $n$-gram was never seen in training (to which an unsmoothed model assigns probability zero). The loss on the training text keeps falling; the held-out loss improves more and more slowly, and the gap between the two is overfitting. This cell takes about six minutes (with `QUICK = True`, two); in class, show Figure 4 of the notes.
""")

code(r"""
start = time.time()
orders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if not QUICK else [1, 3, 5, 7, 9]
held_out = char_test if not QUICK else char_test[:200]
train_part = char_train[:300]
rows = []
for n in orders:
    model = ngram.NGramModel(char_train, 1, alphabet) if n == 1 else ngram.KneserNeyModel(char_train, n, alphabet)
    rows.append((n, ngram.bits_per_token(model, train_part), ngram.bits_per_token(model, held_out),
                 ngram.unseen_fraction(char_train, held_out, n)))
    print(f"n={n}: training {rows[-1][1]:.3f}, held-out {rows[-1][2]:.3f} bits per character; "
          f"{100 * rows[-1][3]:.1f}% of held-out {n}-grams unseen   ({time.time() - start:.0f} s)")
best_char_ngram = min(r[2] for r in rows)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
a.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="training text")
a.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="held-out stories")
a.set_xlabel("order n"); a.set_ylabel("bits per character"); a.legend()
b.bar([r[0] for r in rows], [100 * r[3] for r in rows], color="#D1342F", alpha=0.6)
b.set_xlabel("order n"); b.set_ylabel("held-out n-grams never seen (%)")
plt.tight_layout(); plt.show()
""")

# ============================================================================= Part 3
md(r"""
# Part 3: Word vectors from counts

A word as an index (a one-hot vector) is equally far from every other word. We want vectors in which words used alike are near each other. The first route counts: how often does each word occur within two positions of each other word? **Pointwise mutual information** compares a co-occurrence count with what independence would predict,
$$\mathrm{PMI}(w, c) = \log \frac{p(w, c)}{p(w)\,p(c)}.$$
We keep the positive part (PPMI) and compress the rows with a truncated SVD to 100 numbers per word.
""")

code(r"""
start = time.time()
text = embeddings.content_words(train)                          # drop punctuation and <unk>
top_words, word_counts = embeddings.frequent_words(text, how_many=3000)
counts = embeddings.cooccurrence_counts(text, top_words, window=2)
P = embeddings.pmi(counts)
ix = {w: i for i, w in enumerate(top_words)}
for a_, b_ in [("ice", "cream"), ("once", "upon"), ("the", "dog"), ("happy", "sad")]:
    print(f"PMI({a_}, {b_}) = {P[ix[a_], ix[b_]]:5.2f}   ({int(counts[ix[a_], ix[b_]]):,} co-occurrences)")
count_vectors = embeddings.svd_vectors(embeddings.ppmi(counts), 100)
print(f"\n{len(top_words)} words x 100 dimensions  ({time.time() - start:.0f} s)\n")
for q in ["mom", "red", "ran", "sad", "apple", "three"]:
    print(f"{q:>6s}: " + ", ".join(w for w, s in embeddings.nearest(count_vectors, top_words, q)))
""")

# ============================================================================= Part 4
md(r"""
# Part 4: word2vec

The second route predicts. **Skip-gram with negative sampling** gives each word a vector $\mathbf v_w$ and a context vector $\mathbf u_w$, and trains them so that a logistic regression can tell a real (word, neighbor) pair from a made-up one:
$$\text{loss} = -\log\sigma(\mathbf v_w\cdot\mathbf u_c) - \sum_{i=1}^{k}\log\sigma(-\mathbf v_w\cdot\mathbf u_{c'_i}),$$
with $k = 5$ "negative" contexts $c'_i$ drawn at random (unigram frequencies to the power 3/4). The model is 20 lines in [`tinylm/embeddings.py`](../tinylm/embeddings.py). Every word starts with a random vector.

Ten passes over the training stories take about two minutes on a laptop. *In class:* set `PASSES = 3` (about a minute); the clusters already appear, while the analogies below need the full ten.
""")

code(r"""
PASSES = 10
w2v_words, _ = embeddings.frequent_words(text, min_count=5)
print(f"{len(w2v_words):,} words occur at least five times")
start = time.time()
w2v, w2v_info = embeddings.train_skipgram(text, w2v_words, word_counts, dim=100, passes=PASSES)
print(f"({time.time() - start:.0f} s)")
w2v_vectors = w2v.v.weight.detach().numpy()
for q in ["happy", "sad", "dog", "king", "apple", "park"]:
    print(f"{q:>6s}: " + ", ".join(w for w, s in embeddings.nearest(w2v_vectors, w2v_words, q)))
""")

md(r"""
## Before and after, in two dimensions

Six groups of words, projected to two dimensions with t-SNE (which keeps near neighbors near; distances between clusters mean little). Nothing in the objective names a group.
""")

code(r"""
from sklearn.manifold import TSNE

GROUPS = {"colors": ["red", "blue", "green", "yellow", "pink", "purple"],
          "animals": ["dog", "cat", "bird", "frog", "lion", "duck"],
          "family": ["mom", "dad", "sister", "brother", "grandma", "grandpa"],
          "feelings": ["happy", "sad", "scared", "angry", "excited", "worried"],
          "food": ["apple", "cake", "cookie", "banana", "soup", "bread"],
          "places": ["park", "school", "garden", "forest", "beach", "kitchen"]}
COLORS = {"colors": "#D1342F", "animals": "#E0661B", "family": "#2F5FB3",
          "feelings": "#7B4FA8", "food": "#2E9E5B", "places": "#5F6B7A"}

def project(vectors, dims=2, seed=0):
    unit = embeddings.unit_rows(vectors)
    return TSNE(n_components=dims, perplexity=6, metric="cosine", init="pca", random_state=seed).fit_transform(unit)

def plot_groups(ax, vectors, words, title):
    chosen = [(w, g) for g, ws in GROUPS.items() for w in ws]
    xy = project(np.array([vectors[words.index(w)] for w, g in chosen]))
    for (w, g), (x, y) in zip(chosen, xy):
        ax.scatter(x, y, color=COLORS[g], s=12)
        ax.annotate(w, (x, y), fontsize=8, color=COLORS[g], xytext=(0, 3), textcoords="offset points", ha="center")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
plot_groups(axes[0], w2v_info["before"], w2v_words, "random vectors, before training")
plot_groups(axes[1], w2v_vectors, w2v_words, f"after {PASSES} passes of skip-gram")
plt.tight_layout(); plt.show()
""")

md(r"""
## In three dimensions

The same idea with more words, projected to three dimensions. Drag to rotate (in Jupyter or VS Code; the static copy on GitHub does not show interactive plots). Needs `pip install plotly`.
""")

code(r"""
try:
    import plotly.graph_objects as go
    MORE = {"colors": ["orange", "white", "black", "brown", "gray", "gold"],
            "animals": ["bunny", "bear", "fish", "puppy", "kitten", "horse", "cow", "pig", "monkey"],
            "family": ["mommy", "daddy", "mother", "father", "baby", "aunt", "uncle"],
            "feelings": ["proud", "upset", "lonely", "surprised", "curious", "nervous"],
            "food": ["pie", "candy", "milk", "juice", "pizza", "carrot", "sandwich"],
            "places": ["house", "store", "beach", "zoo", "farm", "river", "castle"]}
    chosen = [(w, g) for g in GROUPS for w in GROUPS[g] + MORE[g] if w in w2v_words]
    chosen = list(dict.fromkeys(chosen))
    xyz = project(np.array([w2v_vectors[w2v_words.index(w)] for w, g in chosen]), dims=3)
    fig3d = go.Figure()
    for g in GROUPS:
        pts = [(p, w) for p, (w, gg) in zip(xyz, chosen) if gg == g]
        fig3d.add_trace(go.Scatter3d(x=[p[0] for p, w in pts], y=[p[1] for p, w in pts], z=[p[2] for p, w in pts],
                                     mode="markers+text", text=[w for p, w in pts], name=g,
                                     marker=dict(size=4, color=COLORS[g]), textfont=dict(size=10, color=COLORS[g])))
    fig3d.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    fig3d.show()
except ImportError:
    print("plotly is not installed: pip install plotly")
""")

md(r"""
## Analogies

If relations are consistent directions, then $\mathbf v_{\text{girl}} - \mathbf v_{\text{boy}} + \mathbf v_{\text{he}}$ should land near $\mathbf v_{\text{she}}$. On a corpus this small some analogies work and some do not.
""")

code(r"""
for a_, b_, c_ in [("boy", "girl", "he"), ("he", "she", "his"), ("mom", "dad", "mommy"), ("run", "ran", "walk"),
                   ("boy", "girl", "brother"), ("king", "queen", "prince"), ("one", "two", "three")]:
    print(f"{a_} : {b_} :: {c_} : ?   " + ", ".join(f"{w} ({s})" for w, s in embeddings.analogy(w2v_vectors, w2v_words, a_, b_, c_)))
""")

md(r"""
## A tiny vector database

Search engines, recommendation systems, and assistants that look up documents all store items as vectors and ask one question: *which stored vectors are nearest to this one?* Exact search compares the query with every stored vector, $O(Nd)$ work per query. Fine for our 6,000 words; slow for a billion items, which is why vector databases use approximate methods (hashing, quantization, graphs of neighbors).
""")

code(r"""
E = embeddings.unit_rows(w2v_vectors)
query = E[w2v_words.index("cake")]
start = time.time()
best = embeddings.top_k(E, query, 6)
print(f"nearest to 'cake' among {len(E):,} vectors: {[w2v_words[i] for i in best]}  ({1000 * (time.time() - start):.1f} ms)")

big = np.random.default_rng(0).standard_normal((1_000_000, 100)).astype(np.float32)
start = time.time()
embeddings.top_k(big, big[0], 5)
print(f"one exact query over 1,000,000 random vectors: {1000 * (time.time() - start):.0f} ms")
del big
""")

md(r"""
## Counting and prediction meet

Levy and Goldberg (2014): if every pair could be fitted freely, skip-gram's optimum is
$$\mathbf v_w\cdot\mathbf u_c = \log\frac{n(w,c)}{k\,n(w)\,p_n(c)} \;\approx\; \mathrm{PMI}(w,c) - \log k,$$
a shifted PMI, computed from the pair counts. Check it on the pairs of the last training pass.
""")

code(r"""
predicted, learned = embeddings.shifted_pmi_check(w2v, w2v_info, min_count=20)
r = np.corrcoef(predicted, learned)[0, 1]
plt.figure(figsize=(4.6, 4))
plt.scatter(predicted, learned, s=3, alpha=0.25)
lo, hi = predicted.min(), predicted.max()
plt.plot([lo, hi], [lo, hi], "r--", lw=1)
plt.xlabel("from counts: log n(w,c) / (k n(w) p_n(c))"); plt.ylabel("learned: v_w . u_c")
plt.title(f"{len(predicted):,} pairs seen at least 20 times; correlation {r:.2f}", fontsize=9)
plt.show()
""")

# ============================================================================= Part 5
md(r"""
# Part 5: Neural language models

Three steps, as in the lecture notes (§6) and Karpathy's [makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) lectures: the bigram model computed by a network, the same network with its matrix factored through word vectors, and the fixed-window model of Bengio et al. (2003).

## 5.1 The bigram model as a network

A one-hot vector times a weight matrix $W$ picks out a row of $W$; the softmax turns the row into probabilities. Trained by gradient descent on the log loss, the rows must converge to the relative frequencies of §2: maximum likelihood is the same whether computed by counting or by descent. Characters keep the matrix small ($34 \times 34$).
""")

code(r"""
char_alpha = neural.char_alphabet(char_train)
end_id = char_alpha.index(neural.STORY_END)
pairs_train = torch.cat([torch.tensor([end_id]), neural.encode_chars(char_train, char_alpha)])
pairs_test = torch.cat([torch.tensor([end_id]), neural.encode_chars(char_test, char_alpha)])
a_tr, b_tr, a_te, b_te = pairs_train[:-1], pairs_train[1:], pairs_test[:-1], pairs_test[1:]
K = len(char_alpha)

# counting
N = torch.zeros(K, K)
N.index_put_((a_tr, b_tr), torch.ones(len(a_tr)), accumulate=True)
P_count = (N + 0.01) / (N.sum(1, keepdim=True) + 0.01 * K)        # a little add-alpha for the 4 unseen held-out pairs
bits = lambda P, a, b: float(-torch.log2(P[a, b]).mean())

# the network, trained by minibatch gradient descent
torch.manual_seed(0)
net = neural.BigramNet(K, K)
opt = torch.optim.Adam(net.parameters(), lr=0.1)
start = time.time()
for step in range(1, 3001):
    i = torch.randint(0, len(a_tr), (4096,))
    loss = torch.nn.functional.cross_entropy(net(a_tr[i]), b_tr[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 2100:
        for g in opt.param_groups: g["lr"] = 0.01
P_net = torch.softmax(net.W.weight.detach(), 1)
print(f"counting:    training {bits(P_count, a_tr, b_tr):.3f}   held-out {bits(P_count, a_te, b_te):.3f} bits per character")
print(f"the network: training {bits(P_net, a_tr, b_tr):.3f}   held-out {bits(P_net, a_te, b_te):.3f} bits per character"
      f"   ({time.time() - start:.0f} s; uniform would be {math.log2(K):.2f})")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, P, title in [(axes[0], P_count, "counting"), (axes[1], P_net, "the network")]:
    ax.imshow(P ** 0.5, cmap="Blues")
    ax.set_xticks(range(K)); ax.set_xticklabels(["␣" if c == " " else c for c in char_alpha], fontsize=6)
    ax.set_yticks(range(K)); ax.set_yticklabels(["␣" if c == " " else c for c in char_alpha], fontsize=6)
    ax.set_title(title); ax.set_xlabel("next character"); ax.set_ylabel("previous character")
plt.tight_layout(); plt.show()
""")

md(r"""
## 5.2 From a table to word vectors

For words the same matrix would be $8{,}682 \times 8{,}681$, 75 million numbers, each row learned only from its own word. Factor it: $q(\cdot\mid v) = \operatorname{softmax}(\mathbf b + U\,C(v))$ with a 64-dimensional vector $C(v)$ per word. Similar words must now share. The notes train it for two passes (about 14,000 steps, 13 minutes on a laptop) to a test perplexity of 49.9; here a shorter run.
""")

code(r"""
BIGRAM_STEPS = 3000 if QUICK else 4000                  # 14,000 for the notes' two passes
index = neural.word_index(vocab)
X1, Y1 = neural.make_windows(train, index, context=1)
X1_val, Y1_val = neural.make_windows(val, index, context=1)
torch.manual_seed(0)
fb = neural.BigramNet(len(vocab) + 1, len(vocab), embed_dim=64)
print(f"{sum(p.numel() for p in fb.parameters()):,} parameters, against {(len(vocab) + 1) * len(vocab):,} for the full matrix")
start = time.time()
neural.train_window_lm(fb, X1, Y1, X1_val[:50000], Y1_val[:50000], epochs=2, eval_every=1000, max_steps=BIGRAM_STEPS)
print(f"({time.time() - start:.0f} s)   Kneser-Ney bigram: {results[('Kneser-Ney', 2)]:.1f}")
""")

md(r"""
## 5.3 The fixed-window model

Bengio, Ducharme, Vincent, and Jauvin (2003) predicted the next word from the previous four through learned word vectors:
$$q(\cdot \mid x_{t-4:t-1}) = \operatorname{softmax}\big(\mathbf b + U \tanh(\mathbf d + H[C(x_{t-4}); \dots; C(x_{t-1})])\big).$$
$C$ is a table of word vectors, learned together with everything else by minimizing the cross-entropy of the next word. Here is the whole model:
""")

code(r"""
import inspect
print(inspect.getsource(neural.FixedWindowLM))

X_train, Y_train = neural.make_windows(train, index, context=4)
X_val, Y_val = neural.make_windows(val, index, context=4)
X_test, Y_test = neural.make_windows(test, index, context=4)
print(f"{len(X_train):,} training examples: (four previous words, next word)")
""")

md(r"""
## Watch it learn

The training loop: take a random minibatch of 512 examples, compute the cross-entropy, backpropagate, take an Adam step. A few hundred steps already beat the add-one models of Part 2. Training fully (two passes, about ten minutes on a laptop) gives the checkpoint loaded below; set `TRAIN_FULL = True` to retrain it yourself.
""")

code(r"""
TRAIN_FULL = False
torch.manual_seed(0)
fresh = neural.FixedWindowLM(len(vocab) + 1, len(vocab), context=4, embed_dim=64, hidden=128)
print(f"{sum(p.numel() for p in fresh.parameters()):,} parameters")
start = time.time()
if TRAIN_FULL:
    neural.train_window_lm(fresh, X_train, Y_train, X_val, Y_val, epochs=2)
    torch.save(fresh.state_dict(), ROOT / "checkpoints" / "week02_fixed_window.pt")
else:
    neural.train_window_lm(fresh, X_train, Y_train, X_val[:50000], Y_val[:50000], epochs=1, eval_every=100, max_steps=500)
print(f"({time.time() - start:.0f} s)")

lm = neural.FixedWindowLM(len(vocab) + 1, len(vocab), context=4, embed_dim=64, hidden=128)
lm.load_state_dict(torch.load(ROOT / "checkpoints" / "week02_fixed_window.pt"))
print(f"\ntrained model (checkpoint): test perplexity {math.exp(neural.average_loss(lm, X_test, Y_test)):.1f}"
      f"   (Kneser-Ney 4-gram: {results[('Kneser-Ney', 4)]:.1f})")
""")

md(r"""
## Neural and count models make different errors

Average the two models' probabilities, $\lambda\,q_{\text{neural}} + (1-\lambda)\,q_{\text{KN}}$, with $\lambda$ chosen on validation. Bengio and colleagues found the same: the mixture beats both.
""")

code(r"""
def token_probabilities(model, X, Y):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out.append(torch.softmax(model(X[i:i + 8192]), -1).gather(1, Y[i:i + 8192, None])[:, 0])
    return torch.cat(out).numpy()

def kn_probabilities(model, stories):
    return np.array([model.prob(h, w) for s in stories for h, w in ngram.ngram_events(s, model.n)])

start = time.time()
p_nn_val, p_kn_val = token_probabilities(lm, X_val, Y_val), kn_probabilities(kn[4], val)
best = min((-np.log(l * p_nn_val + (1 - l) * p_kn_val).mean(), l) for l in np.linspace(0.05, 0.95, 19))
lam = best[1]
p_nn_test, p_kn_test = token_probabilities(lm, X_test, Y_test), kn_probabilities(kn[4], test)
print(f"test perplexity: neural {np.exp(-np.log(p_nn_test).mean()):.1f}, Kneser-Ney {np.exp(-np.log(p_kn_test).mean()):.1f}, "
      f"mixture with weight {lam:.2f} on the neural model {np.exp(-np.log(lam * p_nn_test + (1 - lam) * p_kn_test).mean()):.1f}"
      f"   ({time.time() - start:.0f} s)")
""")

md(r"""
## Look inside

Its next-word distributions, its samples (each probability raised to the power $1/T$ with temperature $T = 0.8$, which sharpens the distribution), and its word table $C$, which was never asked about meaning.
""")

code(r"""
for prompt in ["she was very", "once upon a", "the dog wagged its", "they went to the"]:
    top = neural.next_word_distribution(lm, index, vocab, prompt.split())
    print(f"{prompt + ' ...':22s} " + ", ".join(f"{w} {p:.2f}" for w, p in top))
print()
for seed in [1, 2, 3]:
    print(neural.detokenize(neural.sample_words(lm, index, vocab, temperature=0.8, seed=seed)), "\n")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
plot_groups(axes[0], w2v_vectors, w2v_words, "skip-gram vectors")
plot_groups(axes[1], lm.C.weight.detach().numpy()[:len(vocab)], vocab, "the language model's word table C")
plt.tight_layout(); plt.show()
""")

# ============================================================================= Part 6
md(r"""
# Part 6 (take home): recurrent language models

This part follows §§7–8 of the notes and Karpathy's essay [*The Unreasonable Effectiveness of Recurrent Neural Networks*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). A fixed window forgets everything more than four words back. A recurrent network keeps a state that it updates once per token, $\mathbf h_t = \tanh(W\mathbf h_{t-1} + V\mathbf e_t + \mathbf d)$, with the same weights at every step; the LSTM adds a cell updated *additively*, $\mathbf c_t = \mathbf f_t\odot\mathbf c_{t-1} + \mathbf i_t\odot\mathbf g_t$, so gradients can reach far back. The models are character-level: they read and write one character at a time. They are defined in [`tinylm/neural.py`](../tinylm/neural.py) (`CharRNN`, `train_char_rnn`).

Both were trained for 3,000 steps of 64 chunks of 128 characters (about 25 million characters; the LSTM takes about ten minutes on a laptop) by `tools/train_week02_checkpoints.py`. The cell below loads them; the one after it trains a fresh model for a minute so you can watch.
""")

code(r"""
train_stream = neural.encode_chars(char_train, char_alpha)
test_stream = neural.encode_chars(char_test, char_alpha)
models = {}
for kind in ["rnn", "lstm"]:
    m = neural.CharRNN(len(char_alpha), kind=kind, hidden=256)
    m.load_state_dict(torch.load(ROOT / "checkpoints" / f"week02_char_{kind}.pt"))
    models[kind] = m
    print(f"{kind:4s}: {sum(p.numel() for p in m.parameters()):,} parameters, "
          f"held-out {neural.char_bits_per_char(m, test_stream):.3f} bits per character")
print(f"best character n-gram (Part 2.4): {best_char_ngram:.3f} bits per character")
print("\nLSTM sample:", neural.sample_chars(models["lstm"], char_alpha, temperature=0.8))
""")

md(r"""
A wider LSTM, trained longer, goes past every n-gram model: `tools/train_week02_checkpoints.py lstm 6000 512` trains 512 units for 6,000 steps on the same 6,000 stories (about an hour on two CPU cores). If its checkpoint is present, the cell below scores it and samples from it.
""")

code(r"""
path = ROOT / "checkpoints" / "week02_char_lstm512.pt"
if path.exists():
    big = neural.CharRNN(len(char_alpha), kind="lstm", hidden=512)
    big.load_state_dict(torch.load(path))
    print(f"LSTM, 512 units: {sum(p.numel() for p in big.parameters()):,} parameters, "
          f"held-out {neural.char_bits_per_char(big, test_stream):.3f} bits per character\n")
    for seed in [1, 2, 3]:
        print(neural.sample_chars(big, char_alpha, max_chars=400, temperature=0.8, seed=seed), "\n")
else:
    print("no checkpoint: run tools/train_week02_checkpoints.py lstm 6000 512")
""")

code(r"""
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for kind in ["rnn", "lstm"]:
    g = neural.gradient_by_distance(models[kind], train_stream).detach().numpy()
    ax.semilogy(g[:150] / g[0], label=kind)
ax.set_xlabel("steps back from the prediction"); ax.set_ylabel("gradient norm (relative)")
ax.set_title("how far back the gradient reaches", fontsize=10); ax.legend(); plt.show()
""")

md(r"""
## Why the plain RNN forgets

Each step multiplies the gradient by the weight matrix $W$ and by the slopes $1-h^2$ of the tanh units (§7 of the notes). A unit near $\pm1$ is *saturated*: its slope is nearly 0, and almost no gradient passes through it. Karpathy shows this in [*Building makemore Part 3*](https://youtu.be/P6sfmUTpUmc) with a picture of the activations, white where a unit is saturated. Here is the picture for our plain RNN reading 120 characters of a held-out story: one row per unit, one column per character.
""")

code(r"""
rnn = models["rnn"]
with torch.no_grad():
    h, _ = rnn.rnn(rnn.emb(test_stream[None, 2000:2120]))
h = h[0].numpy()                                     # (120 characters, 256 units)
print(f"saturated (|h| > 0.99): {(np.abs(h) > 0.99).mean():.0%} of the activations; "
      f"average slope 1 - h^2: {(1 - h ** 2).mean():.2f}")
print(f"largest singular value of W: {torch.linalg.matrix_norm(rnn.rnn.weight_hh_l0, 2):.1f}  (W is not small)")
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.imshow(np.abs(h.T) > 0.99, cmap="gray", interpolation="nearest", aspect="auto")
ax.set_xlabel("character"); ax.set_ylabel("hidden unit"); ax.set_title("white: saturated", fontsize=10)
plt.show()
""")

md(r"""
### A memory test

Can a network learn to use a token far back? Each sequence is a symbol from eight, then $T$ random noise symbols from eight others, then a query; at the query the network must output the first symbol. Chance is 1/8. The cell below trains a plain RNN and an LSTM from scratch for each gap $T$, with the same budget (each run takes from a few seconds to half a minute). The LSTM's forget-gate biases start at 3, so that a new network keeps 95% of its cell at each step; set `forget_bias=0.0` (PyTorch's default) and watch the LSTM fail too.
""")

code(r"""
class MemoryNet(torch.nn.Module):
    def __init__(self, kind, hidden=64, forget_bias=3.0):
        super().__init__()
        self.emb = torch.nn.Embedding(17, 32)
        self.rnn = (torch.nn.LSTM if kind == "lstm" else torch.nn.RNN)(32, hidden, batch_first=True)
        if kind == "lstm":                              # PyTorch's gate order: input, forget, candidate, output
            with torch.no_grad():
                self.rnn.bias_ih_l0[hidden:2 * hidden] = forget_bias
                self.rnn.bias_hh_l0[hidden:2 * hidden] = 0.0
        self.out = torch.nn.Linear(hidden, 8)

    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0][:, -1])

def memory_batch(T, n, g):
    first = torch.randint(0, 8, (n, 1), generator=g)
    noise = torch.randint(8, 16, (n, T), generator=g)
    return torch.cat([first, noise, torch.full((n, 1), 16)], 1), first[:, 0]

def memory_test(kind, T, steps=1500, **kw):
    torch.manual_seed(0); g = torch.Generator().manual_seed(T)
    net = MemoryNet(kind, **kw); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = memory_batch(T, 128, g)
        loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    x, y = memory_batch(T, 4000, torch.Generator().manual_seed(999))
    with torch.no_grad():
        return (net(x).argmax(-1) == y).float().mean().item()

for T in [5, 10, 20, 40]:
    print(f"T = {T:3d}:  plain RNN {memory_test('rnn', T):.2f}   LSTM {memory_test('lstm', T):.2f}")
""")

md(r"""
## Reading a phrase

At each step the network reads one character and outputs a distribution over the next. Here are the plain RNN's four most probable next characters along a phrase, with the probability it gave the character that actually came next (after Karpathy's "hello" example).
""")

code(r"""
def read_phrase(model, phrase):
    ix = {c: i for i, c in enumerate(char_alpha)}
    x = torch.tensor([[ix[neural.STORY_END]] + [ix[c] for c in phrase]])
    with torch.no_grad():
        P = torch.softmax(model(x)[0][0], -1)
    for t, nxt in enumerate(phrase):
        top = torch.topk(P[t], 4)
        shown = "  ".join(f"{char_alpha[j]!r} {v:.2f}" for v, j in zip(top.values.tolist(), top.indices.tolist()))
        read = "start" if t == 0 else repr(phrase[t - 1])
        print(f"read {read:7s} next {nxt!r}: p = {P[t, ix[nxt]]:.2f}   top four: {shown}")

read_phrase(models["rnn"], "the dog barked")
""")

md(r"""
## Temperature

Dividing the scores by a temperature $T$ before the softmax raises each probability to the power $1/T$: $T < 1$ sharpens, $T > 1$ flattens.
""")

code(r"""
for T in [0.5, 1.0, 1.5]:
    print(f"T = {T}:", neural.sample_chars(models["lstm"], char_alpha, max_chars=300, temperature=T, seed=1), "\n")
""")

md(r"""
## A quotation cell

Karpathy, Johnson, and Fei-Fei (2015) found LSTM cells that switch on inside quotations. The cell $\mathbf c_t$ of our LSTM is a vector of 256 numbers, recomputed after every character; each coordinate is a *cell unit*. The cell below runs the LSTM over 50 held-out stories and records all 256 cell units after each character. It marks each character 1 if it is inside a quotation (from an opening quotation mark up to its closing mark) and 0 otherwise, and computes, for each unit, the correlation between the unit's value and the mark: near 1 for a unit that is high exactly inside quotations, near 0 for a unit unrelated to them. It then colors an excerpt by the best unit (red positive, blue negative).
""")

code(r"""
from IPython.display import HTML
lstm = models["lstm"]
text = "".join(t + neural.STORY_END for t in char_test[:50])
x = neural.encode_chars(char_test[:50], char_alpha)[None]
cells = []
with torch.no_grad():
    state = None
    for t in range(x.shape[1]):                       # one step at a time, to read the cell c_t
        _, state = lstm.rnn(lstm.emb(x[:, t:t + 1]), state)
        cells.append(state[1][0, 0].clone())
cells = torch.stack(cells).numpy()
inside, flag = np.zeros(len(text)), 0
for i, ch in enumerate(text):
    flag = 0 if ch == neural.STORY_END else (1 - flag if ch == '"' else flag)
    inside[i] = flag
corr = np.array([np.corrcoef(cells[:, u], inside)[0, 1] for u in range(cells.shape[1])])
u = int(np.nanargmax(np.abs(corr)))
print(f"cell unit {u}: correlation {corr[u]:.2f} with being inside a quotation; "
      f"next best {np.sort(np.abs(corr))[-2]:.2f}; {inside.mean():.0%} of characters are inside")
start = text.index('"', 2000) - 120
vals = cells[start:start + 400, u] / np.abs(cells[start:start + 400, u]).max()
spans = "".join(f'<span style="background: rgba({255 if v > 0 else 60},{90 if v > 0 else 110},{90 if v > 0 else 255},{abs(v):.2f})">{c}</span>'
                for c, v in zip(text[start:start + 400], vals))
HTML(f'<div style="font-family: monospace; line-height: 1.6; max-width: 46em">{spans}</div>')
""")

md(r"""
## Watching it learn

Train a fresh, smaller LSTM for a minute and sample from it as it goes, as Karpathy's essay does: random characters, then spaces and common letters, then words.
""")

code(r"""
torch.manual_seed(0)
student = neural.CharRNN(len(char_alpha), kind="lstm", hidden=128)
print("untrained:", neural.sample_chars(student, char_alpha, max_chars=120, seed=3))
for rounds in range(4):
    neural.train_char_rnn(student, train_stream, test_stream[:20000], steps=50, eval_every=50)
    print(f"after {50 * (rounds + 1)} steps:", neural.sample_chars(student, char_alpha, max_chars=120, seed=3), "\n")
""")

md(r"""
## Words instead of characters

The same LSTM over words, sized like the fixed-window model (64-dimensional word vectors, 128 units, about 1.8 million parameters), scored story by story on the same prediction events as the n-gram models. `tools/train_week02_checkpoints.py word` trains it (about 20 minutes on a laptop).
""")

code(r"""
wl = neural.WordLSTM(len(vocab) + 1, len(vocab), embed_dim=64, hidden=128)
wl.load_state_dict(torch.load(ROOT / "checkpoints" / "week02_word_lstm.pt"))
sx, sy = neural.story_tensors(test, index)
print(f"{sum(p.numel() for p in wl.parameters()):,} parameters")
print(f"test perplexity: word LSTM {neural.story_perplexity(wl, sx, sy):.1f}, fixed window "
      f"{math.exp(neural.average_loss(lm, X_test, Y_test)):.1f}, Kneser-Ney 4-gram {results[('Kneser-Ney', 4)]:.1f}")
""")

md(r"""
## Things to try

1. **No clipping.** Train the plain RNN with `clip=1e9`. What happens to the loss early in training?
2. **Longer chunks.** Train with `length=32` and `length=256`. Truncated backpropagation through time sees at most `length` steps back; does the held-out loss change?
3. **Width.** Compare `hidden=64, 128, 256, 512` for the LSTM at a fixed number of steps. Plot held-out bits per character against the number of parameters.
4. **Temperature.** Sample from the LSTM at temperatures 0.5, 1.0, and 1.5. Which looks most like a story, and which has the lowest loss?
5. **A bigger word LSTM.** Train `WordLSTM` with 256 or 512 units with `neural.train_word_rnn`. How does its test perplexity move against its number of parameters, and against the Kneser–Ney 4-gram?
6. **How far can the LSTM remember?** Run `memory_test('lstm', T)` for T = 80, 160, 320 with `forget_bias` 1, 3, and 5, and with more steps. Which helps more?
7. **Your own embeddings.** Train skip-gram with `dim=20` and `dim=300`, and with windows of 1 and 5. Which gives better neighbors? Which gives better analogies?
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent.parent / "notebooks" / "week02_language_models.ipynb"
nbf.write(nb, out)
print("wrote", out)
