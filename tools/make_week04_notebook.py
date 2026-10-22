"""Write the Week 4 notebooks: a Transformer language model, assembled from its parts and checked.

    python3 tools/make_week04_notebook.py

notebooks/week04_transformer.ipynb has an empty attention() and an empty decoder() for students to write.
With the instructor's answers file (tools/week04_answers.py, not in the public repository) the script also
writes notebooks/week04_transformer_solution.ipynb, the same notebook with the answers filled in; it is
saved with its outputs after running
    jupyter nbconvert --to notebook --execute --inplace notebooks/week04_transformer_solution.ipynb

The notebook follows §§3 and 8 of the Lecture 4 notes. The trained word model of §8 and the model with a
wrong mask are loaded from checkpoints/; one short training run (200 steps, a few minutes) is live.
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import week04_answers as ANSWERS        # the instructor's answers; not in the public repository
except ImportError:
    ANSWERS = None

NOTES = "https://mathrulestheworld.github.io/genai-first-principles/materials/week-04/lecture-04-notes.pdf"


def build(solution):
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md(r"""
# Week 4: a Transformer language model, assembled and checked

*Generative AI from First Principles*, Lecture 4 · [course page](https://mathrulestheworld.github.io/genai-first-principles/) · [notes](%s)

The word model of the Lecture 4 notes (§8) is a decoder-only Transformer: four blocks of masked self-attention and an MLP, rotary positions, and one table of word vectors used at the input and, in reverse, at the output. It was trained on the children's stories of Week 2 and reaches a test perplexity of 9.9, against 23.1 for Week 2's word LSTM. Its parts are prepared in `tinylm/transformer.py`. You write the two pieces that make them a Transformer, load the trained weights into what you wrote, and check it the way §8 does.

| Part | What we do |
|---|---|
| 1. Write `attention` | the scores, the causal mask, the softmax, the read, for every head at once; the hand-worked head of §3 |
| 2. Assemble the decoder | from token ids through the four blocks to the logits; agreement with the trained model; every parameter counted |
| 3. Check the mask | change the future; one pass against one prefix at a time; a model trained with a wrong mask |
| 4. Train | the training step of §8, repeated 200 times from random weights |
| 5. Generate | the generation loop of §8: next-word distributions, temperature, a story |
| At home | GPT-2 small's parameters; what one head reads; what removing a head costs |

Two models are loaded from `checkpoints/`: the word model (1,904,640 parameters) and the same model trained for 300 steps with a wrong mask. The stories are TinyStories' validation file, split as in Week 2 (downloaded into `data/` on the first run, 19 MB). Every cell runs in seconds on a laptop, except the short training run of Part 4 (a few minutes).
""" % NOTES)

    code(r"""
import sys, json, math, time, pathlib
ROOT = pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT))                 # use the repository's tinylm without installing it

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tinylm.data import tinystories_words
from tinylm.neural import word_index, story_tensors, detokenize
from tinylm.transformer import (TransformerLM, load_word_model, attention_solution, leaky_attention, count_parameters,
                                log_probs, prefix_log_probs, story_log_loss, train, generate, WORD_MODEL)

torch.set_num_threads(2)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
CKPT = ROOT / "checkpoints"
data = tinystories_words(ROOT / "data")
vocab = data["vocab"]
index = word_index(vocab)                     # the 8,681 words, then BOS ("<s>") last
BOS = index["<s>"]
print(f"{len(data['train']):,} training, {len(data['val']):,} validation, {len(data['test']):,} test stories; "
      f"{len(vocab):,} words, plus BOS")

sentence = "lily saw a big dog in the park . she gave it a bone and it was happy .".split()
x_run = torch.tensor([BOS] + [index[w] for w in sentence[:-1]])     # the inputs: BOS, then all but the last word
y_run = torch.tensor([index[w] for w in sentence])                  # the targets: the next word at every position
""")

    # ============================================================================ 1
    md(r"""
# 1. Write `attention`

In one head, every position \(t\) compares its query \(\mathbf q_t\) with the keys of the positions it may read, turns the scores into weights with a softmax, and reads a weighted average of their values (notes §3). For a whole sequence at once, with \(Q\), \(K\), \(V\) holding one row per position:

\[
S=\frac{QK^\top}{\sqrt{d_k}}+M,\qquad A=\operatorname{softmax}(S)\ \text{(row by row)},\qquad \text{read}=AV,
\]

where the **causal mask** \(M\) is \(0\) where position \(t\) may read position \(j\) (\(j\le t\)) and \(-\infty\) above the diagonal, so that those weights are \(e^{-\infty}=0\). The model computes \(Q\), \(K\), \(V\) for all \(H\) heads at once and passes them in as tensors of shape `(B, H, T, d_k)`: a batch of \(B\) chunks, \(H\) heads, \(T\) positions, \(d_k\) numbers per query, key, and value. Every head and every chunk is computed by the same matrix products.

Write the shape of every intermediate tensor next to its line.
""")

    if solution:
        code(ANSWERS.ATTENTION)
    else:
        code(r"""
def attention(q, k, v):
    # q, k, v: (B, H, T, d_k)
    # 1. the scores Q K^T / sqrt(d_k), shape (B, H, T, T)
    # 2. the causal mask: -inf above the diagonal (torch.triu, masked_fill)
    # 3. a softmax along each row: the weights, shape (B, H, T, T)
    # 4. the read, the weighted average of the values, shape (B, H, T, d_k)
    raise NotImplementedError("write attention")
    return read, weights
""")

    md(r"""
Three checks on random tensors, then the head the notes compute by hand (§3, "One head by hand"): five tokens, *lily saw tom . she*, whose vectors have five features (name, female, action, end of sentence, pronoun), and maps \(W_Q\), \(W_K\), \(W_V\) chosen by hand so that a pronoun looks for a female name. Before running: which word should *she* read most?
""")

    code(r"""
torch.manual_seed(0)
q, k, v = torch.randn(2, 3, 7, 4), torch.randn(2, 3, 7, 4), torch.randn(2, 3, 7, 4)
read, weights = attention(q, k, v)

# check 1: shapes
assert read.shape == (2, 3, 7, 4) and weights.shape == (2, 3, 7, 7), (read.shape, weights.shape)
print("check 1 passed: read", tuple(read.shape), " weights", tuple(weights.shape))

# check 2: every row of weights is a distribution over the positions up to its own
assert torch.allclose(weights.sum(-1), torch.ones(2, 3, 7), atol=1e-6)
assert (torch.triu(weights, diagonal=1) == 0).all()
print("check 2 passed: every row sums to 1, and every weight above the diagonal is 0")

# check 3: position 0 can read only itself, so its read is its own value
assert torch.allclose(read[:, :, 0], v[:, :, 0], atol=1e-6)
print("check 3 passed: position 0 reads only its own value")
""")

    code(r"""
words = ["lily", "saw", "tom", ".", "she"]
#                   name female action end pronoun
X = torch.tensor([[1., 1, 0, 0, 0],      # lily
                  [0., 0, 1, 0, 0],      # saw
                  [1., 0, 0, 0, 0],      # tom
                  [0., 0, 0, 1, 0],      # .
                  [0., 1, 0, 0, 1]])     # she
W_Q = torch.tensor([[0., 0], [0, 0], [2, 0], [0, 0], [2, 2]])     # queries: (wants a name, wants a female)
W_K = torch.tensor([[1., 0], [0, 1], [0, 0], [0, 0], [0, 0]])     # keys: (is a name, is female)
W_V = torch.tensor([[1., -1], [0, 2], [0, 0], [0, 0], [0, -2]])   # values: (a character, lily rather than tom)
read, weights = attention((X @ W_Q)[None, None], (X @ W_K)[None, None], (X @ W_V)[None, None])   # B = H = 1

print("weights of 'she':", {w: round(a, 2) for w, a in zip(words, weights[0, 0, 4].tolist())})
print("read of 'she':   ", [round(r, 2) for r in read[0, 0, 4].tolist()])
assert torch.allclose(read[0, 0, 4], torch.tensor([0.77, 0.47]), atol=0.006)
print("check 4 passed: the numbers of the notes")

fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.imshow(weights[0, 0], cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(5), words); ax.set_yticks(range(5), words)
ax.set_xlabel("position read (key)"); ax.set_ylabel("position reading (query)")
plt.show()
""")

    # ============================================================================ 2
    md(r"""
# 2. Assemble the decoder

The trained model's parts are loaded with **your** `attention` inside every layer. Write the decoder's forward pass from them: the left side of Figure 34 of the notes, and the table "One batch, step by step" in §8.

1. **Embed:** \(\mathbf h_t=E[x_t]\), a row of the token table for every position: `model.tok(x)`, shape `(B, T, d)`.
2. **Blocks:** for each of `model.blocks`, \(\mathbf h\leftarrow\mathbf h+\operatorname{Attention}(\operatorname{LayerNorm}_1(\mathbf h))\), then \(\mathbf h\leftarrow\mathbf h+\operatorname{MLP}(\operatorname{LayerNorm}_2(\mathbf h))\). A block has `norm1`, `attn`, `norm2`, and `mlp`.
3. **Logits:** \(\operatorname{LayerNorm}(\mathbf h)\,E^\top\), the token table used in reverse: `model.norm` and `model.tok.weight`, shape `(B, T, |V|)`.
4. **BOS is never predicted:** set the logits of `model.no_output` to \(-\infty\).
""")

    if solution:
        code(ANSWERS.DECODER)
    else:
        code(r"""
def decoder(model, x):
    # x: (B, T) token ids
    # 1. embed, (B, T, d)
    # 2. the blocks, each adding an attention read and then an MLP output to h
    # 3. the logits, (B, T, |V|)
    # 4. the logit of model.no_output set to -inf
    raise NotImplementedError("write decoder")
    return logits
""")

    code(r"""
model = load_word_model(CKPT / "week04_word_transformer.pt", attention=attention)      # your attention inside
reference = load_word_model(CKPT / "week04_word_transformer.pt")                        # tinylm's own

tx, ty = story_tensors(data["train"], index)
X_train, Y_train = torch.cat(tx), torch.cat(ty)            # the training stories, joined one after another
gen = torch.Generator().manual_seed(0)
starts = torch.randint(0, len(X_train) - 256, (16,), generator=gen)
x = torch.stack([X_train[s:s + 256] for s in starts])      # a batch of B = 16 chunks of T = 256 words
y = torch.stack([Y_train[s:s + 256] for s in starts])

with torch.no_grad():
    logits = decoder(model, x)
    gap = (logits - reference(x)).abs()[..., :-1].max().item()
assert logits.shape == (16, 256, 8682), logits.shape
assert gap < 1e-4, gap
loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
print(f"check 5 passed: logits {tuple(logits.shape)}, largest difference from the reference {gap:.1e}")
print(f"the batch's average log loss: {loss.item():.3f} nats per word (perplexity {math.exp(loss.item()):.1f}), "
      f"over {y.numel():,} predictions made in one pass")
""")

    md(r"""
**Every parameter.** A block has \(4d^2\) weights in attention (\(W_Q\), \(W_K\), \(W_V\), \(W_O\), whatever the number of heads) and \(8d^2\) in its MLP, so \(L\) blocks hold about \(12Ld^2\); the token table holds \(|\mathcal V|d\) more, shared with the output layer (notes §10). Before running: which is larger in this model, the blocks or the table?
""")

    code(r"""
d, L, V = 128, 4, len(index)
groups = {"token table (also the output layer)": model.tok.weight.numel(),
          "attention: W_Q, W_K, W_V, W_O and biases": sum(p.numel() for b in model.blocks for p in b.attn.parameters()),
          "MLPs: W_1, W_2 and biases": sum(p.numel() for b in model.blocks for p in b.mlp.parameters()),
          "layer normalizations": sum(p.numel() for b in model.blocks for n in (b.norm1, b.norm2) for p in n.parameters())
                                  + sum(p.numel() for p in model.norm.parameters())}
for name, n in groups.items():
    print(f"{n:>10,}  {name}")
print(f"{count_parameters(model):>10,}  in all;   12 L d^2 = {12 * L * d * d:,};   |V| d = {V * d:,}")
assert count_parameters(model) == sum(groups.values()) == 1904640
""")

    # ============================================================================ 3
    md(r"""
# 3. Check the mask

The argument of §3 shows that a correct implementation cannot use the future. An implementation can still be wrong, and a mask that lets a prediction see its answer makes both the training loss and the held-out loss look excellent, since held-out text is fed in the same way. Two tests catch it (§8, "Checking the mask"). The second model, `leaky`, was trained for 300 steps with a mask that also lets each position read the next one.

**Test 1, change the future.** Replace the input *she* in the running sentence by *he*, and measure, at every position, the largest change of any log-probability of the next word. Nothing before the replaced position may change.
""")

    code(r"""
leaky = load_word_model(CKPT / "week04_leaky.pt", attention=leaky_attention)
k = sentence.index("she") + 1                    # the position whose input is "she" (BOS is position 0)
x_he = x_run.clone(); x_he[k] = index["he"]
labels = ["<s>"] + sentence[:-1]

fig, ax = plt.subplots(figsize=(8, 2.6))
for m, name, color in [(model, "your model", "#2E5EAA"), (leaky, "wrong mask", "#8A8F98")]:
    with torch.no_grad():
        diff = torch.log_softmax(m(x_he[None])[0], -1) - torch.log_softmax(m(x_run[None])[0], -1)
        change = diff[:, :BOS].abs().max(-1).values          # over the words (BOS is never predicted: its logit is -inf)
    print(f"{name:>10s}: largest change before the replaced position {change[:k].max().item():.1e}, "
          f"at the position just before it {change[k - 1].item():.2f}")
    ax.plot(range(len(labels)), change.clamp(min=1e-8), "o-", color=color, label=name)
ax.axvline(k, color="#D62728", lw=0.8); ax.set_yscale("log")
ax.set_xticks(range(len(labels)), labels, rotation=60); ax.set_ylabel("largest change (nats)")
ax.legend(frameon=False); plt.show()
""")

    md(r"""
**Test 2, one pass against one prefix at a time.** In use, a model only ever sees prefixes. Compute the log loss of short test stories in one pass, and again by feeding each prefix alone and keeping only the prediction at its last position, where there is no next input to read. The two must agree up to rounding.
""")

    code(r"""
sx, sy = story_tensors(data["test"], index)
short = [i for i in range(len(sx)) if len(sx[i]) <= 120][:20]          # 20 short test stories
for m, name in [(model, "your model"), (leaky, "wrong mask")]:
    one_pass = -torch.cat([log_probs(m, sx[i], sy[i]) for i in short]).mean().item()
    prefixes = -torch.cat([prefix_log_probs(m, sx[i], sy[i]) for i in short]).mean().item()
    print(f"{name:>10s}: one pass {one_pass:.4f} nats per word, one prefix at a time {prefixes:.4f}, "
          f"difference {abs(one_pass - prefixes):.1e}")
""")

    md(r"""
**Two sentences.** Why does the model with the wrong mask reach a loss far below the honest model's in one pass, and why would a separate test set not reveal it? Why is the second test also the definition of what the model should compute?
""")
    if solution:
        md(ANSWERS.MASK)

    # ============================================================================ 4
    md(r"""
# 4. Train

The training step of §8, repeated (`train` in `tinylm/transformer.py`):

1. **Sample a batch:** 16 chunks of 256 consecutive positions from the training stories, joined one after another.
2. **Forward:** the logits at every position of every chunk, and the average log loss of the 4,096 predictions.
3. **Backward:** the gradient of the loss with respect to every parameter, by backpropagation.
4. **Update:** move each parameter against its gradient, with Adam and a learning rate that rises for 100 steps and then decays (Lecture 5 explains both).

A new model, with random weights and your `attention`, trained for 200 steps (a few minutes). Before running: the loss of a model that gives every word the same probability is \(\ln 8{,}681\approx9.07\) nats. Where will it start, and where will it be after 200 steps?
""")

    code(r"""
QUICK = True                                       # 200 steps; set False for 1,000 (about five times as long)
torch.manual_seed(0)
fresh = TransformerLM(len(index), no_output=BOS, **WORD_MODEL)
fresh.set_attention(attention)
curve = train(fresh, X_train, Y_train, steps=200 if QUICK else 1000, every=25)

trained = json.load(open(CKPT / "week04_word_transformer.json")) if (CKPT / "week04_word_transformer.json").exists() else None
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.plot([s for s, l in curve], [math.exp(l) for s, l in curve], "o-", color="#2E5EAA", label="this run (training batches)")
if trained:
    ax.plot([c[0] for c in trained["curve"]], [c[2] for c in trained["curve"]], "s-", color="#596273",
            label="the checkpoint (validation)")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("perplexity"); ax.legend(frameon=False)
plt.show()
""")

    code(r"""
for m, name in [(fresh, f"after {curve[-1][0]} steps"), (model, "after 4,000 steps")]:
    ids = generate(m, [BOS], 40, temperature=0.7, seed=3, stop_id=index["</s>"])
    print(f"{name}: {detokenize([vocab[i] for i in ids])}\n")
""")

    # ============================================================================ 5
    md(r"""
# 5. Generate

Generation is the loop of §8: feed the text so far, take the distribution at the last position, choose a word from it, append it, and repeat. First the distributions themselves (Figure 36 of the notes): the five most probable next words after three prompts.
""")

    code(r"""
for prompt in ["she was very", "they went to the", "lily saw a big dog in the park . she gave it a"]:
    ids = torch.tensor([[BOS] + [index[w] for w in prompt.split()]])
    with torch.no_grad():
        p = torch.softmax(model(ids)[0, -1], -1)
    top = torch.topk(p, 5)
    print(f"{prompt:>48s} ->  " + ",  ".join(f"{vocab[j]} {v:.2f}" for v, j in zip(top.values.tolist(), top.indices.tolist())))
""")

    md(r"""
**Temperature.** Dividing the logits by a temperature \(\tau\) before the softmax sharpens the distribution when \(\tau<1\) and flattens it when \(\tau>1\) (Week 2). Three stories from BOS, with the same seed:
""")

    code(r"""
for tau in [0.5, 0.8, 1.3]:
    ids = generate(model, [BOS], 70, temperature=tau, seed=0, stop_id=index["</s>"])
    print(f"temperature {tau}: {detokenize([vocab[i] for i in ids])}\n")
""")

    md(r"""
**Write two sentences.** Every step of this loop runs the whole prefix through the model again and uses only the last position. What is recomputed that could be kept, and why can it be kept (notes §10, "The key–value cache")?
""")
    if solution:
        md(ANSWERS.CACHE)

    # ============================================================================ at home
    md(r"""
# At home

**GPT-2 small.** The same module with GPT-2's sizes: \(|\mathcal V|=50{,}257\), learned positions for a context of 1,024, \(L=12\) blocks of width \(d=768\), 12 heads. Count its parameters without allocating them (the `meta` device), and compare with \(12Ld^2\) and the table (notes §10).
""")

    code(r"""
with torch.device("meta"):
    gpt2 = TransformerLM(50257, d_model=768, n_layers=12, n_heads=12, max_len=1024, positions="learned")
n = count_parameters(gpt2)
print(f"{n:,} parameters;  12 L d^2 = {12 * 12 * 768**2:,};  token table {50257 * 768:,};  "
      f"position table {1024 * 768:,}")
""")

    md(r"""
**What one head reads.** The weights of every head on the running sentence. Appendix D of the notes shows block 2, head 4: from *she* it reads *lily*, and from each *it* mostly *dog* and *gave*. Find other heads with a clear pattern: one that reads the previous word, one that reads BOS.
""")

    code(r"""
model.keep_attention(True)
with torch.no_grad():
    model(x_run[None])
A = model.attention_weights()                    # a list over blocks of (1, H, T, T)
model.keep_attention(False)

fig, axes = plt.subplots(4, 4, figsize=(11, 11))
for b in range(4):
    for h in range(4):
        ax = axes[b, h]
        ax.imshow(A[b][0, h], cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"block {b + 1}, head {h + 1}", fontsize=8)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5)
        ax.set_yticks(range(len(labels)), labels, fontsize=5)
plt.tight_layout(); plt.show()
""")

    md(r"""
**What removing a head costs.** Set one head's read to zero and measure the validation perplexity on 300 stories (about a minute per head on a laptop). The notes found that every head matters somewhat, and that a head whose pattern looks like a relation is not necessarily the one the model relies on for it.
""")

    code(r"""
def without_head(head):
    def attend(q, k, v):
        read, weights = attention(q, k, v)
        read = read.clone()
        read[:, head] = 0.0
        return read, weights
    return attend


vx, vy = story_tensors(data["val"][:300], index)
total, count = story_log_loss(model, vx, vy)
base = math.exp(total / count)
print(f"all heads: validation perplexity {base:.2f}")
for b, h in [(1, 3), (1, 2)]:                        # block 2, heads 4 and 3 (counted from 1)
    model.blocks[b].attn.attention = without_head(h)
    total, count = story_log_loss(model, vx, vy)
    model.blocks[b].attn.attention = attention
    print(f"without block {b + 1}, head {h + 1}: {math.exp(total / count):.2f}")
""")

    md(r"""
**Next week** the decoder becomes a base model trained on far more text: how text becomes tokens, the optimizer and its schedule, how large a model should be for its budget, and what the base model can do; then a small chat model, by fine-tuning on conversations.
""")

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    targets = [(False, "week04_transformer.ipynb")]
    if ANSWERS is not None:
        targets.append((True, "week04_transformer_solution.ipynb"))
    for solution, name in targets:
        nbf.write(build(solution), folder / name)
        print("wrote", folder / name)
