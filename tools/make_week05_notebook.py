"""Write the Week 5 notebooks: from text to a chat model.

    python3 tools/make_week05_notebook.py

notebooks/week05_pretraining.ipynb has an empty train_by_hand(), adam_step(), and render() for students to
write. With the instructor's answers file (tools/week05_answers.py, not in the public repository) the script
also writes notebooks/week05_pretraining_solution.ipynb, the same notebook with the answers filled in; it is
saved with its outputs after running
    jupyter nbconvert --to notebook --execute --inplace notebooks/week05_pretraining_solution.ipynb

The notebook follows §§2-13 of the Lecture 5 notes. The notes' base model, midtrained model, and chat model,
their tokenizer, the notes' runs, and the fine-tuning conversations are loaded from checkpoints/; two short
runs are live: four optimizers on a small model (Part 3, a minute or two) and the fine-tuning of the base
model on 1,000 conversations (Part 6, a few minutes). The whole solution notebook runs in about a quarter of an hour
on two processor threads, the LoRA runs at home included.
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import week05_answers as ANSWERS        # the instructor's answers; not in the public repository
except ImportError:
    ANSWERS = None

NOTES = "https://mathrulestheworld.github.io/genai-first-principles/materials/week-05/lecture-05-notes.pdf"


def build(solution):
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md(r"""
# Week 5: from text to a chat model

*Generative AI from First Principles*, Lecture 5 · [course page](https://mathrulestheworld.github.io/genai-first-principles/) · [notes](%s)

Lecture 5 takes Week 4's decoder through the steps that made GPT-3 and then ChatGPT. Text becomes tokens; a base model of 3,456,000 parameters is pretrained on 49 million of them; and fine-tuning on conversations, written in a chat template, turns it into a model that answers a request and stops. The notes' tokenizer and models are in `checkpoints/`, so every step can be checked against them. You write three pieces, the training loop of byte-pair encoding, one step of Adam, and the chat template with its loss mask, and then fine-tune the base model into a chat model of your own.

| Part | What we do |
|---|---|
| 1. Tokens | write byte-pair encoding and check it on the notes' ten words; the same loop on the bytes of stories; a round trip; our 4,096-token tokenizer and the running conversation in tokens |
| 2. Optimizers | write one step of Adam and check the notes' two steps by hand; gradient descent, momentum, and Adam on a quadratic |
| 3. Pretraining | packed batches; the learning-rate schedule; four optimizers on a small model; the base model's loss, in nats per token and bits per byte, and its stories |
| 4. How large a model? | the IsoFLOP runs of the notes: the best size for each budget, and how it grows |
| 5. The base model | sums from a few examples; copying an invented name, and the head that does it; the running conversation, which it continues and never answers |
| 6. A chat model | write the template and its loss mask; fine-tune the base model on 1,000 conversations; the notes' chat model |
| At home | LoRA, with and without the rows of the new tokens; summaries before and after fine-tuning; the system message |

Four models and three files are loaded from `checkpoints/`: the base model (`week05_base.pt`, trained for six hours on a processor), the midtrained model (`week05_mid.pt`), the chat model (`week05_chat.pt`), the tokenizer (`week05_tokenizer.json`), the notes' runs (`week05_runs.json`), and the conversations of fine-tuning (`week05_conversations.json`). The stories are TinyStories' validation file, split as in Week 2 (downloaded into `data/` on the first run, 19 MB); the base model was trained on TinyStories' training file, a different set of stories. Most cells run in seconds on a laptop; the comparison of optimizers in Part 3 takes a minute or two, and the fine-tuning of Part 6 with its measurements a few minutes.
""" % NOTES)

    code(r"""
import sys, json, math, time, random, pathlib
from collections import Counter
ROOT = pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT))                 # use the repository's tinylm without installing it

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from tinylm.data import download_tinystories, load_stories, split_stories
from tinylm.transformer import TransformerLM, attention_solution, count_parameters
from tinylm.bpe import pretokens, count_pretokens, train_bpe, Tokenizer, load_tokenizer, bits_per_byte
from tinylm.pretrain import (BASE_MODEL, load_lm, fast_attention, pack, get_batch, lr_at, pretrain, stories_loss,
                             token_losses, sample)
from tinylm.finetune import (render_solution, prompt_ids, to_example, assistant_loss, finetune, answer, words_used,
                             add_lora, train_rows, merge_lora, trained_parameters)

torch.set_num_threads(2)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
CKPT = ROOT / "checkpoints"
train_stories, val_stories, test_stories = split_stories(load_stories(download_tinystories(ROOT / "data")))
tok = load_tokenizer(CKPT / "week05_tokenizer.json")
runs = json.load(open(CKPT / "week05_runs.json"))
conversations = json.load(open(CKPT / "week05_conversations.json", encoding="utf-8"))

EOT, SYS, USR, AST, END = (tok.special_ids[s] for s in ("<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"))
ROLE = {"system": SYS, "user": USR, "assistant": AST}
SYSTEM = "You are a storyteller for young children."
RUNNING = [{"role": "system", "content": SYSTEM},                                   # the running conversation
           {"role": "user", "content": "Tell me a short story about a dog who finds a bone."}]
print(f"{len(train_stories):,} training, {len(val_stories):,} validation, {len(test_stories):,} test stories; "
      f"a tokenizer of {len(tok):,} tokens")
""")

    # ============================================================================ 1
    md(r"""
# 1. Tokens

A model reads a sequence of tokens from a fixed vocabulary. Words make a short sequence but leave rare words unknown; bytes leave nothing unknown but make the sequence long (notes §2). **Byte-pair encoding** (BPE, notes §3) learns a vocabulary between the two from the text itself: start from single characters and repeatedly merge the most frequent pair of adjacent tokens into a new token.

Write `train_by_hand(word_counts, n_merges)`, the training loop of the notes' Figure (*BPE by hand*), on a corpus given as words with their counts:

1. Write every word as a list of single characters.
2. Count every pair of adjacent tokens inside the words, each occurrence weighted by its word's count.
3. Take the most frequent pair; when two pairs tie, take the one that comes first in the alphabet (Python's `min` on tuples of strings does this). Record the merge as `(pair, count)`, and replace the pair by one token, the two strings joined, everywhere, scanning each word from left to right.
4. Repeat from step 2, `n_merges` times.

Return the list of merges and the words as lists of tokens.
""")

    if solution:
        code(ANSWERS.TRAIN_BY_HAND)
    else:
        code(r"""
def train_by_hand(word_counts, n_merges):
    # word_counts: {"hug": 10, "hugs": 5, ...}
    # 1. words = {word: list of its characters}
    # 2. pairs = Counter() of adjacent pairs, each weighted by word_counts[word]
    # 3. best = the most frequent pair, ties broken by min(); merges.append((best, pairs[best]));
    #    then replace best by one token in every word, left to right
    # 4. repeat n_merges times
    raise NotImplementedError("write train_by_hand")
    return merges, words
""")

    md(r"""
The notes' corpus: ten words, 50 occurrences, 180 characters. Before running: which pair is merged first, and how often does it occur?
""")

    code(r"""
corpus = {"hug": 10, "hugs": 5, "hugged": 4, "bug": 6, "bugs": 3, "dog": 8, "dogs": 4, "big": 5, "bigger": 2, "bag": 3}
merges, words = train_by_hand(corpus, 6)
for k, ((a, b), n) in enumerate(merges, 1):
    print(f"merge {k}: {a}·{b}   ({n} times)")
n_chars = sum(len(w) * c for w, c in corpus.items())
n_tokens = sum(len(words[w]) * c for w, c in corpus.items())
print(f"\n{sum(corpus.values())} words, {n_chars} characters -> {n_tokens} tokens")
for w in corpus:
    print(f"   {w:7s} x{corpus[w]:<3d} {'·'.join(words[w])}")

expected = [(("u", "g"), 28), (("h", "ug"), 19), (("d", "o"), 12), (("do", "g"), 12), (("b", "ug"), 9), (("b", "i"), 7)]
assert merges == expected, merges
assert n_tokens == 93, n_tokens
print("\ncheck passed: the notes' six merges, and 93 tokens")
""")

    md(r"""
**Encoding** a new word applies the learned merges in the order they were learned (notes §3). A word the corpus never contained is spelled from the pieces it shares with the corpus:
""")

    code(r"""
def encode_by_hand(word, merges):
    tokens = list(word)
    for (a, b), _ in merges:                      # in the order they were learned
        out, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == (a, b):
                out.append(a + b)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens

for w in ["dug", "hugging", "dogs", "bigger"]:
    print(f"{w:8s} -> {'·'.join(encode_by_hand(w, merges))}")
""")

    md(r"""
## Byte-level BPE on stories

GPT-2's tokenizer runs the same loop on the 256 bytes of UTF-8, so that every text can be encoded (notes §3). First the text is split into **pre-tokens** by a regular expression, a word with the space before it, one digit, a run of punctuation, and merges happen only inside a pre-token. `tinylm/bpe.py` has the loop on bytes as `train_bpe`, with the pair counts updated after each merge instead of recounted, so that hundreds of merges take a second. Pre-tokens first (␣ is a space):
""")

    code(r"""
for text in ['Lily said, "Let\'s play!"', "237 + 682 = 919.", "She didn't know."]:
    print(" | ".join(p.replace(" ", "␣") for p in pretokens(text)))
""")

    md(r"""
Train 744 merges on 5,000 training stories, a vocabulary of 1,000 tokens (256 bytes and 744 merges), and look at the first merges. Then the **round trip**: decoding the tokens of any text gives back the text, for every validation story and for a string with characters that no story contains.
""")

    code(r"""
t = time.time()
counts = count_pretokens(train_stories[:5000])
small = Tokenizer(train_bpe(counts, 744))
print(f"{len(counts):,} distinct pre-tokens; 744 merges in {time.time() - t:.1f} s")
print("first merges:", ", ".join(repr(small.token_bytes[256 + k].decode()) for k in range(12)))

assert all(small.decode(small.encode(s)) == s for s in val_stories)
odd = "Zoë saw 🐉 and said “你好” — 3.14!"
ids = small.encode(odd)
assert small.decode(ids) == odd
print(f"\nround trip passed on {len(val_stories):,} validation stories and on {odd!r}")
print(f"{len(odd)} characters, {len(odd.encode())} bytes, {len(ids)} tokens:", small.pieces(ids))
""")

    md(r"""
The pieces shown as `<xx>` are single bytes of characters the stories never contained: the emoji and the Chinese characters are spelled byte by byte, and the round trip still holds.

## Our tokenizer

The notes' tokenizer has 4,096 tokens: the 256 bytes, 3,832 merges learned on 100,000 stories of TinyStories' training file, and 8 special tokens (notes §§3–4). Count the bytes per token on the validation stories for both tokenizers, then look at Lecture 4's running sentence, the running conversation, and a sum.
""")

    code(r"""
n_bytes = sum(len(s.encode()) for s in val_stories)
for name, t in [("bytes alone", None), ("1,000 tokens (above)", small), ("4,096 tokens (the notes')", tok)]:
    n = n_bytes if t is None else sum(len(t.encode(s)) for s in val_stories)
    print(f"{name:26s} {n / len(val_stories):6.0f} tokens per story   {n_bytes / n:.2f} bytes per token")

def show(text):
    ids = tok.encode(text)
    print(f"{len(ids):3d} tokens: " + " ".join(p.replace(" ", "␣").replace("\n", "↵") for p in tok.pieces(ids)))

print()
show("Lily saw a big dog in the park. She gave it a bone and it was happy.")
show(RUNNING[0]["content"])
show(RUNNING[1]["content"])
show("237 + 682 = 919.")
""")

    md(r"""
The words of the stories are single tokens, with the space before them; *storyteller* and *Tell*, rare in children's stories, are spelled from pieces; and every digit is its own token, so a number is always written digit by digit (notes §4).

The **special tokens** come after the merges, ids 4,088 to 4,095. Before running: what does the encoder give for the text `<|end|>`, typed by a user?
""")

    code(r"""
print({s: i for s, i in tok.special_ids.items()})
ids = tok.encode("<|end|>")
print("tok.encode('<|end|>') =", ids, tok.pieces(ids))
assert END not in ids
""")

    if solution:
        md(ANSWERS.SPECIAL)

    # ============================================================================ 2
    md(r"""
# 2. Optimizers

Pretraining minimizes the average log loss of next-token prediction by gradient steps; which steps work is the subject of the notes' §6. **Adam** keeps two moving averages for every parameter, of its gradients \(g\) and of their squares, and steps by their ratio. At step \(k\) (counted from 1), with \(m\) and \(v\) starting at zero:

1. \(m \leftarrow \beta_1 m + (1-\beta_1)\,g\) and \(v \leftarrow \beta_2 v + (1-\beta_2)\,g^2\), coordinate by coordinate;
2. the bias corrections \(\hat m = m/(1-\beta_1^k)\) and \(\hat v = v/(1-\beta_2^k)\), since both averages start at zero;
3. \(\theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)\).

Each coordinate's step is about \(\eta\) times the sign of its recent gradients, whatever their size. Write `adam_step`, which updates `state`, a dictionary with the step count under `"step"` and the tensors `m` and `v`, and returns the new parameters.
""")

    if solution:
        code(ANSWERS.ADAM_STEP)
    else:
        code(r"""
def adam_step(theta, grad, state, lr, betas=(0.9, 0.95), eps=1e-8):
    # theta, grad: tensors of the same shape; state: {"step": k, "m": tensor, "v": tensor}, from adam_state()
    # 1. state["step"] += 1; update state["m"] and state["v"]
    # 2. the bias-corrected m_hat and v_hat
    # 3. return theta - lr * m_hat / (sqrt(v_hat) + eps)
    raise NotImplementedError("write adam_step")
""")

    md(r"""
Two checks. The first is the notes' step by hand: two parameters, \(\eta=0.01\), \(\beta=(0.9, 0.95)\), with gradients \(g_1=(0.01,\ 2)\) at step 1, two hundred times apart, and \(g_2=(0.03,\ -1)\) at step 2. Before running: how far does each parameter move at step 1? The second compares 100 steps with PyTorch's Adam.
""")

    code(r"""
def adam_state(theta):
    return {"step": 0, "m": torch.zeros_like(theta), "v": torch.zeros_like(theta)}

# check 1: the notes' two steps by hand
theta = torch.zeros(2, dtype=torch.float64)
state = adam_state(theta)
for g in ([0.01, 2.0], [0.03, -1.0]):
    new = adam_step(theta, torch.tensor(g, dtype=torch.float64), state, lr=0.01)
    move = (theta - new).abs()
    print(f"step {state['step']}: gradient {g}, Adam moves the parameters by {move.numpy().round(4)}, "
          f"gradient descent at the same rate would by {(0.01 * torch.tensor(g).abs()).numpy().round(4)}")
    theta = new
    if state["step"] == 1:
        assert torch.allclose(move, torch.tensor([0.01, 0.01], dtype=torch.float64), atol=1e-6)
    else:
        assert move.numpy().round(4).tolist() == [0.0091, 0.0027], move
print("check 1 passed: (0.01, 0.01), then (0.0091, 0.0027)")

# check 2: 100 steps against torch.optim.Adam on random gradients
torch.manual_seed(0)
p = torch.randn(5, dtype=torch.float64, requires_grad=True)
mine = p.detach().clone()
opt = torch.optim.Adam([p], lr=0.003, betas=(0.9, 0.95), eps=1e-8)
state = adam_state(mine)
for _ in range(100):
    g = torch.randn(5, dtype=torch.float64) * torch.tensor([1e-3, 1e-1, 1.0, 10.0, 1e3], dtype=torch.float64)
    p.grad = g.clone()
    opt.step()
    mine = adam_step(mine, g, state, lr=0.003)
assert torch.allclose(mine, p.detach(), atol=1e-12), (mine, p)
print("check 2 passed: 100 steps equal torch.optim.Adam's, on gradients from 0.001 to 1,000")
""")

    md(r"""
At step 1 both parameters move by \(\eta\), where gradient descent would move the second 200 times as far as the first. At step 2 the first gradient has kept its sign and the second has flipped: Adam moves the first almost as far again, and the second by much less, since its average gradient is small against its typical size.

## A quadratic

Near a minimum, a loss looks like a quadratic bowl. Take the notes' \(\mathcal L(\theta)=\tfrac12(\theta_1^2+25\,\theta_2^2)\), 25 times steeper along \(\theta_2\) than along \(\theta_1\), from \(\theta=(-4, 1)\), for 30 steps of:

1. gradient descent, \(\theta \leftarrow \theta-\eta\,\nabla\mathcal L\), at \(\eta=0.02\) and at \(\eta=0.075\) (it diverges above \(2/25=0.08\));
2. momentum, \(\mathbf v \leftarrow \beta\mathbf v+\nabla\mathcal L\), \(\theta\leftarrow\theta-\eta\mathbf v\), at \(\eta=0.04\), \(\beta=0.7\);
3. your Adam at \(\eta=0.2\), \(\beta=(0.9, 0.999)\).

Before running: which ends lowest after 30 steps?
""")

    code(r"""
lam = torch.tensor([1.0, 25.0], dtype=torch.float64)
loss_q = lambda th: 0.5 * float((lam * th ** 2).sum())
start = torch.tensor([-4.0, 1.0], dtype=torch.float64)

def run(method, steps=30):
    th, path, v, st = start.clone(), [start.clone()], torch.zeros(2, dtype=torch.float64), adam_state(start)
    for _ in range(steps):
        g = lam * th
        if method[0] == "gd":
            th = th - method[1] * g
        elif method[0] == "momentum":
            v = method[2] * v + g
            th = th - method[1] * v
        else:
            th = adam_step(th, g, st, lr=method[1], betas=(0.9, 0.999))
        path.append(th.clone())
    return torch.stack(path)

methods = {"gradient descent, 0.02": ("gd", 0.02), "gradient descent, 0.075": ("gd", 0.075),
           "momentum, 0.04, 0.7": ("momentum", 0.04, 0.7), "Adam, 0.2": ("adam", 0.2)}
paths = {name: run(m) for name, m in methods.items()}
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
g1, g2 = np.meshgrid(np.linspace(-4.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
a.contour(g1, g2, 0.5 * (g1 ** 2 + 25 * g2 ** 2), levels=[0.1, 0.5, 1, 2, 4, 8, 16], colors="0.85", linewidths=0.8)
for name, p in paths.items():
    a.plot(p[:, 0], p[:, 1], ".-", ms=3, lw=1, label=name)
    b.semilogy([loss_q(th) for th in p], label=name)
a.set_xlabel(r"$\theta_1$ (flat)"); a.set_ylabel(r"$\theta_2$ (steep)"); a.set_aspect("equal")
b.set_xlabel("step"); b.set_ylabel("loss"); b.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
for name, p in paths.items():
    print(f"{name:24s} loss after 30 steps {loss_q(p[-1]):.4f}")
""")

    md(r"""
One step size cannot suit a flat and a steep direction: at 0.02 gradient descent crawls along \(\theta_1\), and at 0.075 it zig-zags along \(\theta_2\). Momentum adds up the gradients that keep their sign and cancels those that alternate. Adam divides each coordinate's gradient by its own size, so both coordinates move at the same pace; but at a fixed \(\eta\) it keeps taking steps of about \(\eta\) near the minimum, so it overshoots and circles it, and its loss rises again. The schedule of Part 3 lowers \(\eta\) as training ends for this reason (notes §7).
""")

    # ============================================================================ 3
    md(r"""
# 3. Pretraining

## Packed batches

The training text is one long stream: each story's tokens after an `<|endoftext|>` token, end to end (notes §5, *Packing*). A batch is \(B\) chunks of \(T+1\) consecutive tokens cut at random places; the inputs are each chunk's first \(T\) tokens and the targets its last \(T\), so that position \(t\) predicts token \(t+1\). A chunk may cross the end of a story, and then the model learns to predict `<|endoftext|>`, and the start of a new story after it. Two tiny documents, and two chunks of 6:
""")

    code(r"""
docs = [tok.encode("Tom saw a cat."), tok.encode("Sue had a red ball.")]
stream = pack(docs, EOT)
print("stream:", " ".join(p.replace(" ", "␣") for p in tok.pieces(stream.tolist())))
x, y = get_batch(stream, 2, 6, torch.Generator().manual_seed(3))
for i in range(2):
    print(f"\nchunk {i}  inputs: ", " ".join(f"{p.replace(' ', '␣'):>14s}" for p in tok.pieces(x[i].tolist())))
    print(f"         targets:", " ".join(f"{p.replace(' ', '␣'):>14s}" for p in tok.pieces(y[i].tolist())))
""")

    md(r"""
The training split as one stream, and the validation stories as lists of ids:
""")

    code(r"""
t = time.time()
train_stream = pack([tok.encode(s) for s in train_stories], EOT)
val_ids = [tok.encode(s) for s in val_stories]
print(f"{len(train_stream):,} training tokens, {sum(map(len, val_ids)):,} validation tokens ({time.time() - t:.0f} s)")
""")

    md(r"""
## The learning-rate schedule

The learning rate changes over training (notes §7): a linear **warmup** from zero, while Adam's averages are still unreliable, and then a **decay**, which lowers \(\eta\) as the loss nears its minimum. `lr_at(step, steps, peak, warmup, kind)` gives three schedules: a cosine decay to a tenth of the peak, a constant rate, and **warmup-stable-decay** (WSD), constant until the last fifth and then a linear decay. The base model used the cosine, with a peak of 0.002 and 200 steps of warmup over 6,000 steps:
""")

    code(r"""
steps = np.arange(1, 6001)
fig, ax = plt.subplots(figsize=(6, 2.6))
for kind in ["cosine", "constant", "wsd"]:
    ax.plot(steps, [lr_at(s, 6000, 2e-3, 200, kind) for s in steps], label=kind)
ax.set_xlabel("step"); ax.set_ylabel("learning rate"); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
""")

    md(r"""
## Four optimizers on a small model

`pretrain(model, stream, steps, ...)` is the training loop of the notes' §§7–8: a batch from the stream, the average log loss of every next-token prediction, backpropagation, the gradient clipped at norm 1, the learning rate of the schedule, one step of the optimizer. Train the same small model (2 blocks of width 64, our 4,096 tokens, context 128) from the same initialization with four optimizers, each at the rate that was best in the notes' sweep on a larger model (§6):

1. gradient descent (`sgd`), at 1;
2. momentum, \(\beta=0.9\), at 0.3;
3. AdamW, \(\beta=(0.9, 0.95)\), weight decay 0.1 on the matrices, at 0.01;
4. **Muon** at 0.01 on the blocks' matrices, which replaces each matrix's step by the nearest orthogonal matrix (notes §6), with AdamW at 0.01 on the token table, the normalizations, and the biases.

Then measure each on 200 validation stories. Before running: in which order will they finish?
""")

    code(r"""
QUICK = True                                    # 200 steps each, a minute or two in all; False: 800 steps
SMALL = dict(d_model=64, n_layers=2, n_heads=2, max_len=128)
n_steps = 200 if QUICK else 800
rates = {"sgd": 1.0, "momentum": 0.3, "adamw": 0.01, "muon": 0.01}
curves, small_models = {}, {}
for kind, rate in rates.items():
    torch.manual_seed(0)
    m = TransformerLM(len(tok), positions="rope", **SMALL)
    m.set_attention(fast_attention)
    t = time.time()
    curves[kind] = pretrain(m, train_stream, n_steps, batch_size=16, length=128, optimizer=kind, peak=rate,
                            warmup=n_steps // 8, log=None)
    total, count = stories_loss(m, val_ids[:200], EOT)
    small_models[kind] = m
    print(f"{kind:9s} at {rate:<5g} validation loss {total / count:.3f} nats per token   ({time.time() - t:.0f} s)")
print(f"\n{count_parameters(m):,} parameters")
fig, ax = plt.subplots(figsize=(6, 3))
for kind, c in curves.items():
    ax.plot(*zip(*c), label=kind)
ax.set_xlabel("step"); ax.set_ylabel("training loss (nats per token)"); ax.set_ylim(top=8.5); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
""")

    md(r"""
The order is the notes' (§6): Muon lowest, then AdamW, with momentum and gradient descent far behind. The notes' sweep ran five optimizers at five rates each on a larger model (4 blocks of width 128, 1.3 million parameters, 800 steps, about ten minutes a run); its results are in `runs["optimizers"]`. Each optimizer has a best rate, and the best rates span a factor of 100:
""")

    code(r"""
sweep = [r for r in runs["optimizers"]["runs"] if r["schedule"] == "cosine" and r["warmup"] == 100]
fig, ax = plt.subplots(figsize=(6, 3))
for kind in ["sgd", "momentum", "adam", "adamw", "muon"]:
    rs = sorted([r for r in sweep if r["optimizer"] == kind], key=lambda r: r["lr"])
    ax.plot([r["lr"] for r in rs], [r["val_loss"] for r in rs], "o-", ms=4, label=kind)
ax.set_xscale("log"); ax.set_xlabel("peak learning rate"); ax.set_ylabel("validation loss")
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
for kind in ["sgd", "momentum", "adam", "adamw", "muon"]:
    best = min((r for r in sweep if r["optimizer"] == kind), key=lambda r: r["val_loss"])
    print(f"{kind:9s} best at {best['lr']:<6g} {best['val_loss']:.3f}")
""")

    md(r"""
## The base model

The notes' base model (§8) is Week 4's decoder made deeper and wider: 6 blocks of width 192, 6 heads, a context of 512 tokens, rotary positions, and the 4,096-token table tied to the output layer. It was trained with AdamW for 6,000 steps of 16 × 512 tokens, 49 million tokens, in six hours on a processor. Its validation loss, on the first 500 validation stories, is reported in two units:

1. nats per token, the average log loss of the stories' tokens, each story read from `<|endoftext|>` and its end predicted too;
2. **bits per byte**, the same total in bits divided by the stories' bytes, which does not depend on the tokenizer, so that models with different vocabularies can be compared (notes §4).
""")

    code(r"""
base = load_lm(CKPT / "week05_base.pt")
print(f"{count_parameters(base):,} parameters")
total, count = stories_loss(base, val_ids[:500], EOT)
n_bytes = sum(len(s.encode()) for s in val_stories[:500])
print(f"validation loss {total / count:.3f} nats per token, {bits_per_byte(total, n_bytes):.3f} bits per byte "
      f"({n_bytes / count:.2f} bytes per prediction)")
assert abs(total / count - 1.818) < 0.001

curve = runs["base"]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3))
a.plot([s for s, _, _ in curve["train"]], [lr for _, _, lr in curve["train"]])
a.set_xlabel("step"); a.set_ylabel("learning rate")
b.plot([s for s, _, _ in curve["train"]], [l for _, l, _ in curve["train"]], color="0.7", lw=0.8, label="training")
b.plot([s for s, _, _ in curve["val"]], [l for _, l, _ in curve["val"]], "o-", ms=3, label="validation")
b.set_ylim(1.5, 5); b.set_xlabel("step"); b.set_ylabel("loss (nats per token)"); b.legend(frameon=False)
plt.tight_layout(); plt.show()
""")

    md(r"""
Given the running prompt of the notes' §6, the first sentence of Lecture 4's running sentence, the base model writes a story; the small models of the comparison above, trained for seconds, write sentences that do not fit together. `sample` draws one token at a time from the model's distribution at temperature 0.8 (Lecture 4, §8).
""")

    code(r"""
prompt = [EOT] + tok.encode("Lily saw a big dog in the park.")
out = sample(base, prompt, 80, temperature=0.8, seed=1, stop=(EOT,))
print("base model:", tok.decode(prompt[1:] + [i for i in out if i != EOT]), "\n")
for kind in ["sgd", "muon"]:
    out = sample(small_models[kind], prompt, 60, temperature=0.8, seed=1, stop=(EOT,))
    print(f"small model, {kind}:", tok.decode(prompt[1:] + [i for i in out if i != EOT]), "\n")
""")

    # ============================================================================ 4
    md(r"""
# 4. How large a model?

A budget of computation \(C\) buys a model of \(N\) parameters trained on \(D\) tokens, with \(C\approx 6ND\) operations (notes §9). A larger model reads fewer tokens; which split is best? The notes trained five sizes on each of three budgets, \(10^{13}\), \(3\times10^{13}\), and \(10^{14}\) operations, each reading \(D=C/6N\) tokens (the longest run took 69 minutes). Here are the runs, from `runs["scaling"]`:
""")

    code(r"""
scaling = runs["scaling"]
print(f"{'budget':>8s} {'size':>5s} {'d':>4s} {'L':>2s} {'N':>10s} {'tokens D':>12s} {'D/N':>5s} {'loss':>6s}")
for r in sorted(scaling, key=lambda r: (r["budget"], r["N"])):
    print(f"{r['budget']:8.0e} {r['size']:>5s} {r['d']:4d} {r['L']:2d} {r['N']:10,d} {r['tokens']:12,d} "
          f"{r['tokens'] / r['N']:5.0f} {r['val_loss']:6.3f}")
""")

    md(r"""
For each budget, fit a parabola to the loss against \(\log N\) and take its minimum, the best size \(N^*\) for that budget. Then fit a straight line to \(\log N^*\) against \(\log C\): its slope \(a\) is the exponent of the power law \(N^*\propto C^{a}\). Before running: Hoffmann et al. (2022) found \(a\approx0.5\) and about 20 tokens per parameter. What do you expect at our scale?
""")

    code(r"""
budgets = sorted({r["budget"] for r in scaling})
best = []
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
for C in budgets:
    rs = sorted([r for r in scaling if r["budget"] == C], key=lambda r: r["N"])
    N = np.array([r["N"] for r in rs], float)
    L = np.array([r["val_loss"] for r in rs])
    p = np.polyfit(np.log(N), L, 2)                        # loss = p0 (log N)^2 + p1 log N + p2
    n_star = math.exp(-p[1] / (2 * p[0]))                  # the parabola's minimum
    best.append((C, n_star))
    line, = a.plot(N, L, "o", label=f"C = {C:.0e}")
    xs = np.linspace(np.log(N.min()), np.log(N.max()), 100)
    a.plot(np.exp(xs), np.polyval(p, xs), color=line.get_color(), lw=1)
    a.plot([n_star], [np.polyval(p, math.log(n_star))], "*", color="red", ms=10)
    print(f"C = {C:.0e}: best size {n_star:,.0f} parameters, {C / (6 * n_star):,.0f} tokens, "
          f"{C / (6 * n_star) / n_star:.0f} tokens per parameter")
k, c0 = np.polyfit(np.log([c for c, _ in best]), np.log([n for _, n in best]), 1)
print(f"\nN* grows as C^{k:.2f}")
a.set_xscale("log"); a.xaxis.set_minor_formatter(plt.NullFormatter()); a.set_xlabel("parameters N"); a.set_ylabel("validation loss"); a.legend(frameon=False, fontsize=8)
cs = np.logspace(12.8, 14.2, 50)
b.plot([c for c, _ in best], [n for _, n in best], "*", color="red", ms=10)
b.plot(cs, np.exp(c0) * cs ** k, color="red", lw=1, label=f"fit: N* ∝ C^{k:.2f}")
b.plot(cs, np.sqrt(cs / 120), "--", color="0.5", lw=1, label="20 tokens per parameter")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("budget C (operations)"); b.set_ylabel("best size N*")
b.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
""")

    md(r"""
For each budget the loss is lowest at an intermediate size: smaller models have too few parameters to use their tokens, and larger ones read too few. The best size grows about as the square root of the budget, so a budget ten times larger should go to a model about three times larger, reading about three times as many tokens. These are single runs at three budgets; the notes' §9 compares the fit with Hoffmann et al.'s.
""")

    # ============================================================================ 5
    md(r"""
# 5. The base model

A base model continues documents (notes §10). A prompt works as a condition: it selects, among all the documents the model could continue, those that begin this way.

## Sums from a few examples

Our pretraining text included documents of sums, one per line, such as `237 + 682 = 919`. Give the base model a new sum in that format, and before it zero to four solved sums with operands of one to three digits, and read its greedy continuation (the most probable token at each step). The 200 pairs of three-digit numbers occur neither in pretraining nor in fine-tuning. Before running: does an example help?
""")

    code(r"""
nl = tok.encode("\n")[0]
rs = random.Random(1)                           # the notes' examples, drawn in the notes' order

def operand():
    digits = rs.randint(1, 3)
    return rs.randint(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1)

pairs = conversations["sum_pairs_3"]
t = time.time()
for k in range(5):
    correct = 0
    for a_, b_ in pairs:
        demo = [(operand(), operand()) for _ in range(k)]
        text = "".join(f"{x} + {y} = {x + y}\n" for x, y in demo) + f"{a_} + {b_} ="
        out = sample(base, [EOT] + tok.encode(text), 6, temperature=0, stop=(EOT, nl))
        words = tok.decode([i for i in out if i < EOT]).split()
        correct += bool(words) and words[0] == str(a_ + b_)
    print(f"{k} examples: {correct / len(pairs):4.0%} of {len(pairs)} sums right")
print(f"({time.time() - t:.0f} s); the last prompt:\n{text}")
""")

    md(r"""
The examples tell the model that it is reading a document of sums. (The notes report 49% without examples and 76% to 81% with them; the checkpoints here are stored in 16-bit floats, which changes a few greedy answers.) Learning a new task from examples in the prompt, without any change to the weights, is **in-context learning**; GPT-3 showed it at scale. At 3.5 million parameters our model continues the formats it has read: the notes find that examples do not help it with opposites or plurals, which it never read in this form.

## Copying an invented name

The simplest kind of in-context learning is copying: if a sequence occurred earlier in the context, predict that it continues the same way. Take a short story in which a little girl has a name the model has never read, such as *Fuzak*. Measure the log loss of the name's tokens, summed, at each of its four occurrences, averaged over 100 invented names:
""")

    code(r"""
TEMPLATE = ("Once upon a time, there was a little girl named {X}. {X} loved to play in the park. "
            "One day, {X} saw a big dog. {X} was happy.")

def invented_names(n=100, seed=1):
    r = random.Random(seed)
    names = []
    while len(names) < n:
        x = "".join(r.choice(s) for s in ["bdfgklmnprstvz", "aeiou", "bdfgklmnprstvz", "aeiou", "bdfgklmnprstvz"]).capitalize()
        if x not in names:
            names.append(x)
    return names

def occurrences(ids, name):
    # (start, length) of each occurrence of the name's tokens, with or without the space before it
    out = []
    for cand in (tok.encode(" " + name), tok.encode(name)):
        for i in range(len(ids)):
            if ids[i:i + len(cand)] == cand and not any(a <= i < a + n for a, n in out):
                out.append((i, len(cand)))
    return sorted(out)

def name_losses(model, names):
    per = [[], [], [], []]
    for X in names:
        ids = [EOT] + tok.encode(TEMPLATE.format(X=X))
        loss = token_losses(model, ids)                    # loss[t - 1] is the loss of token t
        for k, (i, n) in enumerate(occurrences(ids, X)[:4]):
            per[k].append(loss[i - 1:i - 1 + n].sum().item())
    return [float(np.mean(p)) for p in per]

names = invented_names()
print("names:", ", ".join(names[:8]), "...")
print("Fuzak in tokens:", tok.pieces(tok.encode(" Fuzak")))
for label, ns in [("100 invented names", names), ("familiar names", ["Lily", "Tom", "Sue", "Max", "Anna"])]:
    print(f"{label:19s} loss of the name at its four occurrences (nats):",
          "  ".join(f"{v:5.1f}" for v in name_losses(base, ns)))
""")

    md(r"""
The first *Fuzak* costs about 26 nats, the second about 5: the model has learned to copy from its context. The fourth costs more: at the start of the last sentence, after *One day, Fuzak saw a big dog.*, *She* or *The dog* would fit as well as the name. Two heads in two blocks do the copying (notes §10, *Induction heads*): a **previous-token head** writes into each position which token came before it, and a later **induction head**, at the current token, finds the position whose previous token matches it and reads what is there. Find them: keep the attention weights (`attention_solution` stores them), and for each head average over the names

1. the weight from the first token of the name's second occurrence to the token that followed the name's first token the first time (induction);
2. the weight from each position to the position just before it (previous token).
""")

    code(r"""
base_w = load_lm(CKPT / "week05_base.pt", attention=attention_solution)
base_w.keep_attention(True)
induction, previous = {}, {}
for X in names:
    ids = [EOT] + tok.encode(TEMPLATE.format(X=X))
    (p1, _), (p2, _) = occurrences(ids, X)[:2]
    with torch.no_grad():
        base_w(torch.tensor([ids]))
    for b, w in enumerate(base_w.attention_weights()):       # w: (1, H, T, T)
        for h in range(w.shape[1]):
            induction.setdefault((b + 1, h + 1), []).append(w[0, h, p2, p1 + 1].item())
            previous.setdefault((b + 1, h + 1), []).append(w[0, h, range(2, len(ids)), range(1, len(ids) - 1)].mean().item())
induction = {k: np.mean(v) for k, v in induction.items()}
previous = {k: np.mean(v) for k, v in previous.items()}
ih, ph = max(induction, key=induction.get), max(previous, key=previous.get)
print(f"induction head: block {ih[0]}, head {ih[1]}, {induction[ih]:.2f} of its attention on the token after the first occurrence")
print(f"previous-token head: block {ph[0]}, head {ph[1]}, {previous[ph]:.2f} of its attention on the previous token")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, scores, title in [(axes[0], induction, "induction score"), (axes[1], previous, "previous-token score")]:
    grid = np.array([[scores[(b, h)] for h in range(1, 7)] for b in range(1, 7)])
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6), range(1, 7)); ax.set_yticks(range(6), range(1, 7))
    ax.set_xlabel("head"); ax.set_ylabel("block"); ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.show()
""")

    md(r"""
## The running conversation

The running conversation of the notes: a system message, *You are a storyteller for young children.*, and a request, *Tell me a short story about a dog who finds a bone.* Give it to the base model as plain text, on two lines, and let it continue three times:
""")

    code(r"""
text = RUNNING[0]["content"] + "\n" + RUNNING[1]["content"] + "\n"
for seed in (1, 2, 3):
    out = sample(base, [EOT] + tok.encode(text), 120, temperature=0.8, seed=seed, stop=(EOT,))
    print(f"--- sample {seed}:", tok.decode([i for i in out if i != EOT]), "\n")
""")

    md(r"""
It writes the middle of a story, or copies the form of the request: a base model has no notion of a request, a turn, or an answer. Give it the conversation in the chat template of the notes' §11 instead, which Part 6 writes, and let it write after `<|assistant|>`. Before running: will it stop with `<|end|>`?
""")

    code(r"""
out = sample(base, prompt_ids(RUNNING, tok), 150, temperature=0.8, seed=1, stop=(END, EOT))
print(tok.decode([i for i in out if i < EOT]))
print("\nlast token:", tok.pieces(out[-1:]), " ended its turn:", out[-1] == END)

# the probability it gives <|end|> right after a finished story, in a training conversation
ids, _ = render_solution(conversations["train"][0]["messages"], tok)
p_end = math.exp(-token_losses(base, ids)[-1].item())
print(f"probability of <|end|> after a whole story: {p_end:.1e}")
""")

    md(r"""
Why does the base model give `<|end|>` so little probability, and write after `<|assistant|>` as if in the middle of a word?
""")

    if solution:
        md(ANSWERS.NEVER_ENDS)

    # ============================================================================ 6
    md(r"""
# 6. A chat model

A conversation is a list of messages, each with a role, `system`, `user`, or `assistant`, and its text. The **chat template** writes it as one sequence of tokens (notes §11): `<|endoftext|>`, as at the start of every pretraining document, then for each message its role's token, its text, and `<|end|>`. To get an answer, the program writes the conversation so far and `<|assistant|>`, and the model writes until it emits `<|end|>`.

**Supervised fine-tuning** trains the model on conversations whose assistant turns show what it should write (notes §13), with the loss on the assistant's tokens only: the system message and the user's turns are given, not the model's to write. The template's code therefore returns a **loss mask** with the tokens, 1 where the token is the assistant's to write: the text of each assistant message and the `<|end|>` that closes it. The role tokens are 0 in every case, since the program writes them.

Write `render(messages)`, which returns the token ids and the mask, two lists of the same length. Use `tok.encode`, `EOT`, `END`, and `ROLE` from the setup cell.
""")

    if solution:
        code(ANSWERS.RENDER)
    else:
        code(r"""
def render(messages):
    # messages: [{"role": "system" | "user" | "assistant", "content": text}, ...]
    # ids:  EOT, then for each message: ROLE[role], tok.encode(content), END
    # mask: 1 on the assistant's text and its END, 0 everywhere else (EOT, role tokens, system and user turns)
    raise NotImplementedError("write render")
    return ids, mask
""")

    md(r"""
Three checks, then a two-turn training conversation drawn with its mask: the tokens in the loss in blue, the others in gray, the special tokens outlined.
""")

    code(r"""
train_convs = conversations["train"]
two_turns = next(c for c in train_convs if sum(m["role"] == "assistant" for m in c["messages"]) == 2)
ids, mask = render(two_turns["messages"])

# check 1: the structure
assert len(ids) == len(mask) and ids[0] == EOT and mask[0] == 0
assert [i for i in ids if i >= EOT] == [EOT] + [x for m in two_turns["messages"] for x in (ROLE[m["role"]], END)]
print("check 1 passed:", len(ids), "tokens; the special tokens in order:", tok.pieces([i for i in ids if i >= EOT]))

# check 2: the mask covers exactly the assistant's text and its <|end|>
n_asst = sum(len(tok.encode(m["content"])) + 1 for m in two_turns["messages"] if m["role"] == "assistant")
assert sum(mask) == n_asst and all(mask[k] == 0 for k, i in enumerate(ids) if i in (SYS, USR, AST, EOT))
print(f"check 2 passed: {sum(mask)} of {len(ids)} tokens in the loss")

# check 3: the same as the library's on all 1,000 training conversations
assert all(render(c["messages"]) == render_solution(c["messages"], tok) for c in train_convs)
print("check 3 passed: render agrees with tinylm.finetune.render_solution on", len(train_convs), "conversations")

def draw(ids, mask, limit=None):
    spans = []
    for i, keep in list(zip(ids, mask))[:limit]:
        piece = tok.pieces([i])[0].replace("&", "&amp;").replace("<", "&lt;").replace("\n", "↵")
        style = "color:#1f5fbf" if keep else "color:#999"
        if i >= EOT:
            style += ";border:1px solid #999;border-radius:3px;padding:0 2px;font-weight:bold"
        spans.append(f'<span style="{style}">{piece}</span>')
    display(HTML('<div style="font-family:monospace;line-height:1.8;white-space:pre-wrap">' + "".join(spans) + "</div>"))

short = [{"role": m["role"], "content": " ".join(m["content"].split()[:25]) + (" ..." if len(m["content"].split()) > 25 else "")}
         for m in two_turns["messages"]]
draw(*render(short))
""")

    md(r"""
## Fine-tuning the base model

The fine-tuning conversations of the notes ask for stories, from a summary or a list of words, and for sums (notes §13). Fine-tune the base model on the first 1,000 of the 24,179, those within the context of 512 tokens, read three times: one conversation per row, batches of 16 conversations of similar length, AdamW at a peak of \(3\times10^{-4}\), a warmup over 5% of the steps, and a cosine decay. Each input position still reads everything before it; the targets outside the mask are set to \(-100\), which PyTorch's cross-entropy leaves out (`to_example`). The loss on the assistant's tokens of 80 held-out conversations is measured as training goes.
""")

    code(r"""
examples = [to_example(*render(c["messages"])) for c in train_convs]
examples = [e for e in examples if len(e[0]) <= 512]
heldout = [to_example(*render(c["messages"])) for c in conversations["heldout_stories"] + conversations["heldout_sums"]]
in_loss = sum(sum(t != -100 for t in y) for _, y in examples)
print(f"{len(examples)} conversations within the context, {sum(len(x) for x, _ in examples):,} tokens, "
      f"{in_loss:,} of them in the loss")

EPOCHS = 3                                       # 183 steps, a few minutes; 1 (61 steps) already teaches the format
mine = load_lm(CKPT / "week05_base.pt")
t = time.time()
sft_curve = finetune(mine, examples, epochs=EPOCHS, heldout=heldout, every=20)
print(f"{time.time() - t:.0f} s")
""")

    md(r"""
Now ask it for stories, 20 held-out requests at temperature 0.8: how many answers end with `<|end|>`, and how many of the words that the word-list requests ask for does it use? Then 20 sums, greedy. The notes measured the same model on 100 requests and 300 sums: before fine-tuning, the base model ended no answer, and after this run it ended 99 of 100 and used 21% of the requested words (`runs["finetuning"]`, the run `n1000`).
""")

    code(r"""
def measure(model, n=20):
    ended = found = asked = length = 0
    for i, c in enumerate(conversations["heldout_stories"][:n]):
        text, e = answer(model, tok, c["messages"][:-1], seed=i)
        ended += e
        length += len(tok.encode(text))
        if c["kind"] == "words":
            f, a = words_used(text, c["words"])
            found += f
            asked += a
    right = 0
    for c in conversations["heldout_sums"][:n]:
        text, _ = answer(model, tok, c["messages"][:-1], temperature=0, max_new=20)
        right += text.strip().rstrip(".").split("=")[-1].strip() == str(c["a"] + c["b"])
    print(f"ended {ended} of {n} answers with <|end|>, {length / n:.0f} tokens on average; "
          f"used {found} of {asked} requested words; {right} of {n} sums right ({text.strip()!r})")

t = time.time()
measure(mine)
print(f"({time.time() - t:.0f} s)\n")
text, ended = answer(mine, tok, RUNNING, seed=1)
print(text, "\n\nended its turn:", ended)
""")

    md(r"""
A thousand conversations teach the format: the model answers in the assistant's turn and stops. They teach little arithmetic: the first thousand conversations hold 206 sums, and the notes' model gets 1% of unseen three-digit sums right after them, against 70% after all 24,179 (5,000 conversations of sums).

## The notes' chat model

The notes' chat model was midtrained on stories and conversations (§12) and then fine-tuned on all 24,179 conversations (§13), in about half an hour on a processor. Ask it the running conversation (the notes' figure shows sample 1), then a second turn: the program appends the answer, the user's next message, and a new `<|assistant|>`.
""")

    code(r"""
chat = load_lm(CKPT / "week05_chat.pt")
for seed in (1, 2):
    text, ended = answer(chat, tok, RUNNING, seed=seed)
    print(f"--- sample {seed} (ended: {ended}):\n{text}\n")
text, _ = answer(chat, tok, RUNNING, temperature=0)
print(f"--- greedy:\n{text}\n")

conv = RUNNING + [{"role": "assistant", "content": answer(chat, tok, RUNNING, seed=1)[0]},
                  {"role": "user", "content": "Now tell me one about a cat."}]
text, ended = answer(chat, tok, conv, seed=1)
print(f"--- turn 2 (ended: {ended}):\n{text}")
""")

    md(r"""
Its second answer retells the first, Max and his tank, and is not about a cat. In six samples of this turn (seeds 1 to 6), none was about a cat, and five reused the first story's Max or his tank. Only one training conversation in fourteen has a second request, too few, it seems, to teach the model that the newest request decides the answer, and the first answer, in its context, is a story it can copy (Part 5).

Ask it questions outside the stories, and sums of three and of four digits. Before running: what will it answer to *What is the capital of France?*
""")

    code(r"""
for q in ["What is the capital of France?", "How many legs does a spider have?", "What is 12 + 7?", "Who wrote Hamlet?"]:
    text, _ = answer(chat, tok, [{"role": "user", "content": q}], seed=1, max_new=60)
    print(f"{q}\n   {text[:200]}\n")
for key in ["heldout_sums", "heldout_sums4"]:
    convs = conversations[key]
    right = sum(answer(chat, tok, c["messages"][:-1], temperature=0, max_new=20)[0].strip().rstrip(".").split("=")[-1].strip()
                == str(c["a"] + c["b"]) for c in convs)
    print(f"{key}: {right} of {len(convs)} right, e.g. {answer(chat, tok, convs[0]['messages'][:-1], temperature=0, max_new=20)[0]!r} "
          f"for {convs[0]['a']} + {convs[0]['b']}")
""")

    md(r"""
Fine-tuning taught the format of an answer, not knowledge the base model never read: asked about France, it tells a story, since its pretraining text holds stories and sums and nothing else. It adds three-digit numbers it has never seen, most of the time, and fails on four digits, a length no conversation contained. Nothing in its training told it which of its answers are good, only what the demonstrations wrote; rewards, in Weeks 6 and 7, add that (notes §14).

# At home

## LoRA, and the four new rows

**LoRA** freezes every weight matrix \(W\) and trains a low-rank change beside it, \(\mathbf xW+\mathbf xAB\), with \(A\) of \(d_{\text{in}}\times r\) random and \(B\) of \(r\times d_{\text{out}}\) zero, so that training starts from the base model (notes §7). `add_lora(model, 8)` wraps every attention and MLP matrix of every block. The notes found that LoRA alone does not learn to end its turn, and that training the rows of the token table for the four new tokens as well, `train_rows`, fixes it (§13). Repeat it on the 1,000 conversations, at LoRA's usual larger rate, \(3\times10^{-3}\), without weight decay (about five minutes for both). Before running: why would LoRA alone fail to end its turns?
""")

    code(r"""
lora_models = {}
for rows in (False, True):
    m = add_lora(load_lm(CKPT / "week05_base.pt"), 8)
    if rows:
        train_rows(m, [SYS, USR, AST, END])
    n = trained_parameters(m, rows=4 if rows else 0)
    label = "LoRA and the four rows" if rows else "LoRA alone"
    print(f"{label}: {n:,} trained numbers, {n / count_parameters(base):.1%} of the model's")
    finetune(m, examples, epochs=EPOCHS, peak=3e-3, weight_decay=0.0, log=None)
    lora_models[label] = merge_lora(m)
    measure(lora_models[label])
    print()
""")

    if solution:
        md(ANSWERS.LORA)

    md(r"""
## Summaries, taught and forgotten

Midtraining (notes §12) continued pretraining on a mixture of stories and conversations, among them requests to summarize a story, which fine-tuning's data never contain. Ask the midtrained model and the chat model for a summary of a held-out story:
""")

    code(r"""
mid = load_lm(CKPT / "week05_mid.pt")
request = conversations["heldout_summaries"][0]["messages"][:-1]
print(request[-1]["content"][:400], "...\n")
for name, model in [("midtrained", mid), ("chat model", chat)]:
    text, ended = answer(model, tok, request, seed=1, max_new=150)
    print(f"--- {name} (ended: {ended}):\n{text}\n")
print("--- the summary in the data:\n" + conversations["heldout_summaries"][0]["messages"][-1]["content"])
""")

    md(r"""
A task taught in midtraining survives fine-tuning only if fine-tuning's data keep some of it (notes §13).

## The system message

Some fine-tuning conversations have a sentence in their system message for each feature of the story, such as *In your stories, the characters talk to each other.* Does the chat model follow it? Count the stories that contain dialogue (a quotation mark) among the answers to 20 held-out requests, with and without that sentence. The notes did the same with 100 requests (§13).
""")

    code(r"""
talk = SYSTEM + " In your stories, the characters talk to each other."
for name, system in [("storyteller alone", SYSTEM), ("with the dialogue sentence", talk)]:
    n = 0
    for i, c in enumerate(conversations["heldout_stories"][:20]):
        user = [m for m in c["messages"] if m["role"] == "user"][:1]
        text, _ = answer(chat, tok, [{"role": "system", "content": system}] + user, seed=i)
        n += any(q in text for q in '"“”')
    print(f"{name:27s} {n} of 20 stories with dialogue")
""")

    md(r"""
With 20 stories in each condition, a difference of a few stories is within chance. The notes' answer, on 100: the sentence moved the model far less than it moved the training stories, because the request, written from the story's summary, already says most of what the story will be.

**Next week** is reinforcement learning: a model learns from rewards for what it writes, rather than from demonstrations of what to write. Week 6's notebook estimates the policy gradient (REINFORCE), with and without a baseline, on one small problem; Week 7's applies preferences and rewards to this week's chat model.
""")

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    targets = [(False, "week05_pretraining.ipynb")]
    if ANSWERS is not None:
        targets.append((True, "week05_pretraining_solution.ipynb"))
    for solution, name in targets:
        nbf.write(build(solution), folder / name)
        print("wrote", folder / name)
