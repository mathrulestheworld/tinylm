"""Write the Week 3 notebooks: one attention read, inside a trained English-to-French translator.

    python3 tools/make_week03_notebook.py

notebooks/week03_attention.ipynb has an empty attend() for students to write.
With the instructor's answers file (tools/week03_answers.py, not in the public repository) the script also
writes notebooks/week03_attention_solution.ipynb, the same notebook with the answers filled in; it is saved
with its outputs after running
    jupyter nbconvert --to notebook --execute --inplace notebooks/week03_attention_solution.ipynb

The notebook follows lectures/03-sequence-modeling-attention/attention-exercise.md in the course repository.
Nothing is trained: the two translators of the Lecture 3 notes are loaded from checkpoints/.
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import week03_answers as ANSWERS        # the instructor's answers; not in the public repository
except ImportError:
    ANSWERS = None

NOTES = "https://mathrulestheworld.github.io/genai-first-principles/materials/week-03/lecture-03-notes.pdf"


def build(solution):
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md(r"""
# Week 3: one attention read, inside a trained translator

*Generative AI from First Principles*, Lecture 3 · [course page](https://mathrulestheworld.github.io/genai-first-principles/) · [notes](%s)

The translator of the Lecture 3 notes (§§4–6) reads an English image caption with a bidirectional LSTM and writes French with an LSTM decoder. Before writing each word, the decoder makes one **attention read**: a weighted average of the encoder's states, with weights computed from the decoder's state. You write that read, check it, put it inside the trained model, and look at what it does.

| Part | In class | What we do |
|---|---|---|
| 1. Write `attend` | 3 minutes | the scores, the softmax over source positions, the context |
| 2. Check it | 4 minutes | shapes, normalization, masking, the worked example of §6, agreement with the trained model, BLEU |
| 3. Compare | 3 minutes | the same translator with and without attention, overall and by caption length |
| 4. Inspect | 3 minutes | one reordering (*tenue de karaté*) and one failure (*faisant*) |
| At home | | two interventions; the reversal task of §5 |

Nothing is trained here: both translators were trained for the notes and are loaded from `checkpoints/` (4.4 and 4.0 million parameters). The data are [Multi30k](https://aclanthology.org/W16-3210/) (Elliott et al., 2016), English captions of Flickr photos with French translations, downloaded into `data/multi30k/` on the first run (about 2 MB). Every cell runs in a few seconds on a laptop.
""" % NOTES)

    code(r"""
import sys, json, time, pathlib
ROOT = pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT))                 # use the repository's tinylm without installing it

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tinylm.translator import (load_multi30k, load_translator, encode, pad, translate, corpus_bleu,
                               attention_matrix, greedy_steps, PAD)

torch.set_num_threads(2)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
CKPT = ROOT / "checkpoints"
data = load_multi30k(ROOT / "data" / "multi30k")
test_en = [e for e, f in data["test"]]
test_fr = [f for e, f in data["test"]]
print(f"{len(data['train']):,} training pairs, {len(data['test']):,} test pairs; "
      f"{len(data['src_words']):,} English and {len(data['tgt_words']):,} French words in the vocabularies")
print("test pair 2:", " ".join(test_en[2]), " ->  ", " ".join(test_fr[2]))
""")

    # ============================================================================ 1
    md(r"""
# 1. Write `attend`

At step \(t\) the decoder's previous state \(\mathbf s_{t-1}\) is the **query**. For every source position \(j\) the model scores the encoder state \(\mathbf h_j\), turns the scores into weights with a softmax over the source positions, and averages (notes §6, "The read"):

\[
e_{tj}=\tanh\bigl(\mathbf s_{t-1}W_a+\mathbf h_jU_a+\mathbf b\bigr)\,\mathbf v_a^\top,\qquad
\alpha_{tj}=\frac{\exp e_{tj}}{\sum_{k}\exp e_{tk}},\qquad
\mathbf c_t=\sum_j\alpha_{tj}\,\mathbf h_j .
\]

The model computes the **keys** \(\mathbf h_jU_a+\mathbf b\) once per sentence and passes them in. A batch holds \(B\) sentences padded to \(n\) tokens; `mask` is `True` at real tokens, and padding must get weight exactly zero. In PyTorch, `W(x)` computes \(x W_a\) for the query, and `v(x)` computes \(x\,\mathbf v_a^\top\).

Write the shape of every intermediate tensor next to its line.
""")

    if solution:
        code(ANSWERS.ATTEND)
    else:
        code(r"""
def attend(query, keys, values, mask, W, v):
    # query (B, H), keys (B, n, H), values (B, n, H), mask (B, n)
    # 1. scores e_tj, shape (B, n)
    # 2. padding to -inf, then a softmax over the source positions: weights, shape (B, n)
    # 3. the context, the weighted sum of the values, shape (B, H)
    raise NotImplementedError("write attend")
    return context, weights
""")

    # ============================================================================ 2
    md(r"""
# 2. Check it

Six checks, in order; each must pass before the next. The trained translator is loaded with **your** `attend` inside its decoder (run this cell again whenever you change `attend`).
""")

    code(r"""
model = load_translator(CKPT / "week03_translate_attention.pt", data, attention=True, attend=attend)


def first_read(model, sentences):
    # the inputs of the decoder's first attention read, for a batch of English sentences
    src = pad(encode(sentences, data["src_vocab"], False))
    with torch.no_grad():
        hs, state, mask = model.encode(src)
        keys = model.keys(hs)
    return state[0], keys, hs, mask


batch = [e for e in test_en if len(e) <= 9][:4]           # four short test captions
query, keys, values, mask = first_read(model, batch)
with torch.no_grad():
    context, weights = attend(query, keys, values, mask, model.W, model.v)

# check 1: shapes
B, n = mask.shape
assert context.shape == (B, 256) and weights.shape == (B, n), (context.shape, weights.shape)
print(f"check 1 passed: context {tuple(context.shape)}, weights {tuple(weights.shape)}")

# check 2: every row of weights is a probability distribution over the source positions
assert (weights >= 0).all() and torch.allclose(weights.sum(-1), torch.ones(B), atol=1e-5)
print("check 2 passed: every row of weights is nonnegative and sums to 1")
""")

    code(r"""
# check 3: padding gets weight 0, and extra padding changes nothing
src = pad(encode(batch, data["src_vocab"], False))
longer = torch.cat([src, torch.full((B, 3), PAD)], 1)     # three more padding columns
with torch.no_grad():
    hs2, state2, mask2 = model.encode(longer)
    context2, weights2 = attend(state2[0], model.keys(hs2), hs2, mask2, model.W, model.v)
assert (weights[~mask] == 0).all() and (weights2[:, n:] == 0).all()
assert torch.allclose(weights2[:, :n], weights, atol=1e-6) and torch.allclose(context2, context, atol=1e-6)
print("check 3 passed: padding has weight 0; three more padding positions change neither weights nor context")
""")

    md(r"""
**Check 4, the worked example of §6** ("One read by hand", Figure 15). Three states of two numbers, \(\mathbf s_{t-1}=(1,2)\), and no bias. Before running: which state gets the most weight? (A plain dot product \(\mathbf s_{t-1}\mathbf h_j^\top\) gives 2, 2, 3.)

`nn.Linear` computes \(x A^\top\) for its weight \(A\), so to compute \(\mathbf s_{t-1}W_a\) its weight is \(W_a^\top\).
""")

    code(r"""
W_a = torch.tensor([[1.0, 0.5], [0.0, -0.75]])
U_a = torch.tensor([[0.5, 0.5], [-1.0, 1.0]])
v_a = torch.tensor([2.0, 1.0])
s = torch.tensor([[1.0, 2.0]])                                    # one query, B = 1
h = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])          # three states, n = 3
W = nn.Linear(2, 2, bias=False)
v = nn.Linear(2, 1, bias=False)
with torch.no_grad():
    W.weight.copy_(W_a.T)                                         # so that W(s) = s W_a
    v.weight.copy_(v_a[None, :])                                  # so that v(x) = x v_a^T
    c, a = attend(s, h @ U_a, h, torch.ones(1, 3, dtype=torch.bool), W, v)
print("weights", [round(x, 3) for x in a[0].tolist()], "  context", [round(x, 3) for x in c[0].tolist()])
assert torch.allclose(a[0], torch.tensor([0.579, 0.084, 0.337]), atol=1e-3)
assert torch.allclose(c[0], torch.tensor([1.495, 0.421]), atol=1e-3)
print("check 4 passed: the numbers of the notes")
""")

    code(r"""
# check 5: inside the trained decoder, your weights match the ones stored when the model was trained
reference = json.load(open(CKPT / "week03_reference_weights.json"))["examples"]
pair = reference[2]
A = attention_matrix(model, data, pair["en"], pair["fr"])        # French words x English words, teacher forcing
gap = (A - torch.tensor(pair["attention"])).abs().max().item()
assert gap < 1e-4, gap
row = A[pair["fr"].index("tenue")]
print(f"check 5 passed: largest difference {gap:.1e};  the row of 'tenue': "
      f"karate {row[pair['en'].index('karate')]:.2f}, uniform {row[pair['en'].index('uniform')]:.2f}")

# check 6: the model still translates
start = time.time()
hyps = translate(model, data, test_en)                           # greedy decoding of the 1,000 test captions
bleu = corpus_bleu(hyps, test_fr)
assert abs(bleu - 49.2) < 0.3, bleu
print(f"check 6 passed: BLEU {bleu:.1f} on the test set ({time.time() - start:.0f} s)")
""")

    # ============================================================================ 3
    md(r"""
# 3. Compare: the same translator without attention

The second checkpoint has the same encoder, data, sizes, training steps, and decoding, but no attention: its decoder starts from the encoder's final states and never reads the source again (notes §§4–5). Before running: how much lower will its BLEU be, and on which captions?
""")

    code(r"""
plain = load_translator(CKPT / "week03_translate_no_attention.pt", data, attention=False)
plain_hyps = translate(plain, data, test_en)
print(f"BLEU with attention {bleu:.1f}, without {corpus_bleu(plain_hyps, test_fr):.1f}\n")

groups = [(1, 9), (10, 11), (12, 13), (14, 16), (17, 20), (21, 99)]       # the groups of Figure 17
rows = []
for lo, hi in groups:
    idx = [i for i, e in enumerate(test_en) if lo <= len(e) <= hi]
    with_attn = corpus_bleu([hyps[i] for i in idx], [test_fr[i] for i in idx])
    without = corpus_bleu([plain_hyps[i] for i in idx], [test_fr[i] for i in idx])
    rows.append((f"{lo}-{hi}" if hi < 99 else f"{lo}+", len(idx), with_attn, without))
    print(f"{rows[-1][0]:>6s} tokens  {len(idx):4d} captions   with attention {with_attn:5.1f}   without {without:5.1f}")

fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", color="#D62728", label="with attention")
ax.plot([r[0] for r in rows], [r[3] for r in rows], "o-", color="#596273", label="without attention")
ax.set_xlabel("length of the English caption (tokens)"); ax.set_ylabel("BLEU"); ax.set_ylim(0, 70)
ax.legend(frameon=False); plt.show()
""")

    md(r"""
**Write two sentences.** (1) What differs between the two models besides "attention"? (2) What does the breakdown by length show, and what does it not show?
""")
    if solution:
        md(ANSWERS.LENGTHS)

    # ============================================================================ 4
    md(r"""
# 4. Inspect one reordering and one failure

**The reordering.** The weights of the attention model reading test pair 2 with teacher forcing (the decoder is fed the reference, as in training). French writes *tenue de karaté*, "uniform of karate": look at the outlined row.
""")

    code(r"""
en, fr = pair["en"], pair["fr"]
fig, ax = plt.subplots(figsize=(5.6, 5.4))
ax.imshow(A, cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(len(en)), en, rotation=90); ax.set_yticks(range(len(fr)), fr)
r = fr.index("tenue")
ax.add_patch(plt.Rectangle((-0.5, r - 0.5), len(en), 1, fill=False, ec="black", lw=1.5))
ax.set_xlabel("English (read)"); ax.set_ylabel("French (written)")
plt.show()
""")

    md(r"""
**The failure.** Now let the model translate the caption itself (greedy decoding). For each word it writes: its probability, and the English words with weight above 0.1.
""")

    code(r"""
steps = greedy_steps(model, data, en)
for t, step in enumerate(steps):
    read = ", ".join(f"{en[j]} {step['weights'][j]:.2f}" for j in range(len(en)) if step["weights"][j] > 0.1)
    print(f"{t:2d}  {step['word']:>8s}  p = {step['prob']:.2f}   reads: {read}")
print("\nreference:", " ".join(fr))
""")

    md(r"""
At step 6 the model writes *faisant* ("doing") where the reference has *brisant* ("breaking"), which is in its vocabulary. **Three to five sentences:** Did the decoder read the right source word? Does a large weight on *breaking* show that the model translated *breaking*? What else does the prediction depend on? What test would show whether the word *breaking* caused it?
""")
    if solution:
        md(ANSWERS.FAISANT)

    # ============================================================================ at home
    md(r"""
# At home

**Change the source.** Replace *breaking* by another verb in the vocabulary. Does the French verb change, and does the rest stay the same?
""")

    code(r"""
for verb in ["breaking", "holding", "hitting"]:
    sentence = [verb if w == "breaking" else w for w in en]
    print(f"{verb:>9s}:", " ".join(translate(model, data, [sentence])[0]))
""")

    md(r"""
**Change the weights.** At the step that writes *faisant*, set the largest weight to zero, renormalize the others, and recompute that step's prediction. The earlier steps are unchanged, so the model sees the same prefix.
""")

    code(r"""
def with_edited_read(step_to_edit):
    # an attend that removes the largest weight at one decoder step (counted from 0)
    calls = [0]

    def edited(query, keys, values, mask, W, v):
        context, weights = attend(query, keys, values, mask, W, v)
        if calls[0] == step_to_edit:
            weights = weights.clone()
            weights[0, weights[0].argmax()] = 0.0
            weights = weights / weights.sum(-1, keepdim=True)
            context = (weights[:, :, None] * values).sum(dim=1)
        calls[0] += 1
        return context, weights
    return edited


t = [s["word"] for s in steps].index("faisant")
edited = load_translator(CKPT / "week03_translate_attention.pt", data, attention=True, attend=with_edited_read(t))
after = greedy_steps(edited, data, en)[t]
k = data["tgt_vocab"]["faisant"]
print(f"step {t}: p(faisant) {steps[t]['prob']:.3f} -> {after['probs'][k]:.3f};  the model now writes '{after['word']}' "
      f"(p = {after['prob']:.2f}), reading", ", ".join(f"{en[j]} {after['weights'][j]:.2f}"
                                                for j in range(len(en)) if after["weights"][j] > 0.1))
""")

    md(r"""
**The reversal task of §5.** A network that writes a string of letters backwards, trained on strings of up to 40 letters, with and without attention (Figures 12 and 16 of the notes, redrawn from the stored results; nothing runs). Why might some weight fall on the position *after* the letter being copied? (Hint: the encoder is bidirectional.)
""")

    code(r"""
rev = json.load(open(CKPT / "week03_reversal.json"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
for name, color in [("attention", "#D62728"), ("no attention", "#596273")]:
    rows = rev["models"][name]["rows"]
    ax1.plot([r["n"] for r in rows], [100 * r["letters"] for r in rows], "o-", color=color, label=name)
ax1.axvspan(rev["max_train_length"], 62, color="#EEF0F5")
ax1.set_xlabel("length of the string (letters)"); ax1.set_ylabel("letters written correctly (%)"); ax1.legend(frameon=False)
letters = [chr(ord("a") + k) for k in rev["example"]["source"]]
out = [chr(ord("a") + k) for k in rev["example"]["output"][:-1]]          # the last output is the end symbol
ax2.imshow(torch.tensor(rev["example"]["weights"])[:len(out)], cmap="Reds", vmin=0, vmax=1)
ax2.set_xticks(range(len(letters)), letters); ax2.set_yticks(range(len(out)), out)
ax2.set_xlabel("source letter (read)"); ax2.set_ylabel("output letter (written)")
plt.tight_layout(); plt.show()
""")
    if solution:
        md(ANSWERS.REVERSAL)

    md(r"""
**Next week** the read becomes a matrix product over all positions at once, \(\operatorname{softmax}(QK^\top/\sqrt{d_k})V\), and a Transformer is built from it.
""")

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    targets = [(False, "week03_attention.ipynb")]
    if ANSWERS is not None:
        targets.append((True, "week03_attention_solution.ipynb"))
    for solution, name in targets:
        nbf.write(build(solution), folder / name)
        print("wrote", folder / name)
