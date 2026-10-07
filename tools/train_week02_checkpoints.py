"""Train the character-level models whose checkpoints the Week 2 notebook loads.

    python3 tools/train_week02_checkpoints.py lstm      # about 10 minutes on a laptop CPU
    python3 tools/train_week02_checkpoints.py rnn       # about 4 minutes
    python3 tools/train_week02_checkpoints.py word      # the word-level LSTM, about 20 minutes
    python3 tools/train_week02_checkpoints.py lstm 6000 512   # 512 units, 6,000 steps: about an hour
    python3 tools/train_week02_checkpoints.py word2vec  # skip-gram vectors for the hands-on notebook, about 4 minutes

Writes checkpoints/week02_char_<kind>.pt and checkpoints/week02_char_<kind>.json (learning curve,
sample, gradient by distance); a width other than 256 is added to the name (week02_char_lstm512.pt).
The fixed-window word model in checkpoints/week02_fixed_window.pt is trained by the notebook's own
training cell with TRAIN_FULL = True (about 10 minutes); the script for it is not needed separately.
"""
import json
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tinylm.data import download_tinystories, load_stories, split_stories, clean_chars
from tinylm.neural import CharRNN, char_alphabet, encode_chars, train_char_rnn, sample_chars, gradient_by_distance

ROOT = pathlib.Path(__file__).resolve().parent.parent
kind = sys.argv[1]

if kind == "word2vec":
    # The skip-gram run of Part 4 of the full notebook (ten passes), saved for the in-class notebook:
    # the words, their vectors before and after training, and the shifted-PMI check.
    import numpy as np
    from tinylm.data import tinystories_words
    from tinylm import embeddings
    data = tinystories_words(ROOT / "data")
    text = embeddings.content_words(data["train"])
    _, word_counts = embeddings.frequent_words(text, how_many=3000)
    w2v_words, _ = embeddings.frequent_words(text, min_count=5)
    model, info = embeddings.train_skipgram(text, w2v_words, word_counts, dim=100, passes=10)
    predicted, learned = embeddings.shifted_pmi_check(model, info, min_count=20)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save({"words": w2v_words, "after": model.v.weight.detach().clone(),
                "before": torch.tensor(info["before"]).half(),
                "pmi_check": torch.tensor(np.stack([predicted, learned]), dtype=torch.float)},
               ROOT / "checkpoints" / "week02_word2vec.pt")
    print(f"{len(w2v_words):,} words; shifted-PMI correlation {np.corrcoef(predicted, learned)[0, 1]:.2f}")
    sys.exit()

if kind == "word":
    from tinylm.data import tinystories_words
    from tinylm.neural import WordLSTM, word_index, story_tensors, story_perplexity, train_word_rnn
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    data = tinystories_words(ROOT / "data")
    vocab = data["vocab"]
    index = word_index(vocab)
    tx, ty = story_tensors(data["train"], index)
    vx, vy = story_tensors(data["val"], index)
    sx, sy = story_tensors(data["test"], index)
    torch.manual_seed(0)
    model = WordLSTM(len(vocab) + 1, len(vocab), embed_dim=64, hidden=128)
    curve = train_word_rnn(model, tx, ty, vx[:1000], vy[:1000], steps=steps)   # validation: first 1,000 stories
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "checkpoints" / "week02_word_lstm.pt")
    info = {"kind": "word lstm", "steps": steps, "params": sum(p.numel() for p in model.parameters()), "curve": curve,
            "val_ppl": story_perplexity(model, vx, vy), "test_ppl": story_perplexity(model, sx, sy)}
    json.dump(info, open(ROOT / "checkpoints" / "week02_word_lstm.json", "w"), indent=1)
    print(info)
    sys.exit()

steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 256

stories = load_stories(download_tinystories(ROOT / "data"))
train, val, test = split_stories(stories, seed=0)
train_text = [clean_chars(s) for s in train[:6000]]       # the same 6,000 stories as the character n-grams
test_text = [clean_chars(s) for s in test[:500]]
alphabet = char_alphabet(train_text)
train_stream = encode_chars(train_text, alphabet)
test_stream = encode_chars(test_text, alphabet)

torch.manual_seed(0)
model = CharRNN(len(alphabet), kind=kind, hidden=hidden)
curve = train_char_rnn(model, train_stream, test_stream, steps=steps)
name = f"week02_char_{kind}" + ("" if hidden == 256 else f"{hidden}")
(ROOT / "checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), ROOT / "checkpoints" / f"{name}.pt")
g = gradient_by_distance(model, train_stream)
info = {"kind": kind, "hidden": hidden, "steps": steps, "alphabet": "".join(alphabet),
        "params": sum(p.numel() for p in model.parameters()), "curve": curve,
        "test_bpc": curve[-1][2], "sample": sample_chars(model, alphabet),
        "grad_by_distance": [float(v) for v in g]}
json.dump(info, open(ROOT / "checkpoints" / f"{name}.json", "w"), indent=1)
print({k: v for k, v in info.items() if k not in ("curve", "grad_by_distance")})
