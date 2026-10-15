"""An English-to-French translator with attention (Week 3).

The model of the Lecture 3 notes (§§4-6): an LSTM encoder-decoder trained on Multi30k, with or without
the additive attention read of Bahdanau et al. (2015). Two trained checkpoints are in checkpoints/:

    week03_translate_attention.pt       with attention, 4.4 million parameters, 49.2 BLEU on the test set
    week03_translate_no_attention.pt    without attention, 4.0 million parameters, 19.5 BLEU

Both were trained by scripts/translate.py in the course's Lecture 3 folder (3,000 steps of 64 sentence
pairs, about 18 and 9 minutes on two processor cores). Nothing here trains a model.

The decoder step calls a function attend(query, keys, values, mask, W, v) for the attention read. The
Week 3 notebook asks you to write it; attend_solution below is one answer.

Data: Multi30k (Elliott et al., 2016), English image captions with French translations: 29,000 pairs for
training, 1,014 for validation, 1,000 for testing. The six files (about 2 MB) are downloaded into
data/multi30k/ on the first run. Text is lowercased and split into words and punctuation marks.
"""
import collections
import gzip
import os
import re
import urllib.request

import torch
import torch.nn as nn

MULTI30K_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/{name}.gz"
PAD, UNK, BOS, EOS = 0, 1, 2, 3          # ids of the four special tokens, in both vocabularies


# ----------------------------------------------------------------------------- data

def fetch_multi30k(folder):
    """Download the six Multi30k files into folder (once)."""
    os.makedirs(folder, exist_ok=True)
    for split in ["train", "val", "test_2016_flickr"]:
        for lang in ["en", "fr"]:
            path = os.path.join(folder, split + "." + lang)
            if not os.path.exists(path):
                print("downloading Multi30k", split + "." + lang, "...")
                with urllib.request.urlopen(MULTI30K_URL.format(name=split + "." + lang)) as response:
                    data = gzip.decompress(response.read())
                with open(path, "wb") as f:
                    f.write(data)


def tokenize(line):
    """Lowercase words and punctuation marks: 'A girl, smiling.' -> ['a', 'girl', ',', 'smiling', '.']"""
    return re.findall(r"\w+|[^\w\s]", line.lower())


def read_pairs(folder, split):
    """The (English words, French words) pairs of one split."""
    with open(os.path.join(folder, split + ".en"), encoding="utf-8") as f:
        english = [tokenize(line) for line in f]
    with open(os.path.join(folder, split + ".fr"), encoding="utf-8") as f:
        french = [tokenize(line) for line in f]
    return list(zip(english, french))


def vocabulary(sentences, min_count=2):
    """Words that occur at least min_count times, after the four special tokens; returns (word -> id, words)."""
    counts = collections.Counter(w for s in sentences for w in s)
    words = ["<pad>", "<unk>", "<s>", "</s>"] + sorted(w for w, c in counts.items() if c >= min_count)
    return {w: i for i, w in enumerate(words)}, words


def load_multi30k(folder):
    """The three splits and the two vocabularies (rebuilt from the training pairs, so always the same)."""
    fetch_multi30k(folder)
    data = {split: read_pairs(folder, split) for split in ["train", "val", "test_2016_flickr"]}
    data["test"] = data.pop("test_2016_flickr")
    data["src_vocab"], data["src_words"] = vocabulary([e for e, f in data["train"]])
    data["tgt_vocab"], data["tgt_words"] = vocabulary([f for e, f in data["train"]])
    return data


def encode(sentences, vocab, eos):
    """Lists of ids; unknown words become <unk>, and eos=True appends </s>."""
    return [[vocab.get(w, UNK) for w in s] + ([EOS] if eos else []) for s in sentences]


def pad(rows):
    """A (len(rows), longest) tensor of ids, padded with <pad> at the end."""
    longest = max(len(r) for r in rows)
    out = torch.full((len(rows), longest), PAD)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


# ----------------------------------------------------------------------------- the attention read

def attend_solution(query, keys, values, mask, W, v):
    """One additive attention read for a batch of B sentences of up to n source tokens.

    query  (B, H)     the decoder's previous state s_{t-1}
    keys   (B, n, H)  h_j U_a + b, computed once per sentence by the model
    values (B, n, H)  the encoder states h_j
    mask   (B, n)     True at real source tokens, False at padding
    W      nn.Linear(H, H, bias=False), the map W_a of the query
    v      nn.Linear(H, 1, bias=False), the vector v_a
    returns the context (B, H) and the weights (B, n)
    """
    scores = v(torch.tanh(W(query)[:, None, :] + keys)).squeeze(-1)    # e_tj = tanh(s W_a + h_j U_a + b) v_a, (B, n)
    scores = scores.masked_fill(~mask, float("-inf"))                    # padding gets weight exp(-inf) = 0
    weights = torch.softmax(scores, dim=-1)                              # alpha_tj, over the source positions, (B, n)
    context = (weights[:, :, None] * values).sum(dim=1)                  # c_t = sum_j alpha_tj h_j, (B, H)
    return context, weights


# ----------------------------------------------------------------------------- the model

class Translator(nn.Module):
    """The LSTM encoder-decoder of the notes, with or without attention.

    Encoder: word vectors (E = 128 numbers) read by a bidirectional LSTM, H/2 = 128 units each way; its
    final hidden states and cells, both directions side by side, start the decoder.
    Decoder: an LSTM cell of H = 256 units. With attention, at every step it reads the context c_t from
    its previous state (the query), takes [E(y_{t-1}); c_t] as its input, and predicts the next word from
    one tanh layer over [s_t; c_t; E(y_{t-1})]. Without attention, the input is E(y_{t-1}) and the output
    layer reads [s_t; E(y_{t-1})].
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, attention, attend=attend_solution, E=128, H=256, dropout=0.3):
        super().__init__()
        self.attention = attention
        self.attend = attend                     # the function that does the read (see attend_solution)
        self.src_emb = nn.Embedding(src_vocab_size, E, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, E)
        self.encoder = nn.LSTM(E, H // 2, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(E + (H if attention else 0), H)
        if attention:
            self.W = nn.Linear(H, H, bias=False)     # W_a, applied to the query
            self.U = nn.Linear(H, H)                 # U_a (and a bias), applied once to every encoder state
            self.v = nn.Linear(H, 1, bias=False)     # v_a
        self.hidden = nn.Linear(H + (H if attention else 0) + E, H)
        self.out = nn.Linear(H, tgt_vocab_size)
        self.drop = nn.Dropout(dropout)

    def encode(self, src):
        """src (B, n) ids. Returns the encoder states hs (B, n, H), the decoder's starting state, and the mask."""
        lengths = (src != PAD).sum(1)
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.src_emb(src)), lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, (h, c) = self.encoder(packed)
        hs, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.shape[1])
        state = (torch.cat([h[0], h[1]], -1), torch.cat([c[0], c[1]], -1))   # both directions, side by side
        return hs, state, src != PAD

    def keys(self, hs):
        """h_j U_a + b for every source position, computed once per sentence."""
        if self.attention:
            return self.U(hs)
        return None

    def step(self, y_prev, state, hs, mask, keys):
        """One decoder step. Returns the scores of the next word (B, vocabulary), the new state, the weights."""
        emb = self.drop(self.tgt_emb(y_prev))
        weights = None
        if self.attention:
            context, weights = self.attend(state[0], keys, hs, mask, self.W, self.v)   # the query is s_{t-1}
            h, c = self.decoder(torch.cat([emb, context], -1), state)
            features = torch.cat([h, context, emb], -1)
        else:
            h, c = self.decoder(emb, state)
            features = torch.cat([h, emb], -1)
        return self.out(self.drop(torch.tanh(self.hidden(features)))), (h, c), weights

    def forward(self, src, tgt):
        """Teacher forcing: the decoder reads BOS and the reference. Returns the scores (B, m, vocabulary)
        and the attention weights (B, m, n), or None without attention."""
        hs, state, mask = self.encode(src)
        keys = self.keys(hs)
        prev = torch.full((src.shape[0],), BOS)
        scores, weights = [], []
        for t in range(tgt.shape[1]):
            s, state, a = self.step(prev, state, hs, mask, keys)
            scores.append(s)
            weights.append(a)
            prev = tgt[:, t]
        if self.attention:
            return torch.stack(scores, 1), torch.stack(weights, 1)
        return torch.stack(scores, 1), None


def load_translator(path, data, attention, attend=attend_solution):
    """A trained translator from a checkpoint (a state dictionary), in evaluation mode (no dropout)."""
    model = Translator(len(data["src_words"]), len(data["tgt_words"]), attention, attend)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# ----------------------------------------------------------------------------- using it

def greedy_ids(model, src, max_len):
    """Greedy decoding of a batch: at every step the most probable word, fed back as the next input."""
    with torch.no_grad():
        hs, state, mask = model.encode(src)
        keys = model.keys(hs)
        prev = torch.full((src.shape[0],), BOS)
        chosen = []
        for t in range(max_len):
            scores, state, _ = model.step(prev, state, hs, mask, keys)
            prev = scores.argmax(-1)
            chosen.append(prev)
    return torch.stack(chosen, 1)


def ids_to_words(row, words):
    """French words up to the first </s>."""
    out = []
    for k in row:
        if k == EOS:
            break
        out.append(words[k])
    return out


def translate(model, data, english_sentences):
    """Greedy translations of a list of English sentences (each a list of words), as lists of words."""
    src = pad(encode(english_sentences, data["src_vocab"], False))
    chosen = greedy_ids(model, src, src.shape[1] * 2 + 10)
    return [ids_to_words(row, data["tgt_words"]) for row in chosen.tolist()]


def corpus_bleu(hypotheses, references):
    """BLEU (0-100) of lists of words against one reference each, on the tokenized text, as in the notes."""
    import sacrebleu                             # pip install sacrebleu
    hyps = [" ".join(h) for h in hypotheses]
    refs = [" ".join(r) for r in references]
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none", force=True).score   # the text is tokenized on purpose


def attention_matrix(model, data, english, french):
    """The weights (French words x English words) of the model reading the reference with teacher forcing."""
    src = pad(encode([english], data["src_vocab"], False))
    tgt = pad(encode([french], data["tgt_vocab"], False))
    with torch.no_grad():
        _, weights = model(src, tgt)
    return weights[0]


def greedy_steps(model, data, english, max_len=40):
    """Greedy decoding of one sentence, step by step: the chosen word, its probability, and the weights."""
    src = pad(encode([english], data["src_vocab"], False))
    steps = []
    with torch.no_grad():
        hs, state, mask = model.encode(src)
        keys = model.keys(hs)
        prev = torch.full((1,), BOS)
        for t in range(max_len):
            scores, state, weights = model.step(prev, state, hs, mask, keys)
            probs = torch.softmax(scores[0], -1)
            k = int(probs.argmax())
            steps.append({"word": data["tgt_words"][k], "prob": float(probs[k]), "probs": probs,
                          "weights": None if weights is None else weights[0]})
            if k == EOS:
                break
            prev = torch.tensor([k])
    return steps
