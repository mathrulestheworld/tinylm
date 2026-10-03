"""Neural language models in PyTorch (Week 2).

FixedWindowLM   the neural probabilistic language model of Bengio et al. (2003): the next word
                from the previous few words, through learned word vectors and one hidden layer
CharRNN         a character-level recurrent network, plain (tanh) or LSTM

From Week 2 on we use PyTorch's tensors and autograd; they do what the Week 1 engine does,
on larger arrays and faster. The models below use only basic layers (Embedding, Linear, RNN,
LSTM), so every computation is visible.
"""
import math
import time

import torch
import torch.nn as nn

from .data import END


# ----------------------------------------------------------------------------- fixed-window model

class FixedWindowLM(nn.Module):
    """q(next word | previous `context` words) = softmax(b + U tanh(d + H [C(x_1); ...; C(x_k)])).

    C is the word table: one learned vector of size embed_dim per input symbol (the vocabulary
    plus BOS). The vectors of the context words are concatenated, passed through one tanh layer
    (H, d) and turned into scores for every word of the vocabulary (U, b).
    """

    def __init__(self, n_inputs, vocab_size, context=4, embed_dim=64, hidden=128):
        super().__init__()
        self.context = context
        self.C = nn.Embedding(n_inputs, embed_dim)
        self.H = nn.Linear(context * embed_dim, hidden)
        self.U = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        """x: (batch, context) word ids. Returns (batch, vocab_size) scores (logits)."""
        e = self.C(x).flatten(1)                 # (batch, context * embed_dim): the vectors side by side
        h = torch.tanh(self.H(e))                # (batch, hidden)
        return self.U(h)                         # softmax of these scores = next-word distribution


def word_index(vocab):
    """Ids for the model's input symbols: the vocabulary in its order, then BOS last."""
    index = {}
    for i, w in enumerate(vocab):
        index[w] = i
    index["<s>"] = len(vocab)
    return index


def make_windows(sequences, index, context):
    """All (context words, next word) training examples, as two tensors of ids.

    Each sequence is padded with `context` BOS symbols at the start and ends with END, so its
    last example predicts END.
    """
    bos = index["<s>"]
    xs = []
    ys = []
    for seq in sequences:
        ids = [bos] * context
        for w in seq:
            ids.append(index[w])
        ids.append(index[END])
        for i in range(context, len(ids)):
            xs.append(ids[i - context:i])
            ys.append(ids[i])
    return torch.tensor(xs), torch.tensor(ys)


def average_loss(model, X, Y, batch_size=8192):
    """Average cross-entropy (nats per predicted word) of model on the examples (X, Y)."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            total += loss_fn(model(X[i:i + batch_size]), Y[i:i + batch_size]).item()
    model.train()
    return total / len(X)


def train_window_lm(model, X, Y, X_val, Y_val, epochs=2, batch_size=512, lr=2e-3,
                    eval_every=2000, max_steps=None):
    """Minibatch training with Adam on the cross-entropy of the next word.

    Each epoch visits the training examples in a new random order. Every eval_every steps the
    validation loss is computed; the parameters with the best validation loss are kept. The
    learning rate is halved after each epoch. Returns a list of (step, validation perplexity).
    For a reproducible run, call torch.manual_seed before creating the model.
    """
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_loss = math.inf
    best_state = None
    step = 0
    start = time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch = order[i:i + batch_size]
            loss = loss_fn(model(X[batch]), Y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step == max_steps:
                val = average_loss(model, X_val, Y_val)
                history.append((step, math.exp(val)))
                print(f"step {step:6d}   validation perplexity {math.exp(val):7.2f}   ({time.time() - start:.0f} s)")
                if val < best_loss:
                    best_loss = val
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
        for group in opt.param_groups:
            group["lr"] = group["lr"] / 2
    val = average_loss(model, X_val, Y_val)              # the parameters at the end of training
    if val < best_loss:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def next_word_distribution(model, index, vocab, prompt_words, top=5):
    """The model's `top` most probable next words after prompt_words, with their probabilities."""
    ids = [index["<s>"]] * model.context
    for w in prompt_words:
        ids.append(index[w])
    x = torch.tensor([ids[-model.context:]])
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x)[0], -1)
    model.train()
    values, positions = torch.topk(p, top)
    result = []
    for v, j in zip(values.tolist(), positions.tolist()):
        result.append((vocab[j], v))
    return result


def sample_words(model, index, vocab, prompt_words=(), temperature=1.0, max_words=60,
                 seed=0, exclude=("<unk>",)):
    """Generate words one at a time from the model, starting after prompt_words.

    temperature < 1 sharpens the distribution (each probability raised to the power
    1/temperature, then renormalized), temperature > 1 flattens it.
    """
    gen = torch.Generator().manual_seed(seed)
    ids = [index["<s>"]] * model.context
    for w in prompt_words:
        ids.append(index[w])
    out = list(prompt_words)
    model.eval()
    with torch.no_grad():
        for _ in range(max_words):
            logits = model(torch.tensor([ids[-model.context:]]))[0] / temperature
            for w in exclude:
                logits[index[w]] = -math.inf
            j = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).item()
            if vocab[j] == END:
                break
            out.append(vocab[j])
            ids.append(j)
    model.train()
    return out


def detokenize(tokens):
    """Join word tokens into readable text (no space before punctuation)."""
    text = ""
    for t in tokens:
        if t in ".,!?;:" or text == "":
            text += t
        else:
            text += " " + t
    return text


# ----------------------------------------------------------------------------- character RNNs

STORY_END = "$"     # marks the end of each story in the character stream (not a story character)


class CharRNN(nn.Module):
    """A character-level recurrent language model.

    Each character is mapped to a vector (emb), a recurrent layer (rnn) updates a hidden state
    once per character, and a linear layer (out) turns the state into scores for the next
    character. kind = "rnn" is the plain tanh recurrence h_t = tanh(W h_{t-1} + V e_t + d);
    kind = "lstm" adds the gated cell of Hochreiter and Schmidhuber (1997).
    """

    def __init__(self, n_chars, kind="lstm", hidden=256, embed_dim=32):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(n_chars, embed_dim)
        if kind == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_chars)

    def forward(self, x, state=None):
        """x: (batch, time) character ids. Returns scores (batch, time, n_chars) and the final state."""
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state


def char_alphabet(texts):
    """The characters of the texts, sorted, plus STORY_END."""
    chars = set()
    for t in texts:
        chars.update(t)
    chars.add(STORY_END)
    return sorted(chars)


def encode_chars(texts, alphabet):
    """One long tensor of character ids: the texts one after another, each followed by STORY_END."""
    index = {}
    for i, c in enumerate(alphabet):
        index[c] = i
    ids = []
    for t in texts:
        for c in t + STORY_END:
            ids.append(index[c])
    return torch.tensor(ids)


def char_bits_per_char(model, stream, chunk=4096):
    """Bits per character on a stream, read once from start to end with the state carried along."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total = 0.0
    state = None
    x = stream[:-1]
    y = stream[1:]
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            logits, state = model(x[None, i:i + chunk], state)
            total += loss_fn(logits[0], y[i:i + chunk]).item()
    model.train()
    return total / len(y) / math.log(2)


def train_char_rnn(model, train_stream, test_stream, steps=3000, length=128, batch_size=64,
                   lr=3e-3, clip=1.0, eval_every=250):
    """Truncated backpropagation through time on random chunks of `length` characters.

    Each step takes batch_size random chunks, predicts every next character, and takes one Adam
    step; gradients are clipped to norm `clip`. The learning rate drops to 1e-3 after 70% of the
    steps. Returns (step, characters seen, held-out bits per character) every eval_every steps.
    """
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    start = time.time()
    for step in range(1, steps + 1):
        starts = torch.randint(0, len(train_stream) - length - 1, (batch_size,))
        x = torch.stack([train_stream[s:s + length] for s in starts])
        y = torch.stack([train_stream[s + 1:s + length + 1] for s in starts])
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if step % eval_every == 0:
            bpc = char_bits_per_char(model, test_stream)
            curve.append((step, step * batch_size * length, bpc))
            print(f"step {step:5d}   held-out bits per character {bpc:.3f}   ({time.time() - start:.0f} s)")
        if step == int(steps * 0.7):
            for group in opt.param_groups:
                group["lr"] = 1e-3
    return curve


def sample_chars(model, alphabet, max_chars=400, temperature=1.0, seed=1, start=STORY_END):
    """Generate characters one at a time, starting from `start`, until STORY_END."""
    gen = torch.Generator().manual_seed(seed)
    index = {}
    for i, c in enumerate(alphabet):
        index[c] = i
    x = torch.tensor([[index[start]]])
    state = None
    out = ""
    model.eval()
    with torch.no_grad():
        for _ in range(max_chars):
            logits, state = model(x, state)
            p = torch.softmax(logits[0, -1] / temperature, -1)
            j = torch.multinomial(p, 1, generator=gen).item()
            if alphabet[j] == STORY_END:
                break
            out += alphabet[j]
            x = torch.tensor([[j]])
    model.train()
    return out


def gradient_by_distance(model, stream, length=256, batch_size=32, seed=0):
    """How much the loss of the last prediction in a window depends on each earlier input.

    Returns g, where g[k] is the average norm of the gradient of the last character's loss with
    respect to the input vector k steps earlier. A plain RNN's g falls off exponentially in k
    (vanishing gradients); an LSTM's falls much more slowly.
    """
    gen = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(stream) - length - 1, (batch_size,), generator=gen)
    x = torch.stack([stream[s:s + length] for s in starts])
    y = torch.stack([stream[s + 1:s + length + 1] for s in starts])
    e = model.emb(x).detach().requires_grad_(True)       # the input vectors, as leaves
    h, _ = model.rnn(e)
    loss = nn.functional.cross_entropy(model.out(h[:, -1]), y[:, -1])
    loss.backward()
    g = e.grad.norm(dim=-1).mean(0)                      # (length,): one number per position
    return g.flip(0)                                     # g[0] = the last input, g[k] = k steps back
