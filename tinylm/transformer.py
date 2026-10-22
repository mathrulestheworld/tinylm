"""A decoder-only Transformer language model (Week 4).

The word model of the Lecture 4 notes (§8): a stack of pre-norm blocks of masked self-attention and
an MLP, rotary positions, and one token table used at the input and, in reverse, at the output:

    h = E[x]                                   one vector of d numbers per position: the residual stream
    for each of the L blocks:
        h = h + attention(LayerNorm(h))        each position reads itself and the positions before it
        h = h + MLP(LayerNorm(h))              each position is transformed on its own
    logits = LayerNorm(h) E^T                  one logit per token of the vocabulary, at every position

Two trained checkpoints are in checkpoints/:

    week04_word_transformer.pt   the word model: d = 128, 4 blocks, 4 heads, context 256, the 8,681 words
                                 of Week 2 plus BOS; 1,904,640 parameters; test perplexity 9.9
    week04_leaky.pt              the same model trained for 300 steps with a wrong mask, which also lets
                                 each position read the next one (§8, "Checking the mask")

Both were trained by scripts/word_transformer.py and scripts/leaky.py in the course's Lecture 4 folder
(4,000 steps of 16 chunks of 256 words, 54 minutes on two processor cores; and 300 steps).

The attention computes, for each head, scores S = Q K^T / sqrt(d_k) + M, weights A = softmax(S) row by
row, and the read A V, where the mask M is 0 where a position may read and -infinity above the
diagonal. Every SelfAttention layer calls one function attention(q, k, v) for this; the Week 4 notebook
asks you to write it, and attention_solution below is one answer.
"""
import math
import time

import torch
import torch.nn as nn

WORD_MODEL = dict(d_model=128, n_layers=4, n_heads=4, max_len=256)


# ----------------------------------------------------------------------------- attention

def attention_solution(q, k, v):
    """Causal attention for every head at once.

    q, k, v: (B, H, T, d_k), the queries, keys, and values of H heads at T positions.
    Returns the read (B, H, T, d_k) and the weights (B, H, T, T); row t of the weights is zero after t.
    """
    d_k = q.shape[-1]
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)                      # (B, H, T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))                      # position t cannot read t+1, ...
    weights = torch.softmax(scores, dim=-1)                                 # each row sums to 1
    return weights @ v, weights


def leaky_attention(q, k, v):
    """A wrong mask: each position can also read the next one, the token it must predict."""
    d_k = q.shape[-1]
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=2)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


# ----------------------------------------------------------------------------- rotary positions

def rope_angles(max_len, d_head, base=10000.0):
    """cos and sin of the angles t * w_i, for every position t and pair i of a head's coordinates."""
    w = base ** (-torch.arange(0, d_head, 2, dtype=torch.float) / d_head)       # (d_head/2,)
    angles = torch.arange(max_len, dtype=torch.float)[:, None] * w              # (max_len, d_head/2)
    return torch.cos(angles), torch.sin(angles)


def rotate(x, cos, sin):
    """Rotate each pair of coordinates (x[2i], x[2i+1]) of x (..., T, d_head) by its position's angle."""
    T = x.shape[-2]
    c = cos[:T]
    s = sin[:T]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = even * s + odd * c
    return out


# ----------------------------------------------------------------------------- the parts

class SelfAttention(nn.Module):
    """Multi-head self-attention: project to queries, keys, and values, split into heads, rotate the
    queries and keys by their positions, attend, put the heads side by side, and mix them with W_O."""

    def __init__(self, d_model, n_heads, rope=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)          # W_Q and its bias
        self.key = nn.Linear(d_model, d_model)            # W_K
        self.value = nn.Linear(d_model, d_model)          # W_V
        self.out = nn.Linear(d_model, d_model)            # W_O
        self.rope = rope                                  # (cos, sin) from rope_angles, or None
        self.attention = attention_solution               # the function that computes each head's read
        self.weights = None                               # the weights of the last call, if keep_weights
        self.keep_weights = False

    def split_heads(self, x):
        B, T, d = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)        # (B, H, T, d_head)

    def forward(self, x):
        B, T, d = x.shape
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        if self.rope is not None:
            q = rotate(q, *self.rope)
            k = rotate(k, *self.rope)
        read, weights = self.attention(q, k, v)                               # (B, H, T, d_head)
        if self.keep_weights:
            self.weights = weights.detach()
        read = read.transpose(1, 2).reshape(B, T, d)                          # heads side by side
        return self.out(read)


class MLP(nn.Module):
    """The position-wise network: d -> 4d -> d, with a GELU in between."""

    def __init__(self, d_model):
        super().__init__()
        self.up = nn.Linear(d_model, 4 * d_model)         # W_1, b_1
        self.down = nn.Linear(4 * d_model, d_model)       # W_2, b_2

    def forward(self, x):
        return self.down(nn.functional.gelu(self.up(x), approximate="tanh"))


class Block(nn.Module):
    """One pre-norm block: h + attention(LayerNorm(h)), then h + MLP(LayerNorm(h))."""

    def __init__(self, d_model, n_heads, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, h):
        h = h + self.attn(self.norm1(h))
        h = h + self.mlp(self.norm2(h))
        return h


class TransformerLM(nn.Module):
    """A decoder-only Transformer language model.

    x: (B, T) token ids, T <= max_len. Returns logits (B, T, vocab_size); the logits at position t
    predict token t+1 and depend only on tokens 0..t. positions: "rope" (rotary), "learned" (one
    trained vector per position, as in GPT-2), or "none". no_output: a token id that is never
    predicted (its logit is set to -infinity), such as BOS.
    """

    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, max_len=256, positions="rope", no_output=None):
        super().__init__()
        self.max_len = max_len
        self.no_output = no_output
        self.tok = nn.Embedding(vocab_size, d_model)                          # the token table E
        self.pos = nn.Embedding(max_len, d_model) if positions == "learned" else None
        rope = rope_angles(max_len, d_model // n_heads) if positions == "rope" else None
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(Block(d_model, n_heads, rope))
        self.norm = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.unembed.weight = self.tok.weight                                 # tied: the same table, in reverse
        # GPT-2's initialization: small random weights, zero biases, and the two layers of each block that
        # write into the residual stream scaled down by sqrt(2L), so the stream does not grow with depth
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                std = 0.02
                if name.endswith("out.weight") or name.endswith("down.weight"):
                    std = 0.02 / math.sqrt(2 * n_layers)
                nn.init.normal_(p, 0.0, std)
            elif name.endswith("bias") and "norm" not in name:
                nn.init.zeros_(p)

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x)                                                       # (B, T, d)
        if self.pos is not None:
            h = h + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            h = block(h)
        logits = self.unembed(self.norm(h))                                   # (B, T, vocab)
        if self.no_output is not None:
            logits[..., self.no_output] = float("-inf")
        return logits

    def set_attention(self, fn):
        """Use fn(q, k, v) -> (read, weights) in every attention layer."""
        for block in self.blocks:
            block.attn.attention = fn

    def keep_attention(self, flag=True):
        for block in self.blocks:
            block.attn.keep_weights = flag

    def attention_weights(self):
        """The weights of the last forward pass: a list over blocks of (B, H, T, T)."""
        return [block.attn.weights for block in self.blocks]


def count_parameters(model):
    """The number of parameters, counting the tied table once."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def load_word_model(path, n_inputs=8682, attention=None):
    """The trained word model: the 8,681 words of Week 2 plus BOS (the last id, never predicted)."""
    model = TransformerLM(n_inputs, no_output=n_inputs - 1, **WORD_MODEL)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    if attention is not None:
        model.set_attention(attention)
    model.eval()
    return model


# ----------------------------------------------------------------------------- measuring

def story_log_loss(model, xs, ys, batch_size=32):
    """Total log loss (nats) and number of predictions, each story read from its beginning in one pass.

    xs[i] and ys[i] are the inputs (BOS, then the words) and targets (the words, then the end-of-story
    token) of story i, as made by tinylm.neural.story_tensors. Stories longer than the context are read
    in windows of the context moved by half of it, each prediction scored once.
    """
    T = model.max_len
    pad = model.no_output if model.no_output is not None else 0
    total = 0.0
    count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        short = [i for i in range(len(xs)) if len(xs[i]) <= T]
        long = [i for i in range(len(xs)) if len(xs[i]) > T]
        short.sort(key=lambda i: len(xs[i]))
        for k in range(0, len(short), batch_size):
            group = short[k:k + batch_size]
            X = nn.utils.rnn.pad_sequence([xs[j] for j in group], batch_first=True, padding_value=pad)
            Y = nn.utils.rnn.pad_sequence([ys[j] for j in group], batch_first=True, padding_value=-100)
            logits = model(X)
            total += nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1),
                                                 ignore_index=-100, reduction="sum").item()
            count += int((Y != -100).sum())
        for j in long:
            x = xs[j]
            y = ys[j]
            scored = 0
            s = 0
            while scored < len(x):
                s = min(s, len(x) - T)
                logits = model(x[None, s:s + T])[0]
                first = scored - s
                total += nn.functional.cross_entropy(logits[first:], y[s + first:s + T], reduction="sum").item()
                count += T - first
                scored = s + T
                s += T // 2
    model.train(was_training)
    return total, count


def perplexity(model, xs, ys):
    total, count = story_log_loss(model, xs, ys)
    return math.exp(total / count)


def log_probs(model, x, y):
    """log q(y_t | x_1..x_t) at every position of one sequence, in one pass: a tensor of length T."""
    with torch.no_grad():
        logp = torch.log_softmax(model(x[None])[0], -1)
    return logp.gather(-1, y[:, None])[:, 0]


def prefix_log_probs(model, x, y):
    """The same, computed one prefix at a time: feed x_1..x_t alone and keep only the last position."""
    out = []
    with torch.no_grad():
        for t in range(1, len(x) + 1):
            logp = torch.log_softmax(model(x[None, :t])[0, -1], -1)
            out.append(logp[y[t - 1]])
    return torch.stack(out)


# ----------------------------------------------------------------------------- training and generation

def learning_rate(step, steps, peak, warmup):
    """A linear warmup to peak, then a cosine decay to peak/10 (the recipe is Lecture 5's subject)."""
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model, X, Y, steps, batch_size=16, length=256, peak=3e-3, warmup=100, seed=0, every=25, log=print):
    """The training step of the Lecture 4 notes (§8), repeated.

    X, Y: the training stories' inputs and targets, joined one after another (1-D tensors of equal
    length). Each step samples batch_size chunks of `length` positions, computes the logits and the
    average log loss of every prediction, backpropagates, clips the gradient's norm at 1, and moves every
    parameter with Adam. Returns [(step, training loss)] every `every` steps.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=peak, betas=(0.9, 0.99))
    model.train()
    curve = []
    start = time.time()
    for step in range(1, steps + 1):
        for group in opt.param_groups:
            group["lr"] = learning_rate(step, steps, peak, warmup)
        starts = torch.randint(0, len(X) - length, (batch_size,), generator=gen)
        x = torch.stack([X[s:s + length] for s in starts])                  # 1. a batch
        y = torch.stack([Y[s:s + length] for s in starts])
        logits = model(x)                                                     # 2. forward
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()                                                       # 3. backward
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                                                            # 4. update
        if step % every == 0 or step == 1:
            curve.append((step, loss.item()))
            if log is not None:
                log(f"step {step:4d}   loss {loss.item():.3f}   ({time.time() - start:.0f} s)")
    model.eval()
    return curve


def generate(model, start_ids, max_new, temperature=1.0, seed=1, stop_id=None):
    """The generation loop of the Lecture 4 notes (§8): feed the text so far, sample the next token from
    the last position's distribution (divided by the temperature), append it, and repeat."""
    gen = torch.Generator().manual_seed(seed)
    ids = list(start_ids)
    model.eval()
    with torch.no_grad():
        for i in range(max_new):
            x = torch.tensor([ids[-model.max_len:]])
            p = torch.softmax(model(x)[0, -1] / temperature, -1)
            j = torch.multinomial(p, 1, generator=gen).item()
            if j == stop_id:
                break
            ids.append(j)
    return ids[len(start_ids):]
