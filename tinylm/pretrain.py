"""Pretraining a language model on tokens (Week 5; Lecture 5, §§5-10).

The pieces of a pretraining run, as the Lecture 5 notes define them:

    pack            documents joined into one stream, each after an <|endoftext|> token (§5, "Packing")
    get_batch       B chunks of T + 1 consecutive tokens cut from the stream at random (§5)
    lr_at           the learning rate at a step: a linear warmup, then a cosine decay to a tenth, a constant
                    rate, or warmup-stable-decay (§7, "Learning-rate schedules")
    Muon            momentum, then each matrix's step replaced by the nearest orthogonal matrix (§6)
    make_optimizer  SGD, momentum, Adam, AdamW with weight decay on the matrices only, or Muon for the
                    blocks' matrices with AdamW for the rest (§§6-7)
    pretrain        the training loop: batches, the log loss, backpropagation, clipping, the schedule (§§7-8)
    stories_loss    the log loss of whole stories, each read from <|endoftext|> (§8)

The model is Week 4's TransformerLM (tinylm/transformer.py) with the base model's sizes, BASE_MODEL. Three
trained models of the notes are in checkpoints/, saved in 16-bit floats (float16) to halve their size:

    week05_base.pt   the base model of §8: 6 blocks of width 192, 6 heads, context 512, the 4,096 tokens of
                     week05_tokenizer.json, 3,456,000 parameters; 6,000 steps of 16 x 512 tokens (49 million
                     tokens of TinyStories' training file and sums), 6.0 hours on a processor
    week05_mid.pt    the base model midtrained for 1,000 steps on stories and conversations (§12)
    week05_chat.pt   the midtrained model fine-tuned on 24,179 conversations (§13): the chat model

They were trained by the scripts of the course's Lecture 5 folder (pretrain.py, midtrain.py, sft.py).
"""
import math
import time

import torch
import torch.nn as nn

from .transformer import TransformerLM

BASE_MODEL = dict(d_model=192, n_layers=6, n_heads=6, max_len=512)


def fast_attention(q, k, v):
    """Causal attention by PyTorch's fused kernel: the same read as attention_solution, without storing the
    T x T weights (so it returns None for them). Used for training, where it is several times faster."""
    return nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), None


def load_lm(path, vocab_size=4096, attention=None, **sizes):
    """A trained model from checkpoints/, in 32-bit floats, ready to use (eval mode). The sizes default to
    BASE_MODEL; attention defaults to fast_attention (pass attention_solution to keep the weights)."""
    model = TransformerLM(vocab_size, positions="rope", **(sizes or BASE_MODEL))
    state = {k: v.float() for k, v in torch.load(path, map_location="cpu").items()}
    state.setdefault("unembed.weight", state["tok.weight"])          # the output layer is the token table
    model.load_state_dict(state)
    model.set_attention(attention or fast_attention)
    model.eval()
    return model


# ----------------------------------------------------------------------------- data

def pack(documents, eot):
    """One stream of token ids: each document (a list of ids) after an <|endoftext|> token (id eot)."""
    stream = []
    for doc in documents:
        stream.append(eot)
        stream.extend(doc)
    return torch.tensor(stream, dtype=torch.long)


def get_batch(stream, batch_size, length, gen):
    """batch_size chunks of length + 1 consecutive tokens from random places in the stream: the inputs are the
    first `length` tokens of each, the targets the last `length`, so position t predicts token t + 1."""
    starts = torch.randint(0, len(stream) - length - 1, (batch_size,), generator=gen)
    chunks = torch.stack([stream[s:s + length + 1] for s in starts.tolist()])
    return chunks[:, :-1], chunks[:, 1:]


# ----------------------------------------------------------------------------- the schedule

def lr_at(step, steps, peak, warmup, kind="cosine", final=0.1, decay_frac=0.2):
    """The learning rate at a step (counted from 1): a linear warmup over `warmup` steps, then
    "cosine": half a cosine down to final * peak at the last step;
    "constant": peak;
    "wsd": warmup-stable-decay, constant, then a linear decay to final * peak over the last decay_frac."""
    if warmup and step <= warmup:
        return peak * step / warmup
    if kind == "constant":
        return peak
    if kind == "cosine":
        progress = (step - warmup) / max(steps - warmup, 1)
        return peak * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress)))
    if kind == "wsd":
        start = steps * (1 - decay_frac)
        if step <= start:
            return peak
        return peak * (1 - (1 - final) * (step - start) / max(steps - start, 1))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- optimizers

def newton_schulz(G, steps=5):
    """An approximation of the orthogonal matrix nearest G (U V^T, for G = U S V^T): G scaled to norm 1, then
    five times X <- aX + b(XX^T)X + c(XX^T)^2 X, which maps every singular value s to as + bs^3 + cs^5 and
    leaves the singular vectors alone (Jordan et al., 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = G.shape[0] > G.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    """Muon, for weight matrices: a momentum M <- beta M + G, then a step along the orthogonalized Nesterov
    momentum, newton_schulz(G + beta M), scaled by sqrt(max(1, d_out / d_in)) (the notes' §6)."""

    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                m.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(m, alpha=group["momentum"])                    # Nesterov's form
                o = newton_schulz(g)
                p.add_(o, alpha=-group["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


class Combined:
    """Several optimizers stepped together (Muon for some parameters, AdamW for the others)."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step(self):
        for o in self.optimizers:
            o.step()


def unique_parameters(model):
    seen, out = set(), []
    for name, p in model.named_parameters():
        if id(p) not in seen and p.requires_grad:
            seen.add(id(p))
            out.append((name, p))
    return out


def make_optimizer(model, kind, lr, weight_decay=0.1, betas=(0.9, 0.95), other_lr=0.01):
    """kind: "sgd", "momentum" (0.9), "adam", "adamw" (weight decay on tensors of two or more dimensions only),
    or "muon" (Muon at lr on the matrices inside the blocks; AdamW at other_lr on the token table, the
    normalizations, and the biases). Each param group keeps its peak rate as "peak", for the schedule."""
    params = unique_parameters(model)
    if kind == "sgd":
        opt = torch.optim.SGD([p for _, p in params], lr=lr)
    elif kind == "momentum":
        opt = torch.optim.SGD([p for _, p in params], lr=lr, momentum=0.9)
    elif kind == "adam":
        opt = torch.optim.Adam([p for _, p in params], lr=lr, betas=betas)
    elif kind == "adamw":
        decay = [p for _, p in params if p.dim() >= 2]
        rest = [p for _, p in params if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                 {"params": rest, "weight_decay": 0.0}], lr=lr, betas=betas)
    elif kind == "muon":
        matrices = [p for n, p in params if p.dim() == 2 and n.startswith("blocks.")]
        rest = [p for n, p in params if not (p.dim() == 2 and n.startswith("blocks."))]
        opt = Combined(Muon(matrices, lr=lr),
                       torch.optim.AdamW([{"params": [p for p in rest if p.dim() >= 2], "weight_decay": weight_decay},
                                          {"params": [p for p in rest if p.dim() < 2], "weight_decay": 0.0}],
                                         lr=other_lr, betas=betas))
    else:
        raise ValueError(kind)
    for group in opt.param_groups:
        group["peak"] = group["lr"]
    return opt


# ----------------------------------------------------------------------------- the loop

def pretrain(model, stream, steps, batch_size=16, length=256, optimizer="adamw", peak=3e-3, warmup=100,
             schedule="cosine", weight_decay=0.1, clip=1.0, seed=0, every=25, log=print):
    """The pretraining loop: for each step, a batch of chunks from the stream; the average log loss of every
    next-token prediction; backpropagation; the gradient clipped at norm `clip`; the learning rate of the
    schedule (every group scaled by the same factor); one optimizer step. Returns [(step, training loss)]."""
    gen = torch.Generator().manual_seed(seed)
    opt = make_optimizer(model, optimizer, peak, weight_decay)
    model.train()
    curve = []
    start = time.time()
    for step in range(1, steps + 1):
        factor = lr_at(step, steps, 1.0, warmup, schedule)
        for group in opt.param_groups:
            group["lr"] = group["peak"] * factor
        x, y = get_batch(stream, batch_size, length, gen)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_([p for _, p in unique_parameters(model)], clip)
        opt.step()
        if step % every == 0 or step == 1:
            curve.append((step, loss.item()))
            if log is not None:
                log(f"step {step:4d}   loss {loss.item():.3f}   ({time.time() - start:.0f} s)")
    model.eval()
    return curve


# ----------------------------------------------------------------------------- measuring

def stories_loss(model, stories, eot, batch_size=32):
    """Total log loss (nats) and number of predictions over stories (lists of ids), each read from
    <|endoftext|>: inputs <|endoftext|> + story, targets story + <|endoftext|>, so that the end of the story is
    predicted too. Stories longer than the context are read in windows moved by half of it."""
    T = model.max_len
    total, count = 0.0, 0
    was = model.training
    model.eval()
    with torch.no_grad():
        xs = [torch.tensor([eot] + list(s)) for s in stories]
        ys = [torch.tensor(list(s) + [eot]) for s in stories]
        short = sorted([i for i in range(len(xs)) if len(xs[i]) <= T], key=lambda i: len(xs[i]))
        long = [i for i in range(len(xs)) if len(xs[i]) > T]
        for k in range(0, len(short), batch_size):
            b = short[k:k + batch_size]
            X = nn.utils.rnn.pad_sequence([xs[j] for j in b], batch_first=True, padding_value=eot)
            Y = nn.utils.rnn.pad_sequence([ys[j] for j in b], batch_first=True, padding_value=-100)
            logits = model(X)
            total += nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1),
                                                 ignore_index=-100, reduction="sum").item()
            count += int((Y != -100).sum())
        for j in long:
            x, y = xs[j], ys[j]
            scored, s = 0, 0
            while scored < len(x):
                s = min(s, len(x) - T)
                logits = model(x[None, s:s + T])[0]
                first = scored - s
                total += nn.functional.cross_entropy(logits[first:], y[s + first:s + T], reduction="sum").item()
                count += T - first
                scored = s + T
                s += T // 2
    model.train(was)
    return total, count


def token_losses(model, ids):
    """The log loss (nats) of every token of one sequence after the first: a tensor of len(ids) - 1."""
    x = torch.tensor([ids])
    with torch.no_grad():
        logp = torch.log_softmax(model(x[:, :-1])[0], -1)
    return -logp.gather(-1, x[0, 1:, None])[:, 0]


def sample(model, prompt, max_new, temperature=0.8, seed=1, stop=()):
    """Tokens sampled after prompt (a list of ids), one at a time from the last position's distribution
    divided by the temperature (0: the most probable token), until a token in `stop` (kept) or max_new
    tokens. The whole sequence is fed again at every step; Week 8 keeps the keys and values instead."""
    gen = torch.Generator().manual_seed(seed)
    ids = list(prompt)
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            if len(ids) >= model.max_len:
                break
            logits = model(torch.tensor([ids]))[0, -1]
            if temperature == 0:
                token = int(logits.argmax())
            else:
                token = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).item()
            out.append(token)
            ids.append(token)
            if token in stop:
                break
    return out
