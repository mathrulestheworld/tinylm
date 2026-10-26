"""Supervised fine-tuning: a base model becomes a chat model (Week 5; Lecture 5, §§11-13).

The chat template of the notes (§11): a conversation is one sequence of tokens,

    <|endoftext|> <|system|> text <|end|> <|user|> text <|end|> <|assistant|> text <|end|> ...

with a role's token before each message and <|end|> after it; the system message is optional. To get an
answer, the program writes the conversation so far and <|assistant|>, and the model writes until <|end|>.

    render_solution   a conversation's token ids and its loss mask: 1 on the assistant's text and the <|end|>
                      that closes it, 0 elsewhere (the Week 5 notebook asks you to write this as render)
    prompt_ids        a conversation up to and including the <|assistant|> that opens the model's turn
    to_example        ids and mask -> inputs and targets, with -100 where a target is not in the loss
    assistant_loss    the average log loss over the assistant's tokens of held-out conversations
    LoRALinear, add_lora, train_rows, merge_lora
                      low-rank adaptation (§7): every chosen matrix frozen and a trained change x A B added;
                      train_rows also trains chosen rows of the token table (the four new tokens, §13)
    finetune          the fine-tuning loop of §13: one conversation per row, batches of similar length,
                      AdamW, warmup over 5% of the steps, cosine decay to a tenth, clipping at 1
    answer            the assistant's turn, sampled after the prompt until <|end|>
    words_used        how many of a request's words a story uses
"""
import math
import random
import re
import time

import torch
import torch.nn as nn

from .pretrain import lr_at, make_optimizer, sample, unique_parameters

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def template_ids(tok):
    """The ids of the template's special tokens: eot, system, user, assistant, end."""
    sp = tok.special_ids
    return dict(eot=sp["<|endoftext|>"], system=sp["<|system|>"], user=sp["<|user|>"],
                assistant=sp["<|assistant|>"], end=sp["<|end|>"])


def render_solution(messages, tok):
    """Token ids of a conversation in the template, and its loss mask (1 where the token is the assistant's to
    write: its text and the <|end|> after it)."""
    t = template_ids(tok)
    ids, mask = [t["eot"]], [0]
    for m in messages:
        body = tok.encode(m["content"]) + [t["end"]]
        ids += [t[m["role"]]] + body
        mask += [0] + [1 if m["role"] == "assistant" else 0] * len(body)
    return ids, mask


def prompt_ids(messages, tok):
    """The conversation so far, then <|assistant|>: what the model continues to write its turn."""
    ids, _ = render_solution(messages, tok)
    return ids + [template_ids(tok)["assistant"]]


def to_example(ids, mask):
    """Inputs ids[:-1] and targets ids[1:], each target replaced by -100 (left out of the loss) where its mask
    is 0. Position t reads tokens 0..t and predicts token t + 1."""
    x = list(ids[:-1])
    y = [token if keep else -100 for token, keep in zip(ids[1:], mask[1:])]
    return x, y


def sft_batches(examples, batch_size, gen, rng):
    """Groups of batch_size examples of similar length, in random order, each padded to its longest: inputs with
    0 (the padding comes after every real token, so causal attention never reads it) and targets with -100."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    groups = [order[k:k + batch_size] for k in range(0, len(order), batch_size)]
    for g in torch.randperm(len(groups), generator=gen).tolist():
        xs = [torch.tensor(examples[i][0]) for i in groups[g]]
        ys = [torch.tensor(examples[i][1]) for i in groups[g]]
        X = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
        Y = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        yield X, Y


def masked_loss(model, X, Y, reduction="mean"):
    logits = model(X)
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1), ignore_index=-100,
                                       reduction=reduction)


def assistant_loss(model, examples, batch_size=16):
    """The average log loss (nats per token) over the targets in the loss, here the assistant's tokens.
    Examples longer than the model's context are left out."""
    examples = [e for e in examples if len(e[0]) <= model.max_len]
    total, count = 0.0, 0
    was = model.training
    model.eval()
    with torch.no_grad():
        for X, Y in sft_batches(examples, batch_size, torch.Generator().manual_seed(0), random.Random(0)):
            total += masked_loss(model, X, Y, "sum").item()
            count += int((Y != -100).sum())
    model.train(was)
    return total / count


# ----------------------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    """x W^T + b + x A B, with the layer's W and b frozen, A (d_in x r) random and B (r x d_out) zero, so that
    training starts exactly from the frozen layer. Hu et al. (2022) write W0 x + B A x for column vectors; our A
    and B are their transposes, in the same roles."""

    def __init__(self, base, r):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(base.in_features, r) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(r, base.out_features))

    def forward(self, x):
        return self.base(x) + (x @ self.A) @ self.B

    def merged(self):
        """An ordinary layer with W + (A B)^T: the same function, at the original cost."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + (self.A @ self.B).T)
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


LORA_TARGETS = (("attn", "query"), ("attn", "key"), ("attn", "value"), ("attn", "out"), ("mlp", "up"), ("mlp", "down"))


def add_lora(model, r):
    """Freeze every parameter and wrap every attention and MLP matrix of every block in a LoRALinear of rank r."""
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        for part, name in LORA_TARGETS:
            module = getattr(block, part)
            setattr(module, name, LoRALinear(getattr(module, name), r))
    return model


def train_rows(model, token_ids):
    """Also train the rows of the token table for token_ids (the table is tied to the output layer); a hook
    zeroes the gradient of every other row, so with no weight decay they stay as they are."""
    table = model.tok.weight
    table.requires_grad = True
    keep = torch.zeros(table.shape[0], 1)
    keep[list(token_ids)] = 1.0
    table.register_hook(lambda g: g * keep)
    return model


def merge_lora(model):
    """Replace every LoRALinear by its merged layer."""
    for block in model.blocks:
        for part, name in LORA_TARGETS:
            module = getattr(block, part)
            layer = getattr(module, name)
            if isinstance(layer, LoRALinear):
                setattr(module, name, layer.merged())
    return model


def trained_parameters(model, rows=0):
    """The number of numbers training changes: parameters with requires_grad, counting only `rows` rows of the
    token table when train_rows is used."""
    n = sum(p.numel() for _, p in unique_parameters(model))
    if rows and model.tok.weight.requires_grad:
        n -= model.tok.weight.numel() - rows * model.tok.weight.shape[1]
    return n


# ----------------------------------------------------------------------------- the loop

def finetune(model, examples, epochs=1.0, batch_size=16, peak=3e-4, warmup_frac=0.05, weight_decay=0.1, clip=1.0,
             heldout=None, every=None, seed=0, log=print):
    """Supervised fine-tuning (§13): one conversation per row, batches of similar length, the average log loss
    over the targets in the loss (the assistant's tokens); AdamW (0.9, 0.95) with weight decay on the matrices,
    a warmup over warmup_frac of the steps, a cosine decay to a tenth, clipping at `clip`. Only parameters with
    requires_grad are trained (all of them, or LoRA's). Every `every` steps, the loss on `heldout` examples.
    Returns [(step, training loss, held-out loss or None)]."""
    too_long = sum(len(x) > model.max_len for x, _ in examples)
    assert not too_long, f"{too_long} examples are longer than the context of {model.max_len} tokens; leave them out"
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    steps = max(1, round(math.ceil(len(examples) / batch_size) * epochs))
    warmup = max(1, round(warmup_frac * steps))
    every = every or max(1, steps // 10)
    opt = make_optimizer(model, "adamw", peak, weight_decay)
    params = [p for _, p in unique_parameters(model)]
    curve = []
    if heldout is not None:
        curve.append((0, None, assistant_loss(model, heldout)))
    model.train()
    start, step = time.time(), 0
    while step < steps:
        for X, Y in sft_batches(examples, batch_size, gen, rng):
            step += 1
            for group in opt.param_groups:
                group["lr"] = group["peak"] * lr_at(step, steps, 1.0, warmup, "cosine")
            loss = masked_loss(model, X, Y)
            opt.zero_grad()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            if step % every == 0 or step == steps:
                held = assistant_loss(model, heldout) if heldout is not None else None
                curve.append((step, loss.item(), held))
                if log is not None:
                    log(f"step {step:4d}/{steps}   loss {loss.item():.3f}"
                        + (f"   held-out {held:.3f}" if held is not None else "") + f"   ({time.time() - start:.0f} s)")
            if step >= steps:
                break
    model.eval()
    return curve


# ----------------------------------------------------------------------------- using the chat model

def answer(model, tok, messages, temperature=0.8, seed=1, max_new=400):
    """The assistant's turn after `messages`: its text, and whether it ended with <|end|> (rather than
    <|endoftext|>, the context's end, or max_new tokens)."""
    t = template_ids(tok)
    out = sample(model, prompt_ids(messages, tok), max_new, temperature, seed, stop=(t["end"], t["eot"]))
    ended = bool(out) and out[-1] == t["end"]
    return tok.decode([i for i in out if i not in tok.special_names]), ended


def words_used(text, words):
    """(found, asked): how many of the comma-separated words a text uses (as the start of a word, so that
    'jump' counts 'jumped')."""
    asked = [w.strip().lower() for w in words.split(",") if w.strip()]
    found = sum(1 for w in asked if re.search(r"\b" + re.escape(w), text.lower()))
    return found, len(asked)
