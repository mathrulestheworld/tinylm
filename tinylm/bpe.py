"""Byte-level byte-pair encoding (Week 5; Lecture 5, §§2-4).

Training (the notes' §3):
    1. Split the text into pre-tokens with a regular expression (a word with the space before it, one digit,
       a run of punctuation, a run of spaces), and count each distinct pre-token.
    2. Write every pre-token as its UTF-8 bytes: token ids 0..255.
    3. Repeat until the vocabulary is full: count every adjacent pair of tokens inside the pre-tokens, each
       weighted by its pre-token's count; merge the most frequent pair into a new token everywhere, and
       record the merge.

Encoding a new text applies the recorded merges, in the order they were learned, inside each pre-token.
Decoding joins the tokens' bytes. Every string can be encoded, since every byte is a token. Special tokens
(<|endoftext|>, <|system|>, <|user|>, <|assistant|>, <|end|>, and three reserved) come after the merges; the
encoder never produces them from text, so a user who types "<|end|>" gets ordinary tokens (§4).

checkpoints/week05_tokenizer.json is the tokenizer of the Lecture 5 notes: 4,096 tokens = 256 bytes + 3,832
merges + 8 special tokens, trained on 100,000 stories of TinyStories' training file with their share of sums
(the course's Lecture 5 folder, scripts/bpe.py, about 20 seconds). The Week 5 notebook asks you to write the
training loop on characters (train_by_hand); train_bpe below is the same loop on bytes, with the pair counts
updated incrementally so that thousands of merges take seconds.
"""
import re
from collections import Counter

# Pre-tokens: English contractions; letters with an optional leading space; ONE digit (so that
# numbers are split into digits, as in Llama, not GPT-2); runs of other symbols with an optional
# leading space; spaces before a non-space are left to the next pre-token; other whitespace.
PRETOKEN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")

# GPT-2's pattern differs in one place: a run of digits is one pre-token (" ?\p{N}+").
PRETOKEN_GPT2 = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")

SPECIALS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            "<|reserved1|>", "<|reserved2|>", "<|reserved3|>"]


def pretokens(text, pattern=PRETOKEN):
    return pattern.findall(text)


def count_pretokens(texts, pattern=PRETOKEN):
    counts = Counter()
    for text in texts:
        counts.update(pattern.findall(text))
    return counts


def train_bpe(pretoken_counts, n_merges, verbose=False):
    """Learn n_merges merges from a dict {pre-token string: count}.

    Returns the list of merges [(a, b), ...]; merge i creates token 256 + i.
    Ties between pairs of equal count are broken by the smaller pair (a, b), so the result is
    deterministic.
    """
    words = []        # each distinct pre-token as a list of token ids
    freqs = []        # its count
    for w, c in pretoken_counts.items():
        words.append(list(w.encode("utf-8")))
        freqs.append(c)

    pair_counts = Counter()          # (a, b) -> weighted count over all pre-tokens
    where = {}                       # (a, b) -> set of indices of pre-tokens that contain it
    for i, ids in enumerate(words):
        for pair in zip(ids, ids[1:]):
            pair_counts[pair] += freqs[i]
            where.setdefault(pair, set()).add(i)

    merges = []
    for m in range(n_merges):
        if not pair_counts:
            break
        best = max(pair_counts.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))[0]
        if pair_counts[best] <= 0:
            break
        new = 256 + m
        merges.append(best)
        for i in list(where.get(best, ())):
            ids = words[i]
            f = freqs[i]
            # remove this pre-token's old pairs
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] -= f
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                s = where.get(pair)
                if s is not None:
                    s.discard(i)
            # merge every occurrence of best, left to right
            out = []
            j = 0
            while j < len(ids):
                if j + 1 < len(ids) and ids[j] == best[0] and ids[j + 1] == best[1]:
                    out.append(new)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            words[i] = out
            # add its new pairs
            for pair in zip(out, out[1:]):
                pair_counts[pair] += f
                where.setdefault(pair, set()).add(i)
        where.pop(best, None)
        pair_counts.pop(best, None)
        if verbose and (m + 1) % 500 == 0:
            print(f"  {m + 1} merges", flush=True)
    return merges


class Tokenizer:
    """A byte-level BPE tokenizer: 256 bytes, the learned merges, then the special tokens."""

    def __init__(self, merges, specials=SPECIALS, pattern=PRETOKEN):
        self.merges = [tuple(m) for m in merges]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.pattern = pattern
        self.token_bytes = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self.special_ids = {}
        for s in specials:
            self.special_ids[s] = len(self.token_bytes)
            self.token_bytes.append(s.encode("utf-8"))
        self.special_names = {i: s for s, i in self.special_ids.items()}
        self.cache = {}

    def __len__(self):
        return len(self.token_bytes)

    def encode_pretoken(self, piece):
        if piece in self.cache:
            return self.cache[piece]
        ids = list(piece.encode("utf-8"))
        while len(ids) > 1:
            # the adjacent pair that was merged earliest in training
            best = None
            best_rank = None
            for pair in zip(ids, ids[1:]):
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new = 256 + best_rank
            out = []
            j = 0
            while j < len(ids):
                if j + 1 < len(ids) and ids[j] == best[0] and ids[j + 1] == best[1]:
                    out.append(new)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            ids = out
        if len(self.cache) < 500000:
            self.cache[piece] = ids
        return ids

    def encode(self, text):
        """Token ids of a text. Special-token strings in the text are encoded as ordinary bytes."""
        ids = []
        for piece in self.pattern.findall(text):
            ids.extend(self.encode_pretoken(piece))
        return ids

    def decode(self, ids):
        return b"".join(self.token_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def pieces(self, ids):
        """The text of each token (bytes that are not valid UTF-8 alone shown as <xx>)."""
        out = []
        for i in ids:
            b = self.token_bytes[i]
            try:
                out.append(b.decode("utf-8"))
            except UnicodeDecodeError:
                out.append("".join(f"<{x:02x}>" for x in b))
        return out

    def to_dict(self):
        return {"merges": [list(m) for m in self.merges], "specials": list(self.special_ids),
                "pattern": self.pattern.pattern}

    @staticmethod
    def from_dict(d):
        return Tokenizer(d["merges"], d["specials"], re.compile(d["pattern"]))


def load_tokenizer(path):
    """The tokenizer saved by Tokenizer.to_dict() as JSON, such as checkpoints/week05_tokenizer.json."""
    import json
    with open(path, encoding="utf-8") as f:
        return Tokenizer.from_dict(json.load(f))


def bits_per_byte(total_nats, n_bytes):
    """A log loss summed over a text (in nats), as bits per byte of the text: independent of the tokenizer."""
    import math
    return total_nats / (n_bytes * math.log(2))
