"""Checks of the Week 5 tokenizer (tinylm/bpe.py): the notes' merges by hand, the round trip, special tokens,
and the trained 4,096-token tokenizer in checkpoints/."""
import math
import pathlib
import random

from tinylm.bpe import Tokenizer, count_pretokens, train_bpe, pretokens, load_tokenizer, bits_per_byte, SPECIALS

CKPT = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"
CORPUS = {"hug": 10, "hugs": 5, "hugged": 4, "bug": 6, "bugs": 3, "dog": 8, "dogs": 4, "big": 5, "bigger": 2, "bag": 3}


def test_the_notes_merges_by_hand():
    """The ten-word corpus of the Lecture 5 notes, §3: u·g 28, h·ug 19, d·o 12 (a tie with o·g, broken by the
    smaller pair), do·g 12, b·ug 9, b·i 7; then 93 tokens for 180 characters."""
    merges = train_bpe(CORPUS, 6)
    tok = Tokenizer(merges)
    names = [tok.token_bytes[256 + k].decode() for k in range(6)]
    assert names == ["ug", "hug", "do", "dog", "bug", "bi"]
    assert sum(len(tok.encode(w)) * c for w, c in CORPUS.items()) == 93
    assert tok.pieces(tok.encode("dug")) == ["d", "ug"]
    assert tok.pieces(tok.encode("hugging")) == ["hug", "g", "i", "n", "g"]


def test_round_trip_on_any_text():
    texts = ["Once upon a time, there was a dog.", "237 + 682 = 919.", "Zoë saw 🐉 and said “你好” — 3.14!",
             "  two spaces,\nnew lines\n\nand\ttabs ", ""]
    tok = Tokenizer(train_bpe(count_pretokens(texts * 3), 50))
    r = random.Random(0)
    texts += ["".join(chr(r.randrange(32, 0x2FFF)) for _ in range(40)) for _ in range(20)]
    for t in texts:
        assert tok.decode(tok.encode(t)) == t


def test_pretokens_split_digits_and_keep_the_space():
    assert pretokens("Lily's dog ran 123 m!") == ["Lily", "'s", " dog", " ran", " 1", "2", "3", " m", "!"]


def test_special_tokens_come_last_and_are_never_encoded():
    tok = Tokenizer(train_bpe(CORPUS, 6))
    assert [tok.special_ids[s] for s in SPECIALS] == list(range(256 + 6, 256 + 6 + len(SPECIALS)))
    for s in SPECIALS:
        assert not set(tok.encode(s)) & set(tok.special_ids.values())


def test_the_trained_tokenizer():
    tok = load_tokenizer(CKPT / "week05_tokenizer.json")
    assert len(tok) == 4096 and tok.special_ids["<|endoftext|>"] == 4088 and tok.special_ids["<|end|>"] == 4092
    sentence = "Lily saw a big dog in the park. She gave it a bone and it was happy."
    assert len(tok.encode(sentence)) == 19
    assert tok.pieces(tok.encode("682")) == ["6", "8", "2"]
    assert len(tok.encode("<|endoftext|>")) == 8


def test_bits_per_byte():
    assert math.isclose(bits_per_byte(100 * math.log(2), 50), 2.0)
