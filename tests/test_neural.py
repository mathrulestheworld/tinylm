"""Checks for the PyTorch language models: shapes, windows, and that predictions use only the past."""
import torch

from tinylm.data import END
from tinylm.neural import (FixedWindowLM, CharRNN, word_index, make_windows, detokenize,
                           char_alphabet, encode_chars, STORY_END)


def test_windows_and_fixed_window_model():
    vocab = sorted(["a", "b", END, "<unk>"])
    index = word_index(vocab)
    X, Y = make_windows([["a", "b"]], index, context=2)
    bos = index["<s>"]
    assert X.tolist() == [[bos, bos], [bos, index["a"]], [index["a"], index["b"]]]
    assert Y.tolist() == [index["a"], index["b"], index[END]]
    model = FixedWindowLM(len(vocab) + 1, len(vocab), context=2, embed_dim=3, hidden=5)
    assert model(X).shape == (3, len(vocab))


def test_char_rnn_is_causal():
    # Changing a later character must not change the predictions at earlier positions.
    for kind in ["rnn", "lstm"]:
        torch.manual_seed(0)
        model = CharRNN(10, kind=kind, hidden=8, embed_dim=4)
        x = torch.randint(0, 10, (1, 12))
        y = x.clone()
        y[0, 7] = (x[0, 7] + 1) % 10
        a, _ = model(x)
        b, _ = model(y)
        assert torch.allclose(a[0, :7], b[0, :7])
        assert not torch.allclose(a[0, 7:], b[0, 7:])


def test_char_encoding_and_detokenize():
    alphabet = char_alphabet(["ab", "ba!"])
    assert alphabet[0] == "!" and STORY_END in alphabet
    stream = encode_chars(["ab"], alphabet)
    assert [alphabet[i] for i in stream.tolist()] == ["a", "b", STORY_END]
    assert detokenize(["once", "upon", "a", "time", ",", "a", "cat", "."]) == "once upon a time, a cat."
