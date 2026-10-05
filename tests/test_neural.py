"""Checks for the PyTorch language models: shapes, windows, and that predictions use only the past."""
import torch

from tinylm.data import END
from tinylm.neural import (FixedWindowLM, CharRNN, word_index, make_windows, detokenize,
                           char_alphabet, encode_chars, STORY_END, BigramNet, WordLSTM, story_tensors,
                           story_perplexity)


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


def test_bigram_net_learns_the_counts():
    # previous -> next pairs with counts 3:1 after token 0 and 1:1 after token 1
    a = torch.tensor([0, 0, 0, 0, 1, 1])
    b = torch.tensor([1, 1, 1, 2, 0, 2])
    net = BigramNet(3, 3)
    assert torch.allclose(torch.softmax(net(torch.tensor([0])), -1), torch.full((1, 3), 1 / 3))   # uniform start
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    for _ in range(800):
        loss = torch.nn.functional.cross_entropy(net(a), b)
        opt.zero_grad(); loss.backward(); opt.step()
    p = torch.softmax(net(torch.tensor([0, 1])), -1).detach()
    assert abs(p[0, 1] - 0.75) < 0.02 and abs(p[0, 2] - 0.25) < 0.02 and abs(p[1, 0] - 0.5) < 0.02
    assert BigramNet(4, 3, embed_dim=2)(torch.tensor([[0], [3]])).shape == (2, 3)


def test_word_lstm_story_perplexity_and_causality():
    vocab = sorted(["a", "b", END, "<unk>"])
    index = word_index(vocab)
    xs, ys = story_tensors([["a", "b", "a"], ["b"]], index)
    assert xs[0].tolist() == [index["<s>"], index["a"], index["b"], index["a"]]
    assert ys[0].tolist() == [index["a"], index["b"], index["a"], index[END]]
    torch.manual_seed(0)
    model = WordLSTM(len(vocab) + 1, len(vocab), embed_dim=4, hidden=5)
    with torch.no_grad():
        model.U.weight.zero_(); model.U.bias.zero_()                 # uniform predictions
    assert abs(story_perplexity(model, xs, ys) - len(vocab)) < 1e-4
    x = xs[0][None]
    y1, _ = model(x)
    x2 = x.clone(); x2[0, -1] = index["b"]
    y2, _ = model(x2)
    torch.nn.init.normal_(model.U.weight)
    y1, _ = model(x); y2, _ = model(x2)
    assert torch.allclose(y1[0, :-1], y2[0, :-1]) and not torch.allclose(y1[0, -1], y2[0, -1])
