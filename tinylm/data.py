"""Text data for the course: the TinyStories corpus, split by story, and two simple tokenizers.

TinyStories (Eldan and Li, 2023) is a set of short stories in simple English. We use its
validation file (about 22,000 stories, 19 MB), which is small enough for a laptop, and split it
ourselves into training, validation, and test stories. Splitting by story, not by sentence,
keeps sentences of one story from appearing on both sides of the split.
"""
import os
import random
import re
import urllib.request

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"

# A word is a run of letters, optionally with an apostrophe part ("didn't", "lily's");
# each punctuation mark is its own token.
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:\"]")

# Characters kept by the character-level models: letters, space, and a little punctuation.
CHAR_PATTERN = re.compile(r"[^a-z .,!?'\"]")

UNK = "<unk>"      # stands for any word outside the vocabulary
END = "</s>"       # the end-of-sequence token, EOS


def download_tinystories(folder):
    """Download the TinyStories validation file into folder (once) and return its path."""
    path = os.path.join(folder, "TinyStories-valid.txt")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        print("downloading TinyStories-valid.txt (19 MB) from Hugging Face ...")
        urllib.request.urlretrieve(TINYSTORIES_URL, path)
    return path


def load_stories(path):
    """The stories in the file, as a list of strings. Stories are separated by <|endoftext|>."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    stories = []
    for piece in text.split("<|endoftext|>"):
        piece = piece.strip()
        if piece:
            stories.append(piece)
    return stories


def split_stories(stories, seed=0):
    """Shuffle the stories with a fixed seed and split them 80% / 10% / 10%.

    Returns (train, validation, test). The same seed always gives the same split, so
    numbers in the lecture notes can be reproduced exactly.
    """
    shuffled = list(stories)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    a, b = int(0.8 * n), int(0.9 * n)
    return shuffled[:a], shuffled[a:b], shuffled[b:]


def words(text):
    """Lowercase word tokens: words, and punctuation marks as separate tokens."""
    return WORD_PATTERN.findall(text.lower())


def clean_chars(text):
    """Lowercase text with whitespace collapsed and only the characters of CHAR_PATTERN kept."""
    text = re.sub(r"\s+", " ", text.lower())
    return CHAR_PATTERN.sub("", text)


def build_vocab(sequences, min_count=2):
    """The tokens that occur at least min_count times, plus UNK and END, in sorted order."""
    counts = {}
    for seq in sequences:
        for token in seq:
            counts[token] = counts.get(token, 0) + 1
    kept = set()
    for token, c in counts.items():
        if c >= min_count:
            kept.add(token)
    kept.add(UNK)
    kept.add(END)
    return sorted(kept)


def replace_rare(sequences, vocab):
    """Replace tokens outside vocab by UNK."""
    vocab = set(vocab)
    result = []
    for seq in sequences:
        new_seq = []
        for token in seq:
            if token in vocab:
                new_seq.append(token)
            else:
                new_seq.append(UNK)
        result.append(new_seq)
    return result


def tinystories_words(folder, seed=0, min_count=2):
    """The word-level TinyStories data used in Week 2.

    Returns a dict with the token lists of the train, validation, and test stories (rare words
    replaced by UNK) and the vocabulary built from the training stories.
    """
    stories = load_stories(download_tinystories(folder))
    train, val, test = split_stories(stories, seed)
    train_words = []
    for s in train:
        train_words.append(words(s))
    vocab = build_vocab(train_words, min_count)
    val_words = []
    for s in val:
        val_words.append(words(s))
    test_words = []
    for s in test:
        test_words.append(words(s))
    return {
        "train": replace_rare(train_words, vocab),
        "val": replace_rare(val_words, vocab),
        "test": replace_rare(test_words, vocab),
        "vocab": vocab,
        "stories": (train, val, test),
    }
