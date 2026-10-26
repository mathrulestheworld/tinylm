# tinylm

A small language model built from scratch, one component per week, for the course [Generative AI from First Principles](https://mathrulestheworld.github.io/genai-first-principles/).

The course follows a language model through its whole lifecycle, and this repository is the hands-on half: by the end, the same small codebase has an autograd engine, count models and word vectors, an attention-based translator, a Transformer, tokenizers and a pretraining loop, policy gradients, a chatbot made by fine-tuning, post-training with preferences and rewards, and fast inference. Every model trains on a laptop in minutes to an hour.

## Weeks

| Week | You build | Status |
|---|---|---|
| 1 | An autograd engine on NumPy arrays; logistic regression and a two-layer network trained with it | [notebook](notebooks/week01_autograd.ipynb) |
| 2 | n-gram models with add-alpha and Kneser–Ney smoothing; a sampler; an evaluation harness that measures perplexity; word vectors from counts (PPMI and SVD) and from word2vec, with nearest-neighbor search; the bigram model as a network, and with its matrix factored through word vectors; a neural language model (Bengio et al., 2003); character-level RNN and LSTM models, with why the plain RNN's gradients vanish (saturated units, a memory test), samples, temperature, and a cell that tracks quotations; a wider LSTM that beats every n-gram model; a word-level LSTM | [in class](notebooks/week02_handson.ipynb) (four short stops), [full](notebooks/week02_language_models.ipynb) (every experiment, for study) |
| 3 | The attention read of a trained English–French translator, written and checked against the trained model (shapes, masking, the worked example of the notes, the stored weights, BLEU); the translator with and without attention, overall and by sentence length; where it reads as it translates, one reordering and one failure; two interventions | [notebook](notebooks/week03_attention.ipynb) |
| 4 | A decoder-only Transformer language model assembled from prepared parts: the attention of all heads at once, and the decoder's forward pass; checked against the hand-worked head of the notes and the trained word model; two tests of the mask (a later token cannot change an earlier prediction; one pass equals one prefix at a time) and a model with a wrong mask that fails them; a short training run; next-word distributions, temperature, and generation | [notebook](notebooks/week04_transformer.ipynb) |
| 5 | Byte-pair encoding, written and checked on the ten words of the notes, then trained on the bytes of stories, with a round trip; one step of Adam, checked by hand and against PyTorch, and gradient descent, momentum, and Adam on a quadratic; packed batches, the learning-rate schedule, and four optimizers (SGD, momentum, AdamW, Muon) on a small model; the notes' base model (3.5 million parameters), its loss in nats per token and bits per byte; the best model size for a budget, from the notes' IsoFLOP runs; sums from a few examples, and the induction head that copies an invented name; a chat template with its loss mask, written, and the base model fine-tuned into a chat model; LoRA, with and without the rows of the new tokens | [notebook](notebooks/week05_pretraining.ipynb) |
| 6 | REINFORCE with and without a baseline on one small problem, with the variability of the gradient estimates | planned |
| 7 | DPO on prepared preferences for the Week 5 chat model; GRPO with verifiable rewards on arithmetic as an extension | planned |
| 8 | Temperature, top-k, and nucleus sampling; a key–value cache with timings; speculative decoding as an extra; best-of-n with a verifier | planned |
| 9 (optional) | A tool-use loop in which the model calls a calculator | planned |

## Principles

- **From scratch.** Week 1 writes its own autograd, so that nothing later is a black box. From Week 2 on the code uses PyTorch tensors and autograd, which work the same way at scale, but no higher-level libraries (`transformers`, `peft`, `trl`).
- **Every component has a check.** Gradients against finite differences and PyTorch, a causal-leakage test for the Transformer, a check of the attention read against a trained model, a round trip for the tokenizer, a distribution test for speculative decoding, and so on. They live in `tests/`.
- **Laptop scale.** Models have up to about ten million parameters and train on small datasets: TinyStories, Tiny Shakespeare, English–French image captions (Multi30k), and synthetic arithmetic.

## Getting started

```
git clone https://github.com/mathrulestheworld/tinylm.git
cd tinylm
python3 -m pip install -e ".[dev]"      # numpy, PyTorch, matplotlib, scikit-learn, plotly, sacrebleu, pytest, jupyter
python3 -m pytest                        # the checks
jupyter lab notebooks/                   # the weekly notebooks
```

PyTorch is optional in Week 1 (it is used only to compare gradients) and required from Week 2 on.

## Layout

```
tinylm/            the package: autograd.py, nn.py, optim.py (Week 1); data.py, ngram.py,
                   embeddings.py, neural.py (Week 2); translator.py
                   (Week 3); transformer.py (Week 4); bpe.py, pretrain.py,
                   finetune.py (Week 5); more each week
tests/             checks for every component
notebooks/         one notebook per week, saved with outputs
tools/             scripts that generate the notebooks and train their checkpoints
checkpoints/       trained models the notebooks load (small ones only)
data/              downloaded datasets (not in the repository)
```

## License

MIT; see [LICENSE](LICENSE).
