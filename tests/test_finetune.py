"""Checks of the Week 5 fine-tuning code (tinylm/finetune.py): the chat template and its mask, LoRA, the rows of
the new tokens, and the loop."""
import pathlib

import torch

from tinylm.bpe import load_tokenizer
from tinylm.finetune import (render_solution, prompt_ids, to_example, assistant_loss, LoRALinear, add_lora,
                             train_rows, merge_lora, trained_parameters, finetune, words_used)
from tinylm.pretrain import fast_attention
from tinylm.transformer import TransformerLM

CKPT = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"
TOK = load_tokenizer(CKPT / "week05_tokenizer.json")
EOT, SYS, USR, AST, END = 4088, 4089, 4090, 4091, 4092
CONV = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Bye."},
        {"role": "assistant", "content": "Bye now."}]


def test_the_template_and_its_mask():
    ids, mask = render_solution(CONV, TOK)
    assert [i for i in ids if i >= EOT] == [EOT, SYS, END, USR, END, AST, END, USR, END, AST, END]
    assert sum(mask) == len(TOK.encode("Hello.")) + 1 + len(TOK.encode("Bye now.")) + 1
    assert all(mask[k] == 0 for k, i in enumerate(ids) if i in (EOT, SYS, USR, AST))
    in_loss = [i for i, m in zip(ids, mask) if m]
    assert TOK.decode([i for i in in_loss if i < EOT]) == "Hello.Bye now." and in_loss.count(END) == 2
    assert prompt_ids(CONV[:2], TOK) == render_solution(CONV[:2], TOK)[0] + [AST]


def test_examples_leave_out_what_is_not_in_the_loss():
    ids, mask = render_solution(CONV, TOK)
    x, y = to_example(ids, mask)
    assert x == ids[:-1] and len(y) == len(x)
    assert [t for t in y if t != -100] == [i for i, m in zip(ids[1:], mask[1:]) if m]


def small_model():
    torch.manual_seed(0)
    m = TransformerLM(4096, d_model=16, n_layers=1, n_heads=2, max_len=64)
    m.set_attention(fast_attention)
    return m


def test_lora_starts_at_the_base_and_merges_exactly():
    torch.manual_seed(0)
    base = torch.nn.Linear(6, 5)
    lora = LoRALinear(base, 2)
    x = torch.randn(3, 6)
    assert torch.allclose(lora(x), base(x))                      # B starts at zero
    with torch.no_grad():
        lora.B.normal_()
    assert torch.allclose(lora.merged()(x), lora(x), atol=1e-6)
    assert not base.weight.requires_grad and lora.A.requires_grad


def test_lora_trains_only_its_matrices_and_the_chosen_rows():
    model = add_lora(small_model(), 4)
    train_rows(model, [SYS, USR, AST, END])
    per_block = 4 * 4 * (16 + 16) + 2 * 4 * (16 + 64)
    assert trained_parameters(model, rows=4) == per_block + 4 * 16
    before = model.tok.weight.detach().clone()
    exs = [to_example(*render_solution(CONV, TOK))] * 4
    finetune(model, exs, epochs=2, batch_size=2, peak=1e-2, weight_decay=0.0, log=None)
    changed = (model.tok.weight.detach() != before).any(1).nonzero().flatten().tolist()
    assert changed == [SYS, USR, AST, END]
    out = model(torch.tensor([exs[0][0]]))
    merge_lora(model)
    assert torch.allclose(model(torch.tensor([exs[0][0]])), out, atol=1e-4)


def test_finetuning_lowers_the_loss_on_the_assistants_tokens():
    model = small_model()
    exs = [to_example(*render_solution(CONV, TOK))] * 8
    before = assistant_loss(model, exs)
    curve = finetune(model, exs, epochs=30, batch_size=4, peak=1e-2, log=None)
    assert assistant_loss(model, exs) < 0.25 * before and len(curve) >= 2


def test_words_used():
    assert words_used("The dog jumped over the log.", "jump, log, cat") == (2, 3)
