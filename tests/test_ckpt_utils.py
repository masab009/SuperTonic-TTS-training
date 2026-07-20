import torch
import torch.nn as nn

from training.ckpt_utils import load_state_dict_grow_vocab


class TinyModel(nn.Module):
    def __init__(self, vocab_size, dim=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)


def test_grow_vocab_copies_overlapping_rows_and_keeps_new_rows_fresh():
    small = TinyModel(vocab_size=5)
    with torch.no_grad():
        small.embed.weight.fill_(1.0)

    big = TinyModel(vocab_size=8)
    with torch.no_grad():
        big.embed.weight.fill_(99.0)
    fresh_new_rows = big.embed.weight[5:8].clone()

    load_state_dict_grow_vocab(big, small.state_dict())

    assert torch.equal(big.embed.weight[:5], small.embed.weight)
    assert torch.equal(big.embed.weight[5:8], fresh_new_rows)
    assert torch.equal(big.proj.weight, small.proj.weight)


def test_grow_vocab_exact_shape_match_is_plain_copy():
    a = TinyModel(vocab_size=6)
    b = TinyModel(vocab_size=6)
    load_state_dict_grow_vocab(b, a.state_dict())
    assert torch.equal(a.embed.weight, b.embed.weight)
    assert torch.equal(a.proj.weight, b.proj.weight)


def test_grow_vocab_rejects_non_dim0_shape_mismatch():
    a = TinyModel(vocab_size=5, dim=4)
    b = TinyModel(vocab_size=5, dim=8)
    try:
        load_state_dict_grow_vocab(b, a.state_dict())
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_grow_vocab_rejects_key_mismatch():
    a = TinyModel(vocab_size=5)

    class OtherModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.other = nn.Linear(4, 4)

    b = OtherModel()
    try:
        load_state_dict_grow_vocab(b, a.state_dict())
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
