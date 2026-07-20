import math

import pytest
import torch

from training.modules.layers import (
    Attention,
    ChannelLayerNorm,
    ConditionCrossAttention,
    ConvNeXtBlock1D,
    ConvNeXtStack,
    RelativePositionSelfAttention,
    RelPosTransformerEncoder,
    RotaryEmbedding,
    SinusoidalTimeEmbedding,
    StyleTokenLayer,
    XWLinear,
    apply_rotary,
    sequence_mask,
)


def test_sequence_mask_basic():
    lengths = torch.tensor([3, 1, 5])
    mask = sequence_mask(lengths, max_len=5)
    assert mask.shape == (3, 1, 5)
    assert mask[0, 0].tolist() == [1, 1, 1, 0, 0]
    assert mask[1, 0].tolist() == [1, 0, 0, 0, 0]
    assert mask[2, 0].tolist() == [1, 1, 1, 1, 1]


def test_sequence_mask_infers_max_len():
    lengths = torch.tensor([2, 4])
    mask = sequence_mask(lengths)
    assert mask.shape == (2, 1, 4)


@pytest.mark.parametrize("causal", [False, True])
def test_convnext_block_preserves_shape(causal):
    block = ConvNeXtBlock1D(dim=8, intermediate_dim=16, kernel_size=5, dilation=2, causal=causal)
    x = torch.randn(2, 8, 20)
    y = block(x)
    assert y.shape == x.shape


def test_convnext_block_causal_no_future_leakage():
    torch.manual_seed(0)
    block = ConvNeXtBlock1D(dim=4, intermediate_dim=8, kernel_size=3, dilation=1, causal=True)
    block.eval()
    x = torch.randn(1, 4, 10)
    y1 = block(x)
    x2 = x.clone()
    x2[:, :, 6:] = torch.randn(1, 4, 4)  # perturb only the tail
    y2 = block(x2)
    # outputs at positions before the perturbation must be identical for a causal conv
    assert torch.allclose(y1[:, :, :6], y2[:, :, :6], atol=1e-6)
    assert not torch.allclose(y1[:, :, 6:], y2[:, :, 6:])


def test_convnext_stack_applies_mask():
    stack = ConvNeXtStack(dim=6, intermediate_dim=12, kernel_size=3, dilations=(1, 1))
    x = torch.randn(2, 6, 10)
    mask = sequence_mask(torch.tensor([10, 4]), 10)
    y = stack(x, mask)
    assert torch.all(y[1, :, 4:] == 0)


def test_rotary_embedding_scale_changes_frequencies():
    rot = RotaryEmbedding(dim=8, base=10000.0)
    cos1, sin1 = rot(seq_len=4, scale=1.0)
    cos2, sin2 = rot(seq_len=4, scale=2.0)
    assert cos1.shape == (4, 8)
    assert not torch.allclose(cos1, cos2)
    # position 0 is always angle 0 regardless of scale
    assert torch.allclose(cos1[0], torch.ones(8), atol=1e-6)


def test_apply_rotary_preserves_norm():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 5, 8)
    rot = RotaryEmbedding(dim=8)
    cos, sin = rot(seq_len=5)
    out = apply_rotary(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)


def test_attention_self_attention_shapes_and_grad():
    attn = Attention(q_dim=16, kv_dim=16, n_heads=4, rotary_base=10000.0)
    x = torch.randn(2, 16, 7, requires_grad=True)
    out = attn(x, x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_attention_cross_attention_shapes():
    attn = Attention(q_dim=16, kv_dim=10, n_heads=2)
    q = torch.randn(3, 16, 5)
    kv = torch.randn(3, 10, 9)
    out = attn(q, kv)
    assert out.shape == (3, 16, 5)


def test_attention_mask_blocks_invalid_keys():
    torch.manual_seed(2)
    attn = Attention(q_dim=8, kv_dim=8, n_heads=2)
    attn.eval()
    kv = torch.randn(1, 8, 6)
    mask = sequence_mask(torch.tensor([3]), 6)
    q = torch.randn(1, 8, 1)
    out_masked = attn(q, kv, mask=mask)
    kv2 = kv.clone()
    kv2[:, :, 3:] = torch.randn(1, 8, 3)  # perturb only masked-out positions
    out_masked2 = attn(q, kv2, mask=mask)
    assert torch.allclose(out_masked, out_masked2, atol=1e-5)


def test_relative_position_self_attention_shape_and_grad():
    rpsa = RelativePositionSelfAttention(channels=12, n_heads=3, window_size=2)
    x = torch.randn(2, 12, 9, requires_grad=True)
    mask = sequence_mask(torch.tensor([9, 5]), 9)
    out = rpsa(x, mask)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_relative_position_self_attention_short_sequence():
    # length shorter than window_size + 1 exercises the padding branch
    rpsa = RelativePositionSelfAttention(channels=8, n_heads=2, window_size=4)
    x = torch.randn(1, 8, 2)
    out = rpsa(x)
    assert out.shape == x.shape


def test_rel_pos_transformer_encoder_shape():
    enc = RelPosTransformerEncoder(dim=8, filter_channels=16, n_heads=2, n_layers=2, window_size=2)
    x = torch.randn(2, 8, 11)
    mask = sequence_mask(torch.tensor([11, 6]), 11)
    out = enc(x, mask)
    assert out.shape == x.shape
    assert torch.all(out[1, :, 6:] == 0)


def test_xwlinear_matches_manual_matmul():
    lin = XWLinear(4, 6)
    x = torch.randn(3, 5, 4)
    out = lin(x)
    manual = x @ lin.linear.weight + lin.linear.bias
    assert torch.allclose(out, manual)
    assert lin.linear.weight.shape == (4, 6)  # (in, out), not nn.Linear's (out, in)


def test_condition_cross_attention_shapes_and_masking():
    cca = ConditionCrossAttention(x_dim=16, ctx_dim=8, n_heads=2, hidden=8)
    x = torch.randn(2, 16, 5)
    ctx = torch.randn(2, 6, 8)
    ctx_mask = sequence_mask(torch.tensor([6, 3]), 6).squeeze(1)
    out = cca(x, ctx, ctx_mask=ctx_mask)
    assert out.shape == x.shape


def test_style_token_layer_sequence_mode():
    layer = StyleTokenLayer(
        input_dim=8, n_style=5, style_key_dim=4, style_value_dim=4, prototype_dim=8, n_units=8, n_heads=2
    )
    x = torch.randn(2, 8, 10)
    key, value = layer(x)
    assert key.shape == (2, 5, 4)
    assert value.shape == (2, 5, 4)


def test_style_token_layer_flatten_mode_no_dead_params():
    layer = StyleTokenLayer(
        input_dim=8,
        n_style=5,
        style_key_dim=0,
        style_value_dim=5 * 4,
        prototype_dim=8,
        n_units=8,
        n_heads=2,
        flatten_output=True,
    )
    x = torch.randn(2, 8, 10)
    key, value = layer(x)
    assert key is None
    assert value.shape == (2, 5 * 4)
    assert layer.key_proj is None
    assert layer.key_attn is None
    # the unused key branch must not register as a submodule (no dead trainable params)
    assert "key_attn" not in dict(layer.named_modules())
    assert not any(name.startswith("key_attn") for name, _ in layer.named_parameters())


def test_sinusoidal_time_embedding_shape():
    # Grad-TTS style: output stays at `dim`, hidden_dim is only an internal expansion
    emb = SinusoidalTimeEmbedding(dim=16, hidden_dim=32)
    t = torch.rand(5)
    out = emb(t)
    assert out.shape == (5, 16)
    assert emb.mlp[0].out_features == 32
    assert emb.mlp[2].out_features == 16


def test_channel_layer_norm_shape():
    norm = ChannelLayerNorm(6)
    x = torch.randn(2, 6, 9)
    out = norm(x)
    assert out.shape == x.shape
