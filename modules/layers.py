"""Shared building blocks used by the speech autoencoder, text-to-latent module,
and duration predictor: ConvNeXt blocks (Vocos-style), rotary self/cross-attention,
and the NANSY++-style style-token pooling layer.

All sequence tensors use the (B, C, T) "channels-first" layout at module boundaries,
matching the ONNX graph I/O of the released model (text_ids, style_ttl, noisy_latent, ...).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """lengths: (B,) -> mask: (B, 1, max_len), 1 for valid, 0 for padding."""
    max_len = max_len or int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    mask = (ids.unsqueeze(0) < lengths.unsqueeze(1)).float()
    return mask.unsqueeze(1)


class ConvNeXtBlock1D(nn.Module):
    """Vocos-style ConvNeXt block operating on (B, C, T).

    depthwise conv -> LayerNorm -> pointwise up -> GELU -> pointwise down -> layerscale -> residual.
    `causal=True` left-pads only (no future leakage), used by the streaming latent decoder.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        causal: bool = False,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) * dilation
        self.left_pad = pad if causal else pad // 2
        self.right_pad = 0 if causal else pad - pad // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, groups=dim, dilation=dilation, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_pad, self.right_pad))
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # (B, C, T)
        return residual + x


class ConvNeXtStack(nn.Module):
    def __init__(self, dim, intermediate_dim, kernel_size, dilations, causal=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock1D(dim, intermediate_dim, kernel_size, d, causal=causal)
                for d in dilations
            ]
        )

    def forward(self, x, mask: torch.Tensor | None = None):
        for block in self.blocks:
            x = block(x)
            if mask is not None:
                x = x * mask
        return x


class RotaryEmbedding(nn.Module):
    """Standard RoPE, with an optional per-call position `scale` so that sequences
    running at different effective frame rates (e.g. characters vs. compressed
    latent frames) can be placed on comparable position units in cross-attention.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, scale: float = 1.0, device=None, dtype=None):
        t = torch.arange(seq_len, device=device, dtype=torch.float32) / scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D_head), cos/sin: (T, D_head)
    return x * cos.unsqueeze(0).unsqueeze(0) + _rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)


class Attention(nn.Module):
    """Multi-head attention over (B, C, T) tensors. Supports self-attention
    (q_dim == kv_dim, same sequence) and cross-attention (separate q/kv sources),
    with optional RoPE applied independently to query and key position axes.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        n_heads: int,
        n_units: int | None = None,
        rotary_base: float | None = None,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        n_units = n_units or q_dim
        assert n_units % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_units // n_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(q_dim, n_units)
        self.to_k = nn.Linear(kv_dim, n_units)
        self.to_v = nn.Linear(kv_dim, n_units)
        self.to_out = nn.Linear(n_units, q_dim)
        self.dropout = nn.Dropout(p_dropout)

        self.rotary = RotaryEmbedding(self.head_dim, rotary_base) if rotary_base else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        mask: torch.Tensor | None = None,
        q_pos_scale: float = 1.0,
        k_pos_scale: float = 1.0,
    ) -> torch.Tensor:
        """x_q, x_kv: (B, C, T). mask: (B, 1, T_kv) with 1=valid. Returns (B, C, T_q)."""
        q = self.to_q(x_q.transpose(1, 2))
        k = self.to_k(x_kv.transpose(1, 2))
        v = self.to_v(x_kv.transpose(1, 2))

        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        if self.rotary is not None:
            cos_q, sin_q = self.rotary(q.shape[2], q_pos_scale, q.device, q.dtype)
            cos_k, sin_k = self.rotary(k.shape[2], k_pos_scale, k.device, k.dtype)
            q = apply_rotary(q, cos_q, sin_q)
            k = apply_rotary(k, cos_k, sin_k)

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros_like(mask, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(mask < 0.5, float("-inf"))
            attn_bias = attn_bias.unsqueeze(1)  # (B, 1, 1, T_kv)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(x_q.shape[0], -1, self.n_heads * self.head_dim)
        out = self.to_out(out)
        return out.transpose(1, 2)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + residual, transformer-encoder style, used by the
    text encoder / DP text encoder ("attn_encoder" in the released config)."""

    def __init__(self, dim: int, n_heads: int, filter_channels: int, p_dropout: float = 0.0, rotary_base: float = 10000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, dim, n_heads, rotary_base=rotary_base, p_dropout=p_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, filter_channels), nn.GELU(), nn.Dropout(p_dropout), nn.Linear(filter_channels, dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.attn(h, h, mask=mask)
        h = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.ff(h.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            x = x * mask
        return x


class StyleTokenLayer(nn.Module):
    """NANSY++-style timbre token block, used to pool a variable-length reference
    latent sequence into a fixed-size set of `n_style` (key, value) tokens via two
    independent multi-head-attention-pooling passes with learnable seed queries.

    When `flatten_output=True` (duration predictor's reference encoder), the
    per-token value outputs are instead concatenated along the channel dim into a
    single fixed-size vector, matching Appendix A.3.1's "stacking" description.
    """

    def __init__(
        self,
        input_dim: int,
        n_style: int,
        style_key_dim: int,
        style_value_dim: int,
        prototype_dim: int,
        n_units: int,
        n_heads: int,
        flatten_output: bool = False,
    ):
        super().__init__()
        self.n_style = n_style
        self.flatten_output = flatten_output
        self.key_query = nn.Parameter(torch.randn(1, n_style, prototype_dim) * 0.02)
        self.value_query = nn.Parameter(torch.randn(1, n_style, prototype_dim) * 0.02)
        key_out = (style_key_dim // n_style) if flatten_output and style_key_dim else style_key_dim
        val_out = (style_value_dim // n_style) if flatten_output else style_value_dim
        self.key_attn = Attention(prototype_dim, input_dim, n_heads, n_units=n_units, p_dropout=0.0)
        self.key_proj = nn.Linear(prototype_dim, key_out) if key_out else None
        self.value_attn = Attention(prototype_dim, input_dim, n_heads, n_units=n_units, p_dropout=0.0)
        self.value_proj = nn.Linear(prototype_dim, val_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: (B, C, T) reference latent features. Returns (style_key, style_value).
        Sequence mode: each is (B, n_style, dim). Flattened mode: each is (B, dim) or None for key.
        """
        b = x.shape[0]
        q_k = self.key_query.expand(b, -1, -1).transpose(1, 2)
        q_v = self.value_query.expand(b, -1, -1).transpose(1, 2)

        style_key = None
        if self.key_proj is not None:
            style_key = self.key_attn(q_k, x, mask=mask).transpose(1, 2)
            style_key = self.key_proj(style_key)
        style_value = self.value_attn(q_v, x, mask=mask).transpose(1, 2)
        style_value = self.value_proj(style_value)

        if self.flatten_output:
            style_value = style_value.reshape(b, -1)
            style_key = style_key.reshape(b, -1) if style_key is not None else None
        return style_key, style_value


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)
