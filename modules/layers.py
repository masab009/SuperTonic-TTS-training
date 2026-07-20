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

    Ground-truth confirmed against text_encoder.onnx's `convnext.0.*` node graph: masking
    happens THREE times per block (input, post-dwconv, post-residual), and the residual
    branch is the *masked* input, not the raw one -- masking here, not just once per block
    from the outside (as `ConvNeXtStack` used to), matters whenever `mask` excludes any
    position (i.e. always, once real batches have padding).
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x * mask if mask is not None else x
        residual = x
        # ground truth (every graph's dwconv Pad node): edge/replicate padding, not zero
        x = F.pad(x, (self.left_pad, self.right_pad), mode="replicate")
        x = self.dwconv(x)
        if mask is not None:
            x = x * mask
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # (B, C, T)
        out = residual + x
        return out * mask if mask is not None else out


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
            x = block(x, mask)
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

    def cos_sin_from_positions(self, positions: torch.Tensor, dtype=None):
        """positions: (..., T) float rotary positions. Returns cos, sin each (..., T, dim).
        Used for length-aware RoPE, where each sample's positions are normalized by its own
        sequence length so cross-attention between different-length sequences stays diagonal."""
        freqs = positions.unsqueeze(-1) * self.inv_freq.to(device=positions.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D_head); cos/sin either (T, D_head) shared across the batch,
    # or (B, T, D_head) when positions are per-sample (length-aware RoPE).
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    else:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (B,1,T,D)
    return x * cos + _rotate_half(x) * sin


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
        length_aware: bool = False,
        rotary_gamma: float = 1.0,
        q_len: torch.Tensor | None = None,
        k_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_q, x_kv: (B, C, T). mask: (B, 1, T_kv) with 1=valid. Returns (B, C, T_q).

        length_aware=True uses length-normalized RoPE (arXiv:2509.11084): each sample's query
        positions are gamma * m / q_len and key positions gamma * n / k_len, so cross-attention
        between sequences of very different lengths (here ~1.1 latent frames per text token)
        keeps a monotonic diagonal instead of the ~diagonal-destroying absolute-index scaling.
        q_len/k_len are true (unpadded) lengths, (B,); they fall back to the padded length.
        Otherwise q_pos_scale/k_pos_scale apply the legacy absolute-index scaling."""
        q = self.to_q(x_q.transpose(1, 2))
        k = self.to_k(x_kv.transpose(1, 2))
        v = self.to_v(x_kv.transpose(1, 2))

        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        if self.rotary is not None:
            if length_aware:
                tq, tk = q.shape[2], k.shape[2]
                device = q.device
                ql = (q_len if q_len is not None else torch.full((q.shape[0],), tq, device=device)).clamp(min=1).float()
                kl = (k_len if k_len is not None else torch.full((k.shape[0],), tk, device=device)).clamp(min=1).float()
                pos_q = rotary_gamma * torch.arange(tq, device=device).float().unsqueeze(0) / ql.unsqueeze(1)  # (B, Tq)
                pos_k = rotary_gamma * torch.arange(tk, device=device).float().unsqueeze(0) / kl.unsqueeze(1)  # (B, Tk)
                cos_q, sin_q = self.rotary.cos_sin_from_positions(pos_q, q.dtype)
                cos_k, sin_k = self.rotary.cos_sin_from_positions(pos_k, q.dtype)
            else:
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


def _convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    return [item for sublist in reversed(pad_shape) for item in sublist]


class RelativePositionSelfAttention(nn.Module):
    """Windowed relative-position multi-head self-attention (Shaw et al. 2018),
    as implemented in VITS/Glow-TTS's `attentions.py`. Confirmed (not guessed) as
    the released model's actual self-attention mechanism by inspecting the
    `text_encoder.onnx` / `duration_predictor.onnx` graphs: parameters are named
    `conv_q/conv_k/conv_v/conv_o` (Conv1d, kernel 1) plus a pair of learnable
    relative position embeddings `emb_rel_k`/`emb_rel_v` of shape
    (1, 2*window_size+1, head_dim) -- the paper's text says "rotary position
    embedding", but the shipped weights are unambiguously this scheme instead.
    """

    def __init__(self, channels: int, n_heads: int, window_size: int = 4, p_dropout: float = 0.0):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(p_dropout)

        rel_stddev = self.k_channels ** -0.5
        self.emb_rel_k = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.k_channels) * rel_stddev)
        self.emb_rel_v = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.k_channels) * rel_stddev)

    def _get_relative_embeddings(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            emb = F.pad(emb, _convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        return emb[:, start:end]

    def _relative_to_absolute(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, 2L-1) -> (B, H, L, L)
        b, h, length, _ = x.shape
        x = F.pad(x, _convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view(b, h, length * 2 * length)
        x_flat = F.pad(x_flat, _convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view(b, h, length + 1, 2 * length - 1)[:, :, :length, length - 1 :]

    def _absolute_to_relative(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, L) -> (B, H, L, 2L-1)
        b, h, length, _ = x.shape
        x = F.pad(x, _convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view(b, h, length**2 + length * (length - 1))
        x_flat = F.pad(x_flat, _convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view(b, h, length, 2 * length)[:, :, :, 1:]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, C, T). mask: (B, 1, T) with 1=valid. Returns (B, C, T)."""
        b, c, t = x.shape
        q = self.conv_q(x).view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        k = self.conv_k(x).view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        v = self.conv_v(x).view(b, self.n_heads, self.k_channels, t).transpose(2, 3)

        # ground truth (VITS/Glow-TTS MultiHeadAttention.attention): the query is scaled
        # ONCE and reused for both the main and relative-position score matmuls -- scaling
        # only the main scores (leaving rel_logits unscaled) was a real bug, not a harmless
        # reordering: it made the relative-position term sqrt(k_channels) times too large.
        q_scaled = q / math.sqrt(self.k_channels)
        scores = torch.matmul(q_scaled, k.transpose(-2, -1))
        rel_k = self._get_relative_embeddings(self.emb_rel_k, t)
        rel_logits = torch.matmul(q_scaled, rel_k.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._relative_to_absolute(rel_logits)

        if mask is not None:
            key_mask = mask.unsqueeze(1)  # (B, 1, 1, T)
            scores = scores.masked_fill(key_mask < 0.5, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        rel_v = self._get_relative_embeddings(self.emb_rel_v, t)
        rel_weights = self._absolute_to_relative(attn)
        out = torch.matmul(attn, v) + torch.matmul(rel_weights, rel_v.unsqueeze(0))
        out = out.transpose(2, 3).reshape(b, c, t)
        return self.conv_o(out)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a (B, C, T) tensor, wrapped so the
    parameter path is `<name>.norm.weight` (matching the released naming)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class RelPosTransformerEncoder(nn.Module):
    """Self-attention stack matching `<...>.attn_encoder` in the released config:
    alternating relative-position self-attention and a conv-FFN, each with its own
    post-norm (VITS/Glow-TTS `Encoder`, confirmed against text_encoder.onnx's
    `attn_layers` / `norm_layers_1` / `ffn_layers` / `norm_layers_2` parameter names).
    """

    def __init__(self, dim: int, filter_channels: int, n_heads: int, n_layers: int, window_size: int = 4, p_dropout: float = 0.0):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [RelativePositionSelfAttention(dim, n_heads, window_size, p_dropout) for _ in range(n_layers)]
        )
        self.norm_layers_1 = nn.ModuleList([ChannelLayerNorm(dim) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList(
            [nn.ModuleDict({"conv_1": nn.Conv1d(dim, filter_channels, 1), "conv_2": nn.Conv1d(filter_channels, dim, 1)}) for _ in range(n_layers)]
        )
        self.norm_layers_2 = nn.ModuleList([ChannelLayerNorm(dim) for _ in range(n_layers)])
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for attn, norm1, ffn, norm2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            y = attn(x, mask)
            x = norm1(x + self.drop(y))
            # VITS/Glow-TTS FFN: mask before each conv, ReLU (not GELU) -- ground-truth
            # confirmed against text_encoder.onnx's ffn_layers.* node ops (Conv/Relu/Conv,
            # each preceded by a Mul against text_mask).
            x_in = x * mask if mask is not None else x
            y = F.relu(ffn["conv_1"](x_in))
            y = y * mask if mask is not None else y
            y = ffn["conv_2"](y)
            x = norm2(x + self.drop(y))
            if mask is not None:
                x = x * mask
        return x


class XWLinear(nn.Module):
    """Linear layer storing its weight as (in, out) and computing `x @ W + b`,
    matching the released model's projection convention (as opposed to
    PyTorch's `nn.Linear`, which stores (out, in) and computes `x @ W^T + b`).
    Using this convention lets ONNX weights be copied in directly, unmodified.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        # re-parameterize storage as (in, out) instead of nn.Linear's (out, in)
        del self.linear.weight
        self.linear.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.linear.weight
        if self.linear.bias is not None:
            out = out + self.linear.bias
        return out


class ConditionCrossAttention(nn.Module):
    """Cross-attention from a (B, C, T) stream onto a shared fixed-length context
    (e.g. the 50-token style_ttl), independently re-projected into K and V by this
    layer's own W_key/W_value (ground-truth confirmed by tracing
    vector_estimator.onnx's `main_blocks.*.attention` node graph). Two variants
    observed there: `tanh_score=True` bounds the key with tanh before the score
    matmul (used for style/reference conditioning); mask, if given, zeroes
    attention weights *after* softmax rather than biasing logits beforehand
    (also matched from the traced graph, not a standard pre-softmax mask).
    """

    def __init__(self, x_dim: int, ctx_dim: int, n_heads: int, hidden: int | None = None, tanh_score: bool = True):
        super().__init__()
        hidden = hidden or ctx_dim
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden
        self.tanh_score = tanh_score

        self.W_query = XWLinear(x_dim, hidden)
        self.W_key = XWLinear(ctx_dim, hidden)
        self.W_value = XWLinear(ctx_dim, hidden)
        self.out_fc = XWLinear(hidden, x_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, C_x, T). ctx: (B, T_ctx, C_ctx) channels-last. Returns (B, C_x, T)."""
        q = self._split_heads(self.W_query(x.transpose(1, 2)))
        k = self._split_heads(self.W_key(ctx))
        v = self._split_heads(self.W_value(ctx))

        key = torch.tanh(k) if self.tanh_score else k
        # ground truth (vector_estimator.onnx / text_encoder.onnx Div constant): scaled by
        # sqrt(hidden) -- the full pre-split projection width -- not sqrt(head_dim); this
        # module's multi-head split isn't a standard scaled-dot-product-attention scaling.
        scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(self.hidden)
        attn = F.softmax(scores, dim=-1)
        if ctx_mask is not None:
            attn = attn.masked_fill(ctx_mask.view(x.shape[0], 1, 1, -1) < 0.5, 0.0)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(x.shape[0], -1, self.n_heads * self.head_dim)
        out = self.out_fc(out)
        return out.transpose(1, 2)


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
        if key_out:
            self.key_attn = Attention(prototype_dim, input_dim, n_heads, n_units=n_units, p_dropout=0.0)
            self.key_proj = nn.Linear(prototype_dim, key_out)
        else:
            self.key_attn = None
            self.key_proj = None
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
    """Grad-TTS-style time embedding: sinusoidal features at `dim`, expanded to
    `hidden_dim` and projected back down to `dim` (ground-truth confirmed against
    vector_estimator.onnx's `time_encoder.mlp.{0,2}` shapes -- output stays at
    `dim`, NOT `hidden_dim`; each VFBlock's own `time_linear` does the actual
    projection up to the model's channel width).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)
