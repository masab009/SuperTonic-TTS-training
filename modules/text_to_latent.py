"""Text-to-latent module: maps character-level text + a reference speech latent
to a compressed speech latent via conditional flow matching (Section 3.2).

Components (paper Fig. 1b / Fig. 4, Appendix A.2), with several details corrected
against the real Supertone/supertonic-3 ONNX graphs where they disagreed with the
paper (see training/README.md "Assumptions" for what's ground-truth-verified vs. best-effort):
  - StyleEncoder:  reference latent -> a single style_ttl token sequence
  - TextEncoder:   char ids -> relative-position self-attn -> style-conditioned text embedding
  - UncondMasker:  classifier-free-guidance dropout of text / style conditioning
  - VFEstimator:   noisy latent + t + text_emb + style_ttl -> predicted vector field
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.config import TTLConfig
from training.modules.layers import (
    Attention,
    ConditionCrossAttention,
    ConvNeXtStack,
    RelPosTransformerEncoder,
    SinusoidalTimeEmbedding,
)


class StyleTokenPool(nn.Module):
    """Pools a variable-length reference latent into `n_style` fixed tokens via one
    cross-attention pass with a learnable query bank (ground-truth confirms the
    released model calls this parameter `style_key`, though it's used as the
    attention QUERY here -- the pooled output is later re-projected into separate
    K/V by each consuming ConditionCrossAttention layer).
    """

    def __init__(self, input_dim: int, n_style: int, dim: int, n_heads: int):
        super().__init__()
        self.style_key = nn.Parameter(torch.randn(1, n_style, dim) * 0.02)
        self.pool = ConditionCrossAttention(dim, input_dim, n_heads, hidden=dim, tanh_score=False)

    def forward(self, ref_latent: torch.Tensor, ref_mask: torch.Tensor | None = None) -> torch.Tensor:
        """ref_latent: (B, input_dim, T_ref). Returns style_ttl: (B, n_style, dim)."""
        b = ref_latent.shape[0]
        query = self.style_key.expand(b, -1, -1).transpose(1, 2)  # (B, dim, n_style)
        ctx_mask = ref_mask.squeeze(1) if ref_mask is not None else None
        style_ttl = self.pool(query, ref_latent.transpose(1, 2), ctx_mask=ctx_mask)  # (B, dim, n_style)
        return style_ttl.transpose(1, 2)


class StyleEncoder(nn.Module):
    """Reference encoder (Fig. 4a): compressed reference latent -> style_ttl tokens."""

    def __init__(self, cfg: TTLConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.compressed_dim, cfg.style_dim)
        self.convnext = ConvNeXtStack(
            cfg.style_dim, cfg.style_convnext_interm, cfg.style_convnext_ksz, (1,) * cfg.style_convnext_layers
        )
        self.style_token_layer = StyleTokenPool(cfg.style_dim, cfg.n_style, cfg.style_dim, cfg.style_heads)

    def forward(self, ref_latent: torch.Tensor, ref_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_proj(ref_latent.transpose(1, 2)).transpose(1, 2)
        x = self.convnext(x, ref_mask)
        return self.style_token_layer(x, ref_mask)


class TextEncoder(nn.Module):
    def __init__(self, cfg: TTLConfig, vocab_size: int):
        super().__init__()
        dim = cfg.text_char_emb_dim
        # no padding_idx: masking is fully position-based (text_mask/sequence_mask), and the
        # ported vocab's id 0 is a real (if rare) character, not a reserved pad slot -- see
        # training/README.md "Tokenizer id 0 is not padding" for why padding_idx=0 would be wrong.
        self.embed = nn.Embedding(vocab_size, dim)
        self.convnext = ConvNeXtStack(dim, cfg.text_convnext_interm, cfg.text_convnext_ksz, cfg.text_convnext_dilations)
        self.attn_encoder = RelPosTransformerEncoder(
            dim, cfg.text_attn_filter, cfg.text_attn_heads, cfg.text_attn_layers, cfg.self_attn_window_size
        )
        self.attention1 = ConditionCrossAttention(dim, cfg.style_dim, cfg.speech_prompted_heads, hidden=dim)
        self.attention2 = ConditionCrossAttention(dim, cfg.style_dim, cfg.speech_prompted_heads, hidden=dim)
        self.norm = nn.LayerNorm(dim)
        # "proj_out" has no learned weight in the released graph (text_encoder.onnx's
        # "/text_encoder/proj_out/Mul" is only a mask-multiply, no MatMul feeds it) --
        # convnext.idim == char_emb_dim always here, so an identity projection is exact,
        # not an approximation. Kept as an attribute (rather than removed) so the porter's
        # optional_missing lookup for "proj_out.net.*" documents why it's expected empty.
        self.proj_out = nn.Identity()

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor, style_ttl: torch.Tensor):
        # ground-truth op graph from text_encoder.onnx: the ConvNeXt stack's output is
        # added back in as a residual AFTER the self-attention block (a skip connection
        # around the whole attn_encoder, at "/text_encoder/Add"), then masked, THEN fed
        # into the two speech_prompted_text_encoder cross-attention layers.
        x = self.embed(text_ids).transpose(1, 2)  # (B, C, T)
        x_cn = self.convnext(x, text_mask)
        x_sa = self.attn_encoder(x_cn, text_mask)
        x = self.proj_out(x_sa + x_cn) * text_mask
        x = x + self.attention1(x, style_ttl)
        x = x + self.attention2(x, style_ttl)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x * text_mask


class UncondMasker(nn.Module):
    """Classifier-free guidance dropout: with prob_both_uncond replace (text, style)
    with learnable null embeddings; with an additional prob_text_uncond replace text only.

    Ground truth (vector_estimator.onnx) has *two* null style tokens
    (`style_key_special_token`, `style_value_special_token`), because the real
    model substitutes differently depending on whether a downstream consumer is
    about to re-project style_ttl into K or into V. This module simplifies that
    to one shared null style token (`style_value_special_token`, matched to the
    ONNX name so the porter can load it) since ConditionCrossAttention derives
    both K and V from the same style_ttl tensor here; the ONNX `style_key_special_token`
    is left unported as a result -- see training/README.md.
    """

    def __init__(self, cfg: TTLConfig, text_dim: int, style_dim: int, prob_both_uncond: float = 0.04):
        super().__init__()
        self.p_text = cfg.p_uncond
        self.p_both = prob_both_uncond
        self.text_special_token = nn.Parameter(torch.randn(1, text_dim, 1) * 0.1)
        self.style_value_special_token = nn.Parameter(torch.randn(1, cfg.n_style, style_dim) * 0.1)

    def forward(self, text_emb: torch.Tensor, style_ttl: torch.Tensor):
        if not self.training:
            return text_emb, style_ttl
        b = text_emb.shape[0]
        r = torch.rand(b, device=text_emb.device)
        both_uncond = (r < self.p_both).view(b, 1, 1)
        text_uncond = ((r >= self.p_both) & (r < self.p_both + self.p_text)).view(b, 1, 1)

        text_emb = torch.where(both_uncond | text_uncond, self.text_special_token.expand(b, -1, text_emb.shape[-1]), text_emb)
        style_ttl = torch.where(both_uncond, self.style_value_special_token.expand(b, -1, -1), style_ttl)
        return text_emb, style_ttl


class VFBlock(nn.Module):
    """One repeated block of the VF estimator (Fig. 4c): dilated ConvNeXt -> time
    conditioning -> ConvNeXt -> text cross-attn (rotary; best-effort, see README) ->
    ConvNeXt -> style cross-attn (tanh-bounded, ground-truth verified).
    """

    def __init__(self, cfg: TTLConfig):
        super().__init__()
        dim = cfg.vf_hdim
        self.dilated_convnext = ConvNeXtStack(dim, cfg.vf_interm, cfg.vf_ksz, cfg.vf_dilated_rates)
        self.time_linear = nn.Linear(cfg.time_dim, dim)
        self.convnext_1 = ConvNeXtStack(dim, cfg.vf_interm, cfg.vf_ksz, (1,) * cfg.vf_extra_convnext_per_block)
        self.attn = Attention(dim, cfg.text_char_emb_dim, cfg.vf_text_heads, n_units=dim, rotary_base=cfg.vf_rotary_base)
        self.attn_norm = nn.LayerNorm(dim)
        self.convnext_2 = ConvNeXtStack(dim, cfg.vf_interm, cfg.vf_ksz, (1,) * cfg.vf_extra_convnext_per_block)
        self.attention = ConditionCrossAttention(dim, cfg.style_dim, cfg.vf_style_heads, hidden=cfg.style_dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.rotary_scale = cfg.vf_rotary_scale

    def forward(self, x, t_emb, text_emb, style_ttl, latent_mask, text_mask):
        x = self.dilated_convnext(x, latent_mask)
        x = x + self.time_linear(t_emb).unsqueeze(-1)
        x = x * latent_mask
        x = self.convnext_1(x, latent_mask)

        y = self.attn(x, text_emb, mask=text_mask, q_pos_scale=1.0, k_pos_scale=1.0 / self.rotary_scale)
        x = self.attn_norm((x + y).transpose(1, 2)).transpose(1, 2) * latent_mask
        x = self.convnext_2(x, latent_mask)

        y = self.attention(x, style_ttl)
        x = self.attention_norm((x + y).transpose(1, 2)).transpose(1, 2)
        return x * latent_mask


class VFEstimator(nn.Module):
    def __init__(self, cfg: TTLConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(cfg.compressed_dim, cfg.vf_hdim)
        self.time_encoder = SinusoidalTimeEmbedding(cfg.time_dim, cfg.time_hdim)
        self.main_blocks = nn.ModuleList([VFBlock(cfg) for _ in range(cfg.vf_n_blocks)])
        self.last_convnext = ConvNeXtStack(cfg.vf_hdim, cfg.vf_interm, cfg.vf_ksz, (1,) * cfg.vf_final_convnext_layers)
        self.proj_out = nn.Linear(cfg.vf_hdim, cfg.compressed_dim)

    def forward(self, noisy_latent, t, text_emb, style_ttl, latent_mask, text_mask):
        x = self.proj_in(noisy_latent.transpose(1, 2)).transpose(1, 2) * latent_mask
        t_emb = self.time_encoder(t)
        for block in self.main_blocks:
            x = block(x, t_emb, text_emb, style_ttl, latent_mask, text_mask)
        x = self.last_convnext(x, latent_mask)
        x = self.proj_out(x.transpose(1, 2)).transpose(1, 2)
        return x * latent_mask


class TextToLatentModel(nn.Module):
    def __init__(self, cfg: TTLConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.style_encoder = StyleEncoder(cfg)
        self.text_encoder = TextEncoder(cfg, vocab_size)
        self.uncond_masker = UncondMasker(cfg, cfg.text_char_emb_dim, cfg.style_dim)
        self.vector_field = VFEstimator(cfg)

    def encode_conditions(self, text_ids, text_mask, ref_latent, ref_mask):
        style_ttl = self.style_encoder(ref_latent, ref_mask)
        text_emb = self.text_encoder(text_ids, text_mask, style_ttl)
        text_emb, style_ttl = self.uncond_masker(text_emb, style_ttl)
        return text_emb, style_ttl

    def forward(self, noisy_latent, t, text_ids, text_mask, ref_latent, ref_mask, latent_mask):
        text_emb, style_ttl = self.encode_conditions(text_ids, text_mask, ref_latent, ref_mask)
        return self.vector_field(noisy_latent, t, text_emb, style_ttl, latent_mask, text_mask)


def flow_matching_loss(
    model: TextToLatentModel,
    z1: torch.Tensor,
    latent_mask: torch.Tensor,
    text_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor,
    ref_mask: torch.Tensor,
    ref_time_mask: torch.Tensor,
    n_expand: int = 1,
) -> torch.Tensor:
    """Optimal-transport conditional flow matching loss (Eq. 1), with
    context-sharing batch expansion: conditions (text_emb, style_ttl) are encoded
    once and reused across `n_expand` independent noise/timestep draws.
    """
    cfg = model.cfg
    text_emb, style_ttl = model.encode_conditions(text_ids, text_mask, ref_latent, ref_mask)

    b, c, t = z1.shape
    z1_e = z1.repeat_interleave(n_expand, dim=0)
    mask_e = latent_mask.repeat_interleave(n_expand, dim=0)
    ref_time_mask_e = ref_time_mask.repeat_interleave(n_expand, dim=0)
    text_emb_e = text_emb.repeat_interleave(n_expand, dim=0)
    text_mask_e = text_mask.repeat_interleave(n_expand, dim=0)
    style_ttl_e = style_ttl.repeat_interleave(n_expand, dim=0)

    z0 = torch.randn_like(z1_e)
    t_ = torch.rand(b * n_expand, device=z1.device)
    t_bc = t_.view(-1, 1, 1)
    sigma_min = cfg.sigma_min
    zt = (1 - (1 - sigma_min) * t_bc) * z0 + t_bc * z1_e
    target = z1_e - (1 - sigma_min) * z0

    pred = model.vector_field(zt, t_, text_emb_e, style_ttl_e, mask_e, text_mask_e)

    # exclude the reference-speech crop region from the loss to prevent information leakage
    loss_mask = mask_e * (1 - ref_time_mask_e)
    diff = (pred - target).abs() * loss_mask
    denom = loss_mask.sum().clamp_min(1.0) * c
    return diff.sum() / denom
