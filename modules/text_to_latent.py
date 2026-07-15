"""Text-to-latent module: maps character-level text + a reference speech latent
to a compressed speech latent via conditional flow matching (Section 3.2).

Components (paper Fig. 1b / Fig. 4, Appendix A.2):
  - StyleEncoder:  reference latent -> (style_key, style_value) style tokens (NANSY++-style pooling)
  - TextEncoder:   char ids -> ConvNeXt + self-attention -> speaker-adapted text embedding
  - UncondMasker:  classifier-free-guidance dropout of text / style conditioning
  - VFEstimator:   noisy latent + t + text_emb + style tokens -> predicted vector field
  - ContextSharingBatchExpander: reuses encoded conditions across K_e noise/timestep draws
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.config import TTLConfig
from training.modules.layers import (
    Attention,
    ConvNeXtStack,
    SelfAttentionBlock,
    SinusoidalTimeEmbedding,
    StyleTokenLayer,
)


class StyleEncoder(nn.Module):
    """Reference encoder (Fig. 4a): compressed reference latent -> fixed-size style tokens."""

    def __init__(self, cfg: TTLConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.compressed_dim, cfg.style_dim)
        self.convnext = ConvNeXtStack(
            cfg.style_dim, cfg.style_convnext_interm, cfg.style_convnext_ksz, (1,) * cfg.style_convnext_layers
        )
        self.style_tokens = StyleTokenLayer(
            input_dim=cfg.style_dim,
            n_style=cfg.n_style,
            style_key_dim=cfg.style_dim,
            style_value_dim=cfg.style_dim,
            prototype_dim=cfg.style_dim,
            n_units=cfg.style_dim,
            n_heads=cfg.style_heads,
        )

    def forward(self, ref_latent: torch.Tensor, ref_mask: torch.Tensor | None = None):
        """ref_latent: (B, compressed_dim, T_ref) -> style_key, style_value: (B, n_style, style_dim)"""
        x = self.in_proj(ref_latent.transpose(1, 2)).transpose(1, 2)
        x = self.convnext(x, ref_mask)
        return self.style_tokens(x, ref_mask)


class TextEncoder(nn.Module):
    def __init__(self, cfg: TTLConfig, vocab_size: int):
        super().__init__()
        dim = cfg.text_char_emb_dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.convnext = ConvNeXtStack(dim, cfg.text_convnext_interm, cfg.text_convnext_ksz, cfg.text_convnext_dilations)
        self.self_attn = nn.ModuleList(
            [SelfAttentionBlock(dim, cfg.text_attn_heads, cfg.text_attn_filter) for _ in range(cfg.text_attn_layers)]
        )
        self.cross_attn = nn.ModuleList([Attention(dim, cfg.style_dim, cfg.style_heads, n_units=dim) for _ in range(2)])
        self.cross_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor, style_key: torch.Tensor, style_value: torch.Tensor):
        x = self.embed(text_ids).transpose(1, 2)  # (B, C, T)
        x = self.convnext(x, text_mask)
        for attn in self.self_attn:
            x = attn(x, text_mask)
        # two-stage cross attention: query text against style_key, then against style_value
        h = self.cross_norm[0](x.transpose(1, 2)).transpose(1, 2)
        x = x + self.cross_attn[0](h, style_key.transpose(1, 2))
        h = self.cross_norm[1](x.transpose(1, 2)).transpose(1, 2)
        x = x + self.cross_attn[1](h, style_value.transpose(1, 2))
        x = x * text_mask
        return self.proj_out(x.transpose(1, 2)).transpose(1, 2)


class UncondMasker(nn.Module):
    """Classifier-free guidance dropout: with prob_both_uncond replace (text, style)
    with learnable null embeddings; with an additional prob_text_uncond replace text only.
    """

    def __init__(self, cfg: TTLConfig, text_dim: int, style_dim: int, prob_both_uncond: float = 0.04):
        super().__init__()
        self.p_text = cfg.p_uncond
        self.p_both = prob_both_uncond
        self.null_text = nn.Parameter(torch.randn(1, text_dim, 1) * 0.1)
        self.null_style_key = nn.Parameter(torch.randn(1, cfg.n_style, style_dim) * 0.1)
        self.null_style_value = nn.Parameter(torch.randn(1, cfg.n_style, style_dim) * 0.1)

    def forward(self, text_emb, style_key, style_value):
        if not self.training:
            return text_emb, style_key, style_value
        b = text_emb.shape[0]
        r = torch.rand(b, device=text_emb.device)
        both_uncond = (r < self.p_both).view(b, 1, 1)
        text_uncond = ((r >= self.p_both) & (r < self.p_both + self.p_text)).view(b, 1, 1)

        text_emb = torch.where(both_uncond | text_uncond, self.null_text.expand(b, -1, text_emb.shape[-1]), text_emb)
        style_key = torch.where(both_uncond, self.null_style_key.expand(b, -1, -1), style_key)
        style_value = torch.where(both_uncond, self.null_style_value.expand(b, -1, -1), style_value)
        return text_emb, style_key, style_value


class VFBlock(nn.Module):
    """One repeated block of the VF estimator (Fig. 4c): dilated ConvNeXt -> time
    conditioning -> ConvNeXt -> text cross-attn -> ConvNeXt -> style cross-attn.
    """

    def __init__(self, cfg: TTLConfig):
        super().__init__()
        dim = cfg.vf_hdim
        self.dilated_convnext = ConvNeXtStack(dim, cfg.vf_interm, cfg.vf_ksz, cfg.vf_dilated_rates)
        self.time_proj = nn.Linear(dim, dim)
        self.convnext_1 = ConvNeXtStack(dim, cfg.vf_interm, cfg.vf_ksz, (1,) * cfg.vf_extra_convnext_per_block)
        self.text_norm = nn.LayerNorm(dim)
        self.text_attn = Attention(dim, cfg.text_char_emb_dim, cfg.vf_text_heads, n_units=dim, rotary_base=cfg.vf_rotary_base)
        self.convnext_2 = ConvNeXtStack(dim, cfg.vf_interm, cfg.vf_ksz, (1,) * cfg.vf_extra_convnext_per_block)
        self.style_norm = nn.LayerNorm(dim)
        self.style_attn = Attention(dim, cfg.style_dim, cfg.style_heads if hasattr(cfg, "style_heads") else 4, n_units=dim)
        self.rotary_scale = cfg.vf_rotary_scale

    def forward(self, x, t_emb, text_emb, style_value, latent_mask, text_mask):
        x = self.dilated_convnext(x, latent_mask)
        x = x + self.time_proj(t_emb).unsqueeze(-1)
        x = x * latent_mask
        x = self.convnext_1(x, latent_mask)
        h = self.text_norm(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.text_attn(h, text_emb, mask=text_mask, q_pos_scale=1.0, k_pos_scale=1.0 / self.rotary_scale)
        x = x * latent_mask
        x = self.convnext_2(x, latent_mask)
        h = self.style_norm(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.style_attn(h, style_value.transpose(1, 2))
        return x * latent_mask


class VFEstimator(nn.Module):
    def __init__(self, cfg: TTLConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(cfg.compressed_dim, cfg.vf_hdim)
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_dim, cfg.vf_hdim)
        self.blocks = nn.ModuleList([VFBlock(cfg) for _ in range(cfg.vf_n_blocks)])
        self.final_convnext = ConvNeXtStack(cfg.vf_hdim, cfg.vf_interm, cfg.vf_ksz, (1,) * cfg.vf_final_convnext_layers)
        self.proj_out = nn.Linear(cfg.vf_hdim, cfg.compressed_dim)

    def forward(self, noisy_latent, t, text_emb, style_value, latent_mask, text_mask):
        x = self.proj_in(noisy_latent.transpose(1, 2)).transpose(1, 2) * latent_mask
        t_emb = self.time_embed(t)
        for block in self.blocks:
            x = block(x, t_emb, text_emb, style_value, latent_mask, text_mask)
        x = self.final_convnext(x, latent_mask)
        x = self.proj_out(x.transpose(1, 2)).transpose(1, 2)
        return x * latent_mask


class TextToLatentModel(nn.Module):
    def __init__(self, cfg: TTLConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.style_encoder = StyleEncoder(cfg)
        self.text_encoder = TextEncoder(cfg, vocab_size)
        self.uncond_masker = UncondMasker(cfg, cfg.text_char_emb_dim, cfg.style_dim)
        self.vf_estimator = VFEstimator(cfg)

    def encode_conditions(self, text_ids, text_mask, ref_latent, ref_mask):
        style_key, style_value = self.style_encoder(ref_latent, ref_mask)
        text_emb = self.text_encoder(text_ids, text_mask, style_key, style_value)
        text_emb, _, style_value = self.uncond_masker(text_emb, style_key, style_value)
        return text_emb, style_value

    def forward(self, noisy_latent, t, text_ids, text_mask, ref_latent, ref_mask, latent_mask):
        text_emb, style_value = self.encode_conditions(text_ids, text_mask, ref_latent, ref_mask)
        return self.vf_estimator(noisy_latent, t, text_emb, style_value, latent_mask, text_mask)


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
    context-sharing batch expansion: conditions (text_emb, style tokens) are
    encoded once and reused across `n_expand` independent noise/timestep draws.
    """
    cfg = model.cfg
    text_emb, style_value = model.encode_conditions(text_ids, text_mask, ref_latent, ref_mask)

    b, c, t = z1.shape
    z1_e = z1.repeat_interleave(n_expand, dim=0)
    mask_e = latent_mask.repeat_interleave(n_expand, dim=0)
    ref_time_mask_e = ref_time_mask.repeat_interleave(n_expand, dim=0)
    text_emb_e = text_emb.repeat_interleave(n_expand, dim=0)
    text_mask_e = text_mask.repeat_interleave(n_expand, dim=0)
    style_value_e = style_value.repeat_interleave(n_expand, dim=0)

    z0 = torch.randn_like(z1_e)
    t_ = torch.rand(b * n_expand, device=z1.device)
    t_bc = t_.view(-1, 1, 1)
    sigma_min = cfg.sigma_min
    zt = (1 - (1 - sigma_min) * t_bc) * z0 + t_bc * z1_e
    target = z1_e - (1 - sigma_min) * z0

    pred = model.vf_estimator(zt, t_, text_emb_e, style_value_e, mask_e, text_mask_e)

    # exclude the reference-speech crop region from the loss to prevent information leakage
    loss_mask = mask_e * (1 - ref_time_mask_e)
    diff = (pred - target).abs() * loss_mask
    denom = loss_mask.sum().clamp_min(1.0) * c
    return diff.sum() / denom
