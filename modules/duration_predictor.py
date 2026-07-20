"""Utterance-level duration predictor (Section 3.3, Appendix A.3): predicts the
total duration of the synthesized speech directly from text + a reference speech
latent, avoiding phoneme-level duration alignment entirely.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.config import DPConfig
from training.modules.layers import ConvNeXtStack, RelPosTransformerEncoder, StyleTokenLayer


class DPTextEncoder(nn.Module):
    """Sentence encoder: char ids -> prepended learnable "sentence token" -> ConvNeXt
    + relative-position self-attention -> the sentence token's output is the
    fixed-size text embedding (Appendix A.3.2; self-attention scheme corrected
    per the ground-truth ONNX graph, see training/README.md).
    """

    def __init__(self, cfg: DPConfig, vocab_size: int):
        super().__init__()
        dim = cfg.char_emb_dim
        # see TextEncoder in text_to_latent.py: no padding_idx, ported vocab id 0 is a real char
        self.embed = nn.Embedding(vocab_size, dim)
        self.sentence_token = nn.Parameter(torch.randn(1, dim, 1) * 0.02)
        self.convnext = ConvNeXtStack(dim, cfg.convnext_interm, cfg.convnext_ksz, (1,) * cfg.convnext_layers)
        self.attn_encoder = RelPosTransformerEncoder(dim, cfg.attn_filter, cfg.attn_heads, cfg.attn_layers)
        self.proj_out = nn.Linear(dim, dim, bias=False)  # ground truth: duration_predictor.onnx has no bias here

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        # ground truth (duration_predictor.onnx "/sentence_encoder/Add"): same outer
        # residual as TextEncoder in text_to_latent.py -- the ConvNeXt stack's output is
        # added back in after the self-attention block, then position 0 (the sentence
        # token) is sliced out and proj_out'd.
        b = text_ids.shape[0]
        x = self.embed(text_ids).transpose(1, 2)  # (B, C, T)
        token = self.sentence_token.expand(b, -1, -1)  # (B, C, 1)
        x = torch.cat([token, x], dim=2)
        mask = torch.cat([torch.ones(b, 1, 1, device=text_mask.device), text_mask], dim=2)
        x_cn = self.convnext(x, mask)
        x_sa = self.attn_encoder(x_cn, mask)
        utt_out = (x_sa + x_cn)[:, :, 0]
        return self.proj_out(utt_out)


class DPStyleEncoder(nn.Module):
    """Reference encoder for the duration predictor: produces a single flattened
    reference embedding by concatenating `n_style` pooled tokens along the channel dim.
    """

    def __init__(self, cfg: DPConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.compressed_dim, cfg.style_dim)
        self.convnext = ConvNeXtStack(cfg.style_dim, cfg.style_convnext_interm, 5, (1,) * cfg.style_convnext_layers)
        self.style_tokens = StyleTokenLayer(
            input_dim=cfg.style_dim,
            n_style=cfg.n_style,
            style_key_dim=0,
            style_value_dim=cfg.n_style * cfg.style_value_dim,
            prototype_dim=cfg.style_dim,
            n_units=cfg.style_dim,
            n_heads=cfg.style_heads,
            flatten_output=True,
        )

    def forward(self, ref_latent: torch.Tensor, ref_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_proj(ref_latent.transpose(1, 2)).transpose(1, 2)
        x = self.convnext(x, ref_mask)
        _, style_value = self.style_tokens(x, ref_mask)
        return style_value  # (B, n_style * style_value_dim)


class DurationPredictor(nn.Module):
    def __init__(self, cfg: DPConfig, vocab_size: int):
        super().__init__()
        self.text_encoder = DPTextEncoder(cfg, vocab_size)
        self.style_encoder = DPStyleEncoder(cfg)
        in_dim = cfg.char_emb_dim + cfg.n_style * cfg.style_value_dim
        self.estimator = nn.Sequential(
            nn.Linear(in_dim, cfg.predictor_hdim), nn.PReLU(), nn.Linear(cfg.predictor_hdim, 1)
        )

    def forward(self, text_ids, text_mask, ref_latent, ref_mask=None) -> torch.Tensor:
        # ground truth (duration_predictor.onnx /predictor/Concat_2 + /predictor/Exp):
        # text embedding comes first in the concat (not style/ref), and the estimator
        # predicts LOG-duration -- exponentiated here to return actual seconds, matching
        # what train_duration_predictor.py's target_duration and duration_loss expect.
        text_emb = self.text_encoder(text_ids, text_mask)
        ref_emb = self.style_encoder(ref_latent, ref_mask)
        x = torch.cat([text_emb, ref_emb], dim=-1)
        log_duration = self.estimator(x).squeeze(-1)
        return log_duration.exp()


def duration_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()
