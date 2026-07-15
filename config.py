"""Default architecture + training hyperparameters for SupertonicTTS.

Architecture dimensions default to the values published in the SupertonicTTS paper
(arXiv:2503.23108, ~44M-parameter research checkpoint, Section 4.2 / Appendix A).
Pass --config path/to/tts.json (the config shipped with Supertone/supertonic-3) to
train at the larger released scale instead -- every dimension below is read from
that file if present, otherwise the paper defaults are used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _get(d: dict, path: str, default):
    node = d
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


@dataclass
class AEConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 228
    use_linear_spec: bool = True  # concat log-linear-magnitude spec with mel (idim = n_mels + n_fft//2+1)
    hdim: int = 512
    intermediate_dim: int = 2048
    ldim: int = 24
    enc_num_layers: int = 10
    enc_ksz: int = 7
    dec_num_layers: int = 10
    dec_ksz: int = 7
    dec_dilations: tuple = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1)
    dec_head_hdim: int = 2048

    @property
    def enc_idim(self) -> int:
        linear_bins = self.n_fft // 2 + 1 if self.use_linear_spec else 0
        return self.n_mels + linear_bins


@dataclass
class TTLConfig:
    latent_dim: int = 24
    chunk_compress_factor: int = 6
    normalizer_scale: float = 0.25
    n_batch_expand: int = 4  # K_e, context-sharing batch expansion factor
    p_uncond: float = 0.05
    sigma_min: float = 1e-8

    text_char_emb_dim: int = 128
    text_convnext_dim: int = 128
    text_convnext_interm: int = 512
    text_convnext_layers: int = 6
    text_convnext_ksz: int = 5
    text_convnext_dilations: tuple = (1, 1, 2, 2, 4, 4)
    text_attn_layers: int = 4
    text_attn_heads: int = 4
    text_attn_filter: int = 512

    style_dim: int = 128
    style_convnext_layers: int = 6
    style_convnext_interm: int = 512
    style_convnext_ksz: int = 5
    n_style: int = 50
    style_heads: int = 2

    vf_hdim: int = 256
    vf_interm: int = 1024
    vf_ksz: int = 5
    vf_n_blocks: int = 4  # N_m
    vf_dilated_layers: int = 4
    vf_dilated_rates: tuple = (1, 2, 4, 8)
    vf_extra_convnext_per_block: int = 1  # convnext_1 / convnext_2 each
    vf_final_convnext_layers: int = 4
    vf_text_heads: int = 8
    vf_rotary_base: float = 10000.0
    vf_rotary_scale: float = 10.0
    time_dim: int = 64

    @property
    def compressed_dim(self) -> int:
        return self.latent_dim * self.chunk_compress_factor


@dataclass
class DPConfig:
    latent_dim: int = 24
    chunk_compress_factor: int = 6
    normalizer_scale: float = 1.0

    char_emb_dim: int = 64
    convnext_dim: int = 64
    convnext_interm: int = 256
    convnext_layers: int = 6
    convnext_ksz: int = 5
    attn_layers: int = 2
    attn_heads: int = 2
    attn_filter: int = 256

    style_dim: int = 64
    style_convnext_layers: int = 4
    style_convnext_interm: int = 256
    n_style: int = 8
    style_value_dim: int = 16
    style_heads: int = 2

    predictor_hdim: int = 128

    @property
    def compressed_dim(self) -> int:
        return self.latent_dim * self.chunk_compress_factor


@dataclass
class ModelConfig:
    ae: AEConfig = field(default_factory=AEConfig)
    ttl: TTLConfig = field(default_factory=TTLConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    vocab_size: int = 65536  # unicode code point range, indexed via unicode_indexer.json


def load_model_config(path: str | None) -> ModelConfig:
    cfg = ModelConfig()
    if not path:
        return cfg
    with open(path) as f:
        raw = json.load(f)

    ae, ttl, dp = cfg.ae, cfg.ttl, cfg.dp
    ae.sample_rate = _get(raw, "ae.sample_rate", ae.sample_rate)
    ae.n_fft = _get(raw, "ae.encoder.spec_processor.n_fft", ae.n_fft)
    ae.hop_length = _get(raw, "ae.encoder.spec_processor.hop_length", ae.hop_length)
    ae.win_length = _get(raw, "ae.encoder.spec_processor.win_length", ae.win_length)
    ae.n_mels = _get(raw, "ae.encoder.spec_processor.n_mels", ae.n_mels)
    ae.hdim = _get(raw, "ae.encoder.hdim", ae.hdim)
    ae.intermediate_dim = _get(raw, "ae.encoder.intermediate_dim", ae.intermediate_dim)
    ae.ldim = _get(raw, "ae.ldim", ae.ldim)
    ae.enc_num_layers = _get(raw, "ae.encoder.num_layers", ae.enc_num_layers)
    ae.enc_ksz = _get(raw, "ae.encoder.ksz", ae.enc_ksz)
    ae.dec_num_layers = _get(raw, "ae.decoder.num_layers", ae.dec_num_layers)
    ae.dec_ksz = _get(raw, "ae.decoder.ksz", ae.dec_ksz)
    ae.dec_dilations = tuple(_get(raw, "ae.decoder.dilation_lst", list(ae.dec_dilations)))
    ae.dec_head_hdim = _get(raw, "ae.decoder.head.hdim", ae.dec_head_hdim)
    measured_idim = _get(raw, "ae.encoder.idim", None)
    if measured_idim is not None:
        ae.use_linear_spec = measured_idim != ae.n_mels

    ttl.latent_dim = _get(raw, "ttl.latent_dim", ttl.latent_dim)
    ttl.chunk_compress_factor = _get(raw, "ttl.chunk_compress_factor", ttl.chunk_compress_factor)
    ttl.normalizer_scale = _get(raw, "ttl.normalizer.scale", ttl.normalizer_scale)
    ttl.n_batch_expand = _get(raw, "ttl.batch_expander.n_batch_expand", ttl.n_batch_expand)
    ttl.sigma_min = _get(raw, "ttl.flow_matching.sig_min", ttl.sigma_min)
    ttl.p_uncond = _get(raw, "ttl.uncond_masker.prob_text_uncond", ttl.p_uncond)

    ttl.text_char_emb_dim = _get(raw, "ttl.text_encoder.text_embedder.char_emb_dim", ttl.text_char_emb_dim)
    ttl.text_convnext_dim = _get(raw, "ttl.text_encoder.convnext.idim", ttl.text_convnext_dim)
    ttl.text_convnext_interm = _get(raw, "ttl.text_encoder.convnext.intermediate_dim", ttl.text_convnext_interm)
    ttl.text_convnext_layers = _get(raw, "ttl.text_encoder.convnext.num_layers", ttl.text_convnext_layers)
    ttl.text_convnext_ksz = _get(raw, "ttl.text_encoder.convnext.ksz", ttl.text_convnext_ksz)
    ttl.text_convnext_dilations = tuple(
        _get(raw, "ttl.text_encoder.convnext.dilation_lst", list(ttl.text_convnext_dilations))
    )
    ttl.text_attn_layers = _get(raw, "ttl.text_encoder.attn_encoder.n_layers", ttl.text_attn_layers)
    ttl.text_attn_heads = _get(raw, "ttl.text_encoder.attn_encoder.n_heads", ttl.text_attn_heads)
    ttl.text_attn_filter = _get(raw, "ttl.text_encoder.attn_encoder.filter_channels", ttl.text_attn_filter)

    ttl.style_dim = _get(raw, "ttl.style_encoder.convnext.idim", ttl.style_dim)
    ttl.style_convnext_layers = _get(raw, "ttl.style_encoder.convnext.num_layers", ttl.style_convnext_layers)
    ttl.style_convnext_interm = _get(raw, "ttl.style_encoder.convnext.intermediate_dim", ttl.style_convnext_interm)
    ttl.n_style = _get(raw, "ttl.style_encoder.style_token_layer.n_style", ttl.n_style)
    ttl.style_heads = _get(raw, "ttl.style_encoder.style_token_layer.n_heads", ttl.style_heads)

    ttl.vf_hdim = _get(raw, "ttl.vector_field.proj_in.odim", ttl.vf_hdim)
    ttl.vf_interm = _get(raw, "ttl.vector_field.main_blocks.convnext_0.intermediate_dim", ttl.vf_interm)
    ttl.vf_ksz = _get(raw, "ttl.vector_field.main_blocks.convnext_0.ksz", ttl.vf_ksz)
    ttl.vf_n_blocks = _get(raw, "ttl.vector_field.main_blocks.n_blocks", ttl.vf_n_blocks)
    ttl.vf_dilated_layers = _get(raw, "ttl.vector_field.main_blocks.convnext_0.num_layers", ttl.vf_dilated_layers)
    ttl.vf_dilated_rates = tuple(_get(raw, "ttl.vector_field.main_blocks.convnext_0.dilation_lst", list(ttl.vf_dilated_rates)))
    ttl.vf_final_convnext_layers = _get(raw, "ttl.vector_field.last_convnext.num_layers", ttl.vf_final_convnext_layers)
    ttl.vf_text_heads = _get(raw, "ttl.vector_field.main_blocks.text_cond_layer.n_heads", ttl.vf_text_heads)
    ttl.vf_rotary_base = _get(raw, "ttl.vector_field.main_blocks.text_cond_layer.rotary_base", ttl.vf_rotary_base)
    ttl.vf_rotary_scale = _get(raw, "ttl.vector_field.main_blocks.text_cond_layer.rotary_scale", ttl.vf_rotary_scale)
    ttl.time_dim = _get(raw, "ttl.vector_field.time_encoder.time_dim", ttl.time_dim)

    dp.latent_dim = _get(raw, "dp.latent_dim", dp.latent_dim)
    dp.chunk_compress_factor = _get(raw, "dp.chunk_compress_factor", dp.chunk_compress_factor)
    dp.normalizer_scale = _get(raw, "dp.normalizer.scale", dp.normalizer_scale)
    dp.char_emb_dim = _get(raw, "dp.sentence_encoder.char_emb_dim", dp.char_emb_dim)
    dp.convnext_dim = _get(raw, "dp.sentence_encoder.convnext.idim", dp.convnext_dim)
    dp.convnext_interm = _get(raw, "dp.sentence_encoder.convnext.intermediate_dim", dp.convnext_interm)
    dp.convnext_layers = _get(raw, "dp.sentence_encoder.convnext.num_layers", dp.convnext_layers)
    dp.attn_layers = _get(raw, "dp.sentence_encoder.attn_encoder.n_layers", dp.attn_layers)
    dp.attn_heads = _get(raw, "dp.sentence_encoder.attn_encoder.n_heads", dp.attn_heads)
    dp.attn_filter = _get(raw, "dp.sentence_encoder.attn_encoder.filter_channels", dp.attn_filter)
    dp.style_dim = _get(raw, "dp.style_encoder.convnext.idim", dp.style_dim)
    dp.style_convnext_layers = _get(raw, "dp.style_encoder.convnext.num_layers", dp.style_convnext_layers)
    dp.n_style = _get(raw, "dp.style_encoder.style_token_layer.n_style", dp.n_style)
    dp.style_value_dim = _get(raw, "dp.style_encoder.style_token_layer.style_value_dim", dp.style_value_dim)
    dp.style_heads = _get(raw, "dp.style_encoder.style_token_layer.n_heads", dp.style_heads)
    dp.predictor_hdim = _get(raw, "dp.predictor.hdim", dp.predictor_hdim)

    return cfg


def save_model_config(cfg: ModelConfig, path: str) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2))
