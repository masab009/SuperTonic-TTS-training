
"""Speech autoencoder: mel(+linear)-spectrogram -> low-dimensional continuous
latent -> waveform. Architecture follows Section 3.1 / Appendix A.1 of the
SupertonicTTS paper: a Vocos-style ConvNeXt encoder/decoder with a bottleneck
latent space, and a WaveNeXt-style decoder head that flattens per-frame linear
projections directly into the time-domain waveform (no ISTFT / upsampling convs).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from training.config import AEConfig
from training.modules.layers import ConvNeXtBlock1D


class SpecProcessor(nn.Module):
    """Log mel-spectrogram, optionally concatenated with the log linear-magnitude
    spectrogram (the released model's encoder idim = n_mels + n_fft//2+1 implies
    this concatenation; see tts.json `ae.encoder.idim` vs `spec_processor.n_mels`).
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=1.0,
            center=True,
        )
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length, power=1.0, center=True
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T_samples) -> (B, idim, T_frames)"""
        mel = torch.log(self.mel(wav).clamp_min(1e-5))
        if not self.cfg.use_linear_spec:
            return mel
        lin = torch.log(self.spec(wav).clamp_min(1e-5))
        return torch.cat([mel, lin], dim=1)


class LatentEncoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.spec = SpecProcessor(cfg)
        self.in_pad = cfg.enc_ksz // 2
        self.in_conv = nn.Conv1d(cfg.enc_idim, cfg.hdim, cfg.enc_ksz, padding=0)
        self.in_bn = nn.BatchNorm1d(cfg.hdim)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock1D(cfg.hdim, cfg.intermediate_dim, cfg.enc_ksz) for _ in range(cfg.enc_num_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.hdim)
        self.out_proj = nn.Linear(cfg.hdim, cfg.ldim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.spec(wav)
        return self.encode_from_spec(x)

    def encode_from_spec(self, spec_feat: torch.Tensor) -> torch.Tensor:
        # edge/replicate padding, matching the decoder's ground-truth-confirmed convention
        # (see LatentDecoder / ConvNeXtBlock1D) -- the encoder itself isn't public to verify
        # directly, but shares the same Vocos-derived architecture family throughout.
        x = F.pad(spec_feat, (self.in_pad, self.in_pad), mode="replicate")
        x = self.in_bn(self.in_conv(x))
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x = self.out_proj(self.out_norm(x))
        return x.transpose(1, 2)  # (B, ldim, T)


class LatentDecoder(nn.Module):
    """Causal (streaming-capable) decoder. The head flattens `hop_length`-wide
    per-frame outputs directly into the waveform (WaveNeXt trick): odim == hop_length.
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.in_ksz = cfg.dec_ksz
        self.in_conv = nn.Conv1d(cfg.ldim, cfg.hdim, cfg.dec_ksz, padding=0)
        # ground truth (vocoder.onnx): no separate BatchNorm node after the embed conv --
        # it's fused into in_conv's weights at export time (BN in eval mode is just a
        # per-channel affine transform, algebraically foldable into the preceding conv).
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock1D(cfg.hdim, cfg.intermediate_dim, cfg.dec_ksz, dilation=d, causal=True)
                for d in cfg.dec_dilations
            ]
        )
        self.out_bn = nn.BatchNorm1d(cfg.hdim)
        self.head_conv = nn.Conv1d(cfg.hdim, cfg.dec_head_hdim, 3, padding=0)  # causal, left-pad only
        self.head_act = nn.PReLU()  # single shared parameter, matches the released model
        self.head_proj = nn.Linear(cfg.dec_head_hdim, cfg.hop_length, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # ground truth (vocoder.onnx): "/decoder/embed/Pad" and "/decoder/head/layer1/Pad"
        # both use edge/replicate padding, matching every dwconv Pad in the same graph.
        x = F.pad(latent, (self.in_ksz - 1, 0), mode="replicate")
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_bn(x)
        x = F.pad(x, (2, 0), mode="replicate")
        x = self.head_conv(x)
        x = self.head_act(x)
        x = self.head_proj(x.transpose(1, 2))  # (B, T, hop_length)
        wav = x.reshape(x.shape[0], -1)  # flatten frames into a single waveform
        return wav


class SpeechAutoencoder(nn.Module):
    """Owns `latent_mean`/`latent_std`: per-channel statistics of the raw
    (uncompressed, 24-dim) latent space, shared by both downstream modules
    (ground truth: `ae.latent_mean`/`ae.latent_std` in vocoder.onnx). The
    text-to-latent module and duration predictor normalize with these plus one
    extra scalar multiplier of their own (`ttl.normalizer.scale` / `dp.normalizer.scale`)
    applied *after* temporal compression -- see `training/latent_utils.py`.
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = LatentEncoder(cfg)
        self.decoder = LatentDecoder(cfg)
        self.register_buffer("latent_mean", torch.zeros(1, cfg.ldim, 1))
        self.register_buffer("latent_std", torch.ones(1, cfg.ldim, 1))
        self.register_buffer("latent_stats_fitted", torch.tensor(False))

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(wav)
        recon = self.decoder(latent)
        min_len = min(wav.shape[-1], recon.shape[-1])
        return recon[..., :min_len], latent

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.encoder(wav)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    @torch.no_grad()
    def fit_latent_stats(self, latents: list[torch.Tensor]) -> None:
        flat = torch.cat([lat.transpose(0, 1).reshape(lat.shape[1], -1) for lat in latents], dim=1)
        self.latent_mean.copy_(flat.mean(dim=1).view(1, -1, 1))
        self.latent_std.copy_(flat.std(dim=1).clamp_min(1e-5).view(1, -1, 1))
        self.latent_stats_fitted.fill_(True)

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std + self.latent_mean
