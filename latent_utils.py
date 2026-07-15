"""Temporal latent compression (Section 3.2.1) and running channel-wise
normalization statistics, shared by the text-to-latent module and duration predictor.
"""
from __future__ import annotations

import torch

from training.modules.layers import sequence_mask


def compress_latent(latent: torch.Tensor, kc: int) -> torch.Tensor:
    """(B, C, T) -> (B, C*kc, T//kc). Groups `kc` consecutive frames into the channel
    dim; perfectly invertible by `decompress_latent`. Right-trims T to a multiple of kc.
    """
    b, c, t = latent.shape
    t_trim = (t // kc) * kc
    latent = latent[..., :t_trim]
    latent = latent.view(b, c, t_trim // kc, kc)
    latent = latent.permute(0, 1, 3, 2).reshape(b, c * kc, t_trim // kc)
    return latent


def decompress_latent(latent: torch.Tensor, kc: int, ldim: int) -> torch.Tensor:
    """Inverse of `compress_latent`: (B, C*kc, T) -> (B, C, T*kc)."""
    b, ck, t = latent.shape
    latent = latent.view(b, ldim, kc, t)
    latent = latent.permute(0, 1, 3, 2).reshape(b, ldim, t * kc)
    return latent


class ChannelNormalizer(torch.nn.Module):
    """Precomputed channel-wise mean/std normalization with an extra global scale
    factor (`ttl.normalizer.scale` / `dp.normalizer.scale` in the released config).
    Call `fit` once on a sample of training latents before training starts.
    """

    def __init__(self, num_channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.register_buffer("mean", torch.zeros(1, num_channels, 1))
        self.register_buffer("std", torch.ones(1, num_channels, 1))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, latents: list[torch.Tensor]) -> None:
        # each latent: (B, C, T) -> (C, B*T), pooled across batches to get per-channel stats
        flat = torch.cat([lat.transpose(0, 1).reshape(lat.shape[1], -1) for lat in latents], dim=1)
        self.mean.copy_(flat.mean(dim=1).view(1, -1, 1))
        self.std.copy_(flat.std(dim=1).clamp_min(1e-5).view(1, -1, 1))
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std * self.scale

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale * self.std + self.mean


def sample_reference_crop(
    z1: torch.Tensor,
    lengths: torch.Tensor,
    frame_rate: float,
    min_dur: float = 0.2,
    max_dur: float = 9.0,
):
    """Crop a random per-sample reference segment out of each utterance's own
    compressed latent (Section 3.2.4): duration in [min_dur, max_dur] seconds and
    never more than half the utterance's own length. Returns:
      ref_latent:    (B, C, T_ref)  zero-padded to the batch max crop length
      ref_mask:      (B, 1, T_ref)  1 for valid (non-padded) reference frames
      ref_time_mask: (B, 1, T)      1 at positions of z1 that were used as the
                                     reference crop (to exclude from the FM loss)
    """
    b, c, t = z1.shape
    device = z1.device
    min_frames = max(1, int(round(min_dur * frame_rate)))
    starts = torch.zeros(b, dtype=torch.long, device=device)
    crop_lens = torch.zeros(b, dtype=torch.long, device=device)
    for i in range(b):
        length_i = int(lengths[i].item())
        max_frames_i = max(min_frames, min(int(round(max_dur * frame_rate)), length_i // 2))
        crop_len = min_frames if max_frames_i <= min_frames else torch.randint(min_frames, max_frames_i + 1, (1,)).item()
        crop_len = min(crop_len, max(length_i, 1))
        start = 0 if length_i - crop_len <= 0 else torch.randint(0, length_i - crop_len + 1, (1,)).item()
        starts[i] = start
        crop_lens[i] = crop_len

    t_ref = int(crop_lens.max().item())
    ref_latent = z1.new_zeros(b, c, t_ref)
    ref_time_mask = z1.new_zeros(b, 1, t)
    for i in range(b):
        s, l = int(starts[i]), int(crop_lens[i])
        ref_latent[i, :, :l] = z1[i, :, s : s + l]
        ref_time_mask[i, :, s : s + l] = 1.0

    ref_mask = sequence_mask(crop_lens, t_ref)
    return ref_latent, ref_mask, ref_time_mask
