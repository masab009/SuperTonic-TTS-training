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


def normalize_and_compress(ae, raw_latent: torch.Tensor, kc: int, scale: float) -> torch.Tensor:
    """Ground-truth normalization order (from vocoder.onnx's `ae.latent_mean`/
    `ae.latent_std` + `ttl.normalizer.scale`): normalize the *raw* 24-dim latent
    with the autoencoder's shared per-channel stats, THEN temporally compress,
    THEN apply the stage's own scalar multiplier. `ae` is a SpeechAutoencoder.
    """
    normalized = ae.normalize_latent(raw_latent)
    compressed = compress_latent(normalized, kc)
    return compressed * scale


def decompress_and_denormalize(ae, compressed: torch.Tensor, kc: int, ldim: int, scale: float) -> torch.Tensor:
    normalized = decompress_latent(compressed / scale, kc, ldim)
    return ae.denormalize_latent(normalized)


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
