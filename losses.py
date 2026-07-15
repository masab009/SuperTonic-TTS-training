"""Loss functions for the speech autoencoder GAN stage (Appendix B.1, Eq. 2-6)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio

MEL_RECON_CONFIGS = ((1024, 64), (2048, 128), (4096, 128))  # (n_fft, n_mels), hop = n_fft // 4


class MultiResolutionMelLoss(torch.nn.Module):
    """Multi-resolution spectral L1 loss (L_recon), FFT sizes 1024/2048/4096 with
    64/128/128 mel bands respectively, hop = n_fft // 4 (Appendix B.1).
    """

    def __init__(self, sample_rate: int, configs=MEL_RECON_CONFIGS):
        super().__init__()
        self.transforms = torch.nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    win_length=n_fft,
                    hop_length=n_fft // 4,
                    n_mels=n_mels,
                    power=1.0,
                    center=True,
                )
                for n_fft, n_mels in configs
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for t in self.transforms:
            mx = torch.log(t(x).clamp_min(1e-5))
            my = torch.log(t(y).clamp_min(1e-5))
            loss = loss + F.l1_loss(mx, my)
        return loss / len(self.transforms)


def discriminator_lsgan_loss(real_outs: list[torch.Tensor], fake_outs: list[torch.Tensor]) -> torch.Tensor:
    """Eq. 3/5: E[(D(x)-1)^2] + E[(D(G(x))+1)^2]"""
    loss = 0.0
    for real, fake in zip(real_outs, fake_outs):
        loss = loss + ((real - 1) ** 2).mean() + ((fake + 1) ** 2).mean()
    return loss / len(real_outs)


def generator_lsgan_loss(fake_outs: list[torch.Tensor]) -> torch.Tensor:
    """Eq. 4: E[(D(G(x))-1)^2]"""
    loss = sum(((fake - 1) ** 2).mean() for fake in fake_outs)
    return loss / len(fake_outs)


def feature_matching_loss(real_feats: list[list[torch.Tensor]], fake_feats: list[list[torch.Tensor]]) -> torch.Tensor:
    """Eq. 6: average L1 distance between discriminator intermediate features."""
    loss = 0.0
    n = 0
    for real_layers, fake_layers in zip(real_feats, fake_feats):
        for r, f in zip(real_layers, fake_layers):
            loss = loss + F.l1_loss(f, r.detach())
            n += 1
    return loss / max(n, 1)
