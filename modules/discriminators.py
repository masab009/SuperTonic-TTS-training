"""GAN discriminators for the speech autoencoder: a lightweight multi-period
discriminator (HiFi-GAN style, Appendix A.1.3) and a multi-resolution
discriminator operating on log-linear spectrograms (Table 7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

MPD_PERIODS = (2, 3, 5, 7, 11)
MPD_CHANNELS = (16, 64, 256, 512, 512)
MRD_FFT_SIZES = (512, 1024, 2048)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chans = (1,) + MPD_CHANNELS
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(chans[i], chans[i + 1], (5, 1), (3, 1) if i < len(chans) - 2 else (1, 1), padding=(2, 0)))
                for i in range(len(chans) - 1)
            ]
        )
        self.out_conv = weight_norm(nn.Conv2d(MPD_CHANNELS[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        b, t = x.shape
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = x.view(b, 1, -1, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.out_conv(x)
        feats.append(x)
        return x.flatten(1), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=MPD_PERIODS):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor):
        outs, feats = [], []
        for disc in self.discs:
            o, f = disc(x)
            outs.append(o)
            feats.append(f)
        return outs, feats


class ResolutionDiscriminator(nn.Module):
    """Operates on a log-magnitude linear spectrogram, per Table 7."""

    def __init__(self, n_fft: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.win_length = n_fft
        chans = [(1, 16, (2, 1)), (16, 16, (2, 1)), (16, 16, (2, 1)), (16, 16, (1, 1)), (16, 16, (1, 1))]
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(ci, co, (5, 5), stride=s, padding=(2, 2))) for ci, co, s in chans]
        )
        self.out_conv = weight_norm(nn.Conv2d(16, 1, (3, 3), 1, padding=(1, 1)))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=x.device)
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length, window=window, center=True, return_complex=True
        )
        return torch.log(spec.abs().clamp_min(1e-5))

    def forward(self, x: torch.Tensor):
        x = self.spectrogram(x).unsqueeze(1)  # (B, 1, F, T)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.out_conv(x)
        feats.append(x)
        return x.flatten(1), feats


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, fft_sizes=MRD_FFT_SIZES):
        super().__init__()
        self.discs = nn.ModuleList([ResolutionDiscriminator(n) for n in fft_sizes])

    def forward(self, x: torch.Tensor):
        outs, feats = [], []
        for disc in self.discs:
            o, f = disc(x)
            outs.append(o)
            feats.append(f)
        return outs, feats
