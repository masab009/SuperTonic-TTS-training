import torch

from training.modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    PeriodDiscriminator,
    ResolutionDiscriminator,
)


def test_period_discriminator_pads_non_divisible_input():
    disc = PeriodDiscriminator(period=3)
    x = torch.randn(2, 100)  # not divisible by 3
    out, feats = disc(x)
    assert out.dim() == 2
    assert out.shape[0] == 2
    assert len(feats) == len(disc.convs) + 1


def test_multi_period_discriminator_shapes():
    mpd = MultiPeriodDiscriminator(periods=(2, 3))
    x = torch.randn(2, 400)
    outs, feats = mpd(x)
    assert len(outs) == 2
    assert len(feats) == 2


def test_resolution_discriminator_shape():
    disc = ResolutionDiscriminator(n_fft=64)
    x = torch.randn(2, 4000)
    out, feats = disc(x)
    assert out.dim() == 2
    assert len(feats) == len(disc.convs) + 1


def test_multi_resolution_discriminator_shapes():
    mrd = MultiResolutionDiscriminator(fft_sizes=(64, 128))
    x = torch.randn(2, 4000)
    outs, feats = mrd(x)
    assert len(outs) == 2
