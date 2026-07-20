import torch

from training.losses import (
    MultiResolutionMelLoss,
    discriminator_lsgan_loss,
    feature_matching_loss,
    generator_lsgan_loss,
)


def test_mel_loss_zero_for_identical_signals():
    torch.manual_seed(0)
    loss_fn = MultiResolutionMelLoss(sample_rate=8000, configs=((256, 16), (512, 32)))
    wav = torch.randn(1, 8000)
    loss = loss_fn(wav, wav)
    assert loss.item() < 1e-5


def test_mel_loss_positive_for_different_signals():
    torch.manual_seed(0)
    loss_fn = MultiResolutionMelLoss(sample_rate=8000, configs=((256, 16),))
    wav1 = torch.randn(1, 8000)
    wav2 = torch.randn(1, 8000)
    loss = loss_fn(wav1, wav2)
    assert loss.item() > 0


def test_discriminator_lsgan_loss_zero_at_optimum():
    real = [torch.ones(4, 1)]
    fake = [-torch.ones(4, 1)]
    loss = discriminator_lsgan_loss(real, fake)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_generator_lsgan_loss_zero_at_optimum():
    fake = [torch.ones(4, 1)]
    loss = generator_lsgan_loss(fake)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_feature_matching_loss_zero_for_identical_features():
    feats = [[torch.randn(2, 3), torch.randn(2, 5)]]
    loss = feature_matching_loss(feats, feats)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_feature_matching_loss_detaches_real():
    real = [[torch.randn(2, 3, requires_grad=True)]]
    fake = [[torch.randn(2, 3, requires_grad=True)]]
    loss = feature_matching_loss(real, fake)
    loss.backward()
    assert fake[0][0].grad is not None
    assert real[0][0].grad is None  # real branch is detached inside the loss
