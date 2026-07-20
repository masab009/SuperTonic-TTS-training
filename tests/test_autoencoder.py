import torch

from training.config import AEConfig
from training.modules.autoencoder import SpeechAutoencoder


def _tiny_cfg(use_linear_spec=False):
    return AEConfig(
        sample_rate=8000,
        n_fft=256,
        win_length=256,
        hop_length=64,
        n_mels=16,
        use_linear_spec=use_linear_spec,
        hdim=8,
        intermediate_dim=16,
        ldim=4,
        enc_num_layers=2,
        enc_ksz=5,
        dec_num_layers=2,
        dec_ksz=5,
        dec_dilations=(1, 2),
        dec_head_hdim=8,
    )


def test_enc_idim_property():
    cfg = _tiny_cfg(use_linear_spec=True)
    assert cfg.enc_idim == cfg.n_mels + cfg.n_fft // 2 + 1
    cfg2 = _tiny_cfg(use_linear_spec=False)
    assert cfg2.enc_idim == cfg2.n_mels


def test_autoencoder_forward_shapes():
    cfg = _tiny_cfg()
    ae = SpeechAutoencoder(cfg)
    wav = torch.randn(2, 8000)
    recon, latent = ae(wav)
    assert latent.shape[1] == cfg.ldim
    assert recon.dim() == 2
    assert recon.shape[0] == 2


def test_autoencoder_encode_decode_no_grad():
    cfg = _tiny_cfg()
    ae = SpeechAutoencoder(cfg)
    wav = torch.randn(1, 8000)
    latent = ae.encode(wav)
    assert not latent.requires_grad
    recon = ae.decode(latent)
    assert not recon.requires_grad


def test_autoencoder_forward_backward():
    cfg = _tiny_cfg()
    ae = SpeechAutoencoder(cfg)
    wav = torch.randn(1, 8000, requires_grad=False)
    recon, latent = ae(wav)
    loss = recon.pow(2).mean() + latent.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in ae.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_fit_latent_stats_and_normalize_round_trip():
    cfg = _tiny_cfg()
    ae = SpeechAutoencoder(cfg)
    assert bool(ae.latent_stats_fitted) is False
    latents = [torch.randn(3, cfg.ldim, 20) * 5 + 2 for _ in range(4)]
    ae.fit_latent_stats(latents)
    assert bool(ae.latent_stats_fitted) is True
    raw = torch.randn(2, cfg.ldim, 10)
    normalized = ae.normalize_latent(raw)
    restored = ae.denormalize_latent(normalized)
    assert torch.allclose(restored, raw, atol=1e-5)


def test_decoder_is_causal():
    cfg = _tiny_cfg()
    ae = SpeechAutoencoder(cfg)
    ae.eval()
    latent = torch.randn(1, cfg.ldim, 16)
    with torch.no_grad():
        wav1 = ae.decode(latent)
        latent2 = latent.clone()
        latent2[:, :, 10:] = torch.randn(1, cfg.ldim, 6)
        wav2 = ae.decode(latent2)
    n_frames_unaffected = 10 - cfg.dec_ksz  # conservative bound accounting for receptive field
    prefix_samples = max(n_frames_unaffected, 0) * cfg.hop_length
    if prefix_samples > 0:
        assert torch.allclose(wav1[:, :prefix_samples], wav2[:, :prefix_samples], atol=1e-4)
