import torch

from training.config import AEConfig
from training.latent_utils import (
    compress_latent,
    decompress_and_denormalize,
    decompress_latent,
    normalize_and_compress,
    sample_reference_crop,
)
from training.modules.autoencoder import SpeechAutoencoder


def test_compress_decompress_round_trip():
    latent = torch.randn(2, 4, 12)
    compressed = compress_latent(latent, kc=3)
    assert compressed.shape == (2, 12, 4)
    restored = decompress_latent(compressed, kc=3, ldim=4)
    assert torch.allclose(restored, latent)


def test_compress_trims_to_multiple_of_kc():
    latent = torch.randn(1, 2, 10)  # 10 not divisible by kc=3
    compressed = compress_latent(latent, kc=3)
    assert compressed.shape == (1, 6, 3)


def _tiny_ae_cfg():
    return AEConfig(
        sample_rate=8000,
        n_fft=256,
        win_length=256,
        hop_length=64,
        n_mels=16,
        use_linear_spec=False,
        hdim=8,
        intermediate_dim=16,
        ldim=4,
        enc_num_layers=1,
        dec_num_layers=1,
        dec_dilations=(1,),
        dec_head_hdim=8,
    )


def test_normalize_and_compress_round_trip():
    ae = SpeechAutoencoder(_tiny_ae_cfg())
    ae.fit_latent_stats([torch.randn(2, 4, 50)])
    raw = torch.randn(2, 4, 18)
    scale = 0.25
    compressed = normalize_and_compress(ae, raw, kc=3, scale=scale)
    assert compressed.shape == (2, 12, 6)
    restored = decompress_and_denormalize(ae, compressed, kc=3, ldim=4, scale=scale)
    assert torch.allclose(restored, raw, atol=1e-5)


def test_sample_reference_crop_shapes_and_bounds():
    torch.manual_seed(0)
    z1 = torch.randn(3, 4, 40)
    lengths = torch.tensor([40, 20, 5])
    ref_latent, ref_mask, ref_time_mask = sample_reference_crop(z1, lengths, frame_rate=14.0, min_dur=0.2, max_dur=9.0)
    assert ref_latent.shape[0] == 3
    assert ref_latent.shape[1] == 4
    assert ref_mask.shape[0] == 3
    assert ref_time_mask.shape == (3, 1, 40)
    # crop length never exceeds half the utterance length (except the min_frames floor)
    for i in range(3):
        crop_len = int(ref_mask[i, 0].sum().item())
        length_i = int(lengths[i].item())
        assert crop_len <= max(length_i, 1)
        assert ref_time_mask[i, 0].sum().item() == crop_len

    # every reference frame must actually be a copy of the source latent at its recorded position
    for i in range(3):
        marked = (ref_time_mask[i, 0] > 0).nonzero(as_tuple=True)[0]
        if len(marked) == 0:
            continue
        start = marked.min().item()
        crop_len = int(ref_mask[i, 0].sum().item())
        assert torch.allclose(ref_latent[i, :, :crop_len], z1[i, :, start : start + crop_len])


def test_sample_reference_crop_very_short_utterance():
    z1 = torch.randn(1, 2, 3)
    lengths = torch.tensor([3])
    ref_latent, ref_mask, ref_time_mask = sample_reference_crop(z1, lengths, frame_rate=14.0)
    assert ref_latent.shape[0] == 1
    assert ref_mask.sum().item() >= 1
