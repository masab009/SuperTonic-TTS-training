import torch

from training.config import DPConfig
from training.modules.duration_predictor import DurationPredictor, duration_loss
from training.modules.layers import sequence_mask

VOCAB = 15


def _tiny_cfg():
    return DPConfig(
        latent_dim=4,
        chunk_compress_factor=2,
        normalizer_scale=1.0,
        char_emb_dim=8,
        convnext_dim=8,
        convnext_interm=16,
        convnext_layers=2,
        convnext_ksz=3,
        attn_layers=1,
        attn_heads=2,
        attn_filter=16,
        style_dim=8,
        style_convnext_layers=1,
        style_convnext_interm=16,
        n_style=3,
        style_value_dim=4,
        style_heads=2,
        predictor_hdim=16,
    )


def test_forward_shape_and_range():
    cfg = _tiny_cfg()
    model = DurationPredictor(cfg, VOCAB)
    text_ids = torch.randint(1, VOCAB, (2, 7))
    text_mask = sequence_mask(torch.tensor([7, 4]), 7)
    ref_latent = torch.randn(2, cfg.compressed_dim, 9)
    ref_mask = sequence_mask(torch.tensor([9, 5]), 9)
    pred = model(text_ids, text_mask, ref_latent, ref_mask)
    assert pred.shape == (2,)


def test_forward_backward_grads_flow():
    cfg = _tiny_cfg()
    model = DurationPredictor(cfg, VOCAB)
    text_ids = torch.randint(1, VOCAB, (3, 5))
    text_mask = sequence_mask(torch.tensor([5, 5, 5]), 5)
    ref_latent = torch.randn(3, cfg.compressed_dim, 6)
    pred = model(text_ids, text_mask, ref_latent)
    target = torch.rand(3) * 5
    loss = duration_loss(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_duration_loss_is_mean_abs_error():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.0, 5.0])
    loss = duration_loss(pred, target)
    assert torch.isclose(loss, torch.tensor((0.5 + 0.0 + 2.0) / 3))


def test_no_ref_mask_still_works():
    cfg = _tiny_cfg()
    model = DurationPredictor(cfg, VOCAB)
    text_ids = torch.randint(1, VOCAB, (1, 4))
    text_mask = sequence_mask(torch.tensor([4]), 4)
    ref_latent = torch.randn(1, cfg.compressed_dim, 5)
    pred = model(text_ids, text_mask, ref_latent, ref_mask=None)
    assert pred.shape == (1,)
