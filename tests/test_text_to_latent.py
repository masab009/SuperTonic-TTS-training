import torch

from training.config import TTLConfig
from training.modules.layers import sequence_mask
from training.modules.text_to_latent import TextToLatentModel, UncondMasker, flow_matching_loss


def _tiny_cfg():
    return TTLConfig(
        latent_dim=4,
        chunk_compress_factor=2,
        normalizer_scale=0.25,
        n_batch_expand=2,
        p_uncond=0.05,
        text_char_emb_dim=8,
        text_convnext_dim=8,
        text_convnext_interm=16,
        text_convnext_layers=2,
        text_convnext_ksz=3,
        text_convnext_dilations=(1, 1),
        text_attn_layers=1,
        text_attn_heads=2,
        text_attn_filter=16,
        style_dim=8,
        style_convnext_layers=1,
        style_convnext_interm=16,
        style_convnext_ksz=3,
        n_style=4,
        style_heads=2,
        speech_prompted_heads=2,
        vf_style_heads=2,
        self_attn_window_size=2,
        vf_hdim=8,
        vf_interm=16,
        vf_ksz=3,
        vf_n_blocks=2,
        vf_dilated_layers=2,
        vf_dilated_rates=(1, 2),
        vf_extra_convnext_per_block=1,
        vf_final_convnext_layers=1,
        vf_text_heads=2,
        vf_rotary_base=10000.0,
        vf_rotary_scale=10.0,
        time_dim=8,
    )


VOCAB = 20


def _batch(cfg, b=2, t_latent=10, t_text=6, t_ref=4):
    text_ids = torch.randint(1, VOCAB, (b, t_text))
    text_mask = sequence_mask(torch.full((b,), t_text), t_text)
    z1 = torch.randn(b, cfg.compressed_dim, t_latent)
    latent_mask = sequence_mask(torch.full((b,), t_latent), t_latent)
    ref_latent = torch.randn(b, cfg.compressed_dim, t_ref)
    ref_mask = sequence_mask(torch.full((b,), t_ref), t_ref)
    ref_time_mask = torch.zeros(b, 1, t_latent)
    ref_time_mask[:, :, :t_ref] = 1.0
    return text_ids, text_mask, z1, latent_mask, ref_latent, ref_mask, ref_time_mask


def test_forward_shape():
    cfg = _tiny_cfg()
    model = TextToLatentModel(cfg, VOCAB)
    text_ids, text_mask, z1, latent_mask, ref_latent, ref_mask, _ = _batch(cfg)
    noisy = torch.randn_like(z1)
    t = torch.rand(z1.shape[0])
    out = model(noisy, t, text_ids, text_mask, ref_latent, ref_mask, latent_mask)
    assert out.shape == z1.shape


def test_flow_matching_loss_finite_and_backward():
    cfg = _tiny_cfg()
    model = TextToLatentModel(cfg, VOCAB)
    model.train()
    text_ids, text_mask, z1, latent_mask, ref_latent, ref_mask, ref_time_mask = _batch(cfg)
    loss = flow_matching_loss(
        model, z1, latent_mask, text_ids, text_mask, ref_latent, ref_mask, ref_time_mask, n_expand=cfg.n_batch_expand
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_flow_matching_loss_masks_reference_region():
    """Reference-crop timesteps must be entirely excluded from the loss."""
    cfg = _tiny_cfg()
    model = TextToLatentModel(cfg, VOCAB)
    model.train()
    text_ids, text_mask, z1, latent_mask, ref_latent, ref_mask, _ = _batch(cfg)
    t_latent = z1.shape[-1]
    full_ref_time_mask = torch.ones(z1.shape[0], 1, t_latent)  # entire utterance is "reference"
    loss = flow_matching_loss(
        model, z1, latent_mask, text_ids, text_mask, ref_latent, ref_mask, full_ref_time_mask, n_expand=1
    )
    # with loss_mask entirely zeroed out, denom clamps to 1 and the numerator is 0
    assert loss.item() == 0.0


def test_uncond_masker_noop_in_eval():
    cfg = _tiny_cfg()
    masker = UncondMasker(cfg, text_dim=8, style_dim=8)
    masker.eval()
    text_emb = torch.randn(2, 8, 5)
    style_ttl = torch.randn(2, 4, 8)
    out_text, out_style = masker(text_emb, style_ttl)
    assert torch.equal(out_text, text_emb)
    assert torch.equal(out_style, style_ttl)


def test_uncond_masker_replaces_in_train():
    torch.manual_seed(3)
    cfg = TTLConfig(p_uncond=1.0, n_style=4)  # force text_uncond branch always (p_both defaults to 0.04)
    masker = UncondMasker(cfg, text_dim=8, style_dim=8, prob_both_uncond=0.0)
    masker.train()
    text_emb = torch.randn(5, 8, 6)
    style_ttl = torch.randn(5, 4, 8)
    out_text, out_style = masker(text_emb, style_ttl)
    # p_both=0, p_text=1.0 -> text always replaced, style never replaced
    assert torch.allclose(out_text, masker.text_special_token.expand(5, -1, 6))
    assert torch.equal(out_style, style_ttl)


def test_encode_conditions_shapes():
    cfg = _tiny_cfg()
    model = TextToLatentModel(cfg, VOCAB)
    text_ids, text_mask, _, _, ref_latent, ref_mask, _ = _batch(cfg)
    text_emb, style_ttl = model.encode_conditions(text_ids, text_mask, ref_latent, ref_mask)
    assert text_emb.shape == (2, cfg.text_char_emb_dim, text_ids.shape[1])
    assert style_ttl.shape == (2, cfg.n_style, cfg.style_dim)
