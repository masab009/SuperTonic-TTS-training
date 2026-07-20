from pathlib import Path

import pytest

from training.config import load_model_config

REPO_ROOT = Path(__file__).resolve().parents[2]
TTS_JSON = REPO_ROOT / "supertonic-3-model" / "onnx" / "tts.json"


def test_defaults_match_paper_scale():
    cfg = load_model_config(None)
    assert cfg.ae.ldim == 24
    assert cfg.ttl.chunk_compress_factor == 6
    assert cfg.ttl.compressed_dim == 24 * 6
    assert cfg.ae.enc_idim == cfg.ae.n_mels + cfg.ae.n_fft // 2 + 1  # use_linear_spec defaults True


@pytest.mark.skipif(not TTS_JSON.exists(), reason="supertonic-3-model/onnx/tts.json not present")
def test_load_real_tts_json_is_self_consistent():
    cfg = load_model_config(str(TTS_JSON))
    # dims that must agree with the compressed-latent contract used throughout the codebase
    assert cfg.ttl.compressed_dim == cfg.ttl.latent_dim * cfg.ttl.chunk_compress_factor
    assert cfg.dp.compressed_dim == cfg.dp.latent_dim * cfg.dp.chunk_compress_factor
    assert cfg.ae.enc_idim > cfg.ae.n_mels or not cfg.ae.use_linear_spec
    # sanity bounds -- these should be real, plausible released-model dimensions, not leftover defaults
    assert 0 < cfg.ttl.n_style <= 256
    assert 0 < cfg.dp.n_style <= 256
    assert cfg.vocab_size >= 256
