"""Diagnostic: isolate the autoencoder from the text-to-latent generator.

Encodes a real clip and decodes it straight back (copy synthesis) -- this is
exactly the path stage-1 training optimized, and it does NOT touch the TTL. If
this reconstruction is clean, the encoder+decoder are fine and any garbage in
`synthesize.py` output comes from the (undertrained) TTL generation. If this
reconstruction is itself buzzing/noise, the stage-1 encoder is the problem.

Also round-trips the latent through the TTL-stage normalize/compress/scale chain
and back, to prove that path is loss-free (invertibility sanity check), and prints
whether the encoder's actual output distribution matches the stored latent stats
the decoder was trained against.

    python -m training.reconstruct \
        --ref_wav .../aud-2024.wav \
        --config supertonic-3-model/onnx/tts.json \
        --autoencoder_ckpt runs/ae_ft/ckpt_5000.pt \
        --out_dir tts_outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from training.config import load_model_config
from training.datasets import load_audio
from training.latent_utils import decompress_and_denormalize, normalize_and_compress
from training.modules.autoencoder import SpeechAutoencoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_wav", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--autoencoder_ckpt", required=True)
    p.add_argument("--out_dir", default="tts_outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_model_config(args.config)
    ae = SpeechAutoencoder(cfg.ae).to(device)
    ckpt = torch.load(args.autoencoder_ckpt, map_location=device)
    ae.load_state_dict(ckpt["generator"])
    ae.eval()
    print(f"latent_stats_fitted flag in ckpt: {bool(ae.latent_stats_fitted)}")

    wav, sr = load_audio(args.ref_wav)
    if sr != cfg.ae.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, cfg.ae.sample_rate)
    wav = wav.unsqueeze(0).to(device)
    print(f"input wav: {wav.shape[-1] / cfg.ae.sample_rate:.2f}s @ {cfg.ae.sample_rate}Hz")

    # --- encode, inspect latent distribution ---
    raw = ae.encoder(wav)  # (1, ldim, T)
    print(f"\nencoder output raw latent: shape={tuple(raw.shape)}")
    print(f"  overall  mean={raw.mean().item():+.4f}  std={raw.std().item():.4f}"
          f"  min={raw.min().item():+.3f}  max={raw.max().item():+.3f}")
    print(f"  stored latent_mean: mean-of-channels={ae.latent_mean.mean().item():+.4f}"
          f"   latent_std: mean-of-channels={ae.latent_std.mean().item():.4f}")
    # per-channel: how far is each channel's actual mean/std from the stored stats?
    ch_mean = raw.mean(dim=(0, 2))
    ch_std = raw.std(dim=(0, 2))
    stored_mean = ae.latent_mean.view(-1)
    stored_std = ae.latent_std.view(-1)
    print(f"  |actual_ch_mean - stored_mean| avg={(ch_mean - stored_mean).abs().mean().item():.4f}"
          f"   |actual_ch_std - stored_std| avg={(ch_std - stored_std).abs().mean().item():.4f}")

    # --- Test A: pure copy synthesis (what stage-1 trained: encoder -> decoder) ---
    recon_a = ae.decoder(raw).cpu().squeeze(0).numpy()
    path_a = out_dir / "recon_A_copysynth.wav"
    sf.write(path_a, recon_a, cfg.ae.sample_rate)
    print(f"\n[A] copy synthesis (encoder->decoder): wrote {path_a} ({len(recon_a)/cfg.ae.sample_rate:.2f}s)")

    # --- Test B: round-trip through the TTL normalization chain, then decode ---
    z = normalize_and_compress(ae, raw, cfg.ttl.chunk_compress_factor, cfg.ttl.normalizer_scale)
    raw2 = decompress_and_denormalize(ae, z, cfg.ttl.chunk_compress_factor, cfg.ae.ldim, cfg.ttl.normalizer_scale)
    n = min(raw.shape[-1], raw2.shape[-1])
    max_diff = (raw[..., :n] - raw2[..., :n]).abs().max().item()
    print(f"[B] normalize/compress/scale round-trip max|diff| vs raw = {max_diff:.2e} (should be ~0)")
    recon_b = ae.decoder(raw2).cpu().squeeze(0).numpy()
    path_b = out_dir / "recon_B_via_ttlnorm.wav"
    sf.write(path_b, recon_b, cfg.ae.sample_rate)
    print(f"    decoded round-tripped latent: wrote {path_b} ({len(recon_b)/cfg.ae.sample_rate:.2f}s)")

    print("\nInterpretation:")
    print("  - If [A] sounds like the original clip -> encoder+decoder are OK; buzzing in")
    print("    synthesize.py is the (undertrained) TTL generator, not the autoencoder.")
    print("  - If [A] is buzzing/noise -> stage-1 encoder is the problem (undertrained or")
    print("    mismatched to the frozen decoder); more/better stage-1 training is required.")


if __name__ == "__main__":
    main()
