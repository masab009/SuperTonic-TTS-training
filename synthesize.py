"""Inference smoke test: synthesize speech from text + a reference voice clip,
using whatever mix of fine-tuned/ported checkpoints you currently have for each
of the three stages (Euler-sampled conditional flow matching, Section 3.2).

Example (autoencoder + text-to-latent fine-tuned so far, duration predictor
still the un-fine-tuned ported one -- fine for a rough smoke test):
    python -m training.synthesize \
        --text "..." --lang ur \
        --ref_wav audio_samples_dataset-.../wavs/speaker-1/aud-1.wav \
        --config supertonic-3-model/onnx/tts.json \
        --tokenizer runs/ported/tokenizer_ur.json \
        --autoencoder_ckpt runs/ae_ft/ckpt_5000.pt \
        --ttl_ckpt runs/ttl_ft/ckpt_2000.pt \
        --dp_ckpt runs/ported/dp_ported.pt \
        --out_wav sample.wav

Or skip the duration predictor entirely and pick the length yourself:
    ... --duration_seconds 4.5 (no --dp_ckpt)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from training.ckpt_utils import load_state_dict_grow_vocab
from training.config import load_model_config
from training.datasets import load_audio
from training.latent_utils import decompress_and_denormalize, normalize_and_compress
from training.modules.autoencoder import SpeechAutoencoder
from training.modules.duration_predictor import DurationPredictor
from training.modules.text_to_latent import TextToLatentModel
from training.text import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None, help="raw text; prefer --text_file for non-Latin scripts to avoid shell quoting/RTL paste issues")
    p.add_argument("--text_file", default=None, help="UTF-8 file containing the text (whole file content, stripped)")
    p.add_argument("--lang", default="ur")
    p.add_argument(
        "--ref_wav", required=True, help="reference clip: sets the voice/style, and (if --dp_ckpt is given) the duration predictor's reference"
    )
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--config", default=None, help="path to tts.json; defaults to the paper's 44M config")
    p.add_argument("--autoencoder_ckpt", required=True)
    p.add_argument("--ttl_ckpt", required=True)
    p.add_argument("--dp_ckpt", default=None, help="if omitted, --duration_seconds must be given")
    p.add_argument("--duration_seconds", type=float, default=None, help="overrides/bypasses the duration predictor")
    p.add_argument("--out_wav", required=True)
    p.add_argument("--steps", type=int, default=32, help="Euler ODE steps")
    p.add_argument("--peak_normalize", action="store_true", help="scale output to 0.95 peak (the AE reconstructs quiet)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def encode_ref(ae: SpeechAutoencoder, cfg, wav: torch.Tensor, device: torch.device):
    """Returns the full reference clip's compressed+normalized latent at both the
    ttl and dp stage's own scale (they use different normalizer_scale values)."""
    wav = wav.unsqueeze(0).to(device)
    raw_latent = ae.encoder(wav)  # (1, ldim, T)
    z_ttl = normalize_and_compress(ae, raw_latent, cfg.ttl.chunk_compress_factor, cfg.ttl.normalizer_scale)
    z_dp = normalize_and_compress(ae, raw_latent, cfg.dp.chunk_compress_factor, cfg.dp.normalizer_scale)
    return z_ttl, z_dp


@torch.no_grad()
def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    elif args.text is not None:
        text = args.text
    else:
        raise ValueError("either --text or --text_file is required")

    cfg = load_model_config(args.config)
    tokenizer = CharTokenizer.from_dict(json.loads(Path(args.tokenizer).read_text()))

    ae = SpeechAutoencoder(cfg.ae).to(device)
    ae.load_state_dict(torch.load(args.autoencoder_ckpt, map_location=device)["generator"])
    ae.eval()

    ttl = TextToLatentModel(cfg.ttl, tokenizer.vocab_size).to(device)
    load_state_dict_grow_vocab(ttl, torch.load(args.ttl_ckpt, map_location=device)["model"])
    ttl.eval()

    wav, sr = load_audio(args.ref_wav)
    if sr != cfg.ae.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, cfg.ae.sample_rate)
    ref_z_ttl, ref_z_dp = encode_ref(ae, cfg, wav, device)
    ref_mask_ttl = torch.ones(1, 1, ref_z_ttl.shape[-1], device=device)
    ref_mask_dp = torch.ones(1, 1, ref_z_dp.shape[-1], device=device)

    text_ids = torch.tensor([tokenizer.encode(text, args.lang)], dtype=torch.long, device=device)
    text_mask = torch.ones(1, 1, text_ids.shape[1], device=device)

    if args.duration_seconds is not None:
        duration_seconds = args.duration_seconds
    else:
        if not args.dp_ckpt:
            raise ValueError("either --dp_ckpt or --duration_seconds is required")
        dp = DurationPredictor(cfg.dp, tokenizer.vocab_size).to(device)
        load_state_dict_grow_vocab(dp, torch.load(args.dp_ckpt, map_location=device)["model"])
        dp.eval()
        duration_seconds = dp(text_ids, text_mask, ref_z_dp, ref_mask_dp).item()
    print(f"target duration: {duration_seconds:.2f}s")

    raw_latent_len = int(duration_seconds * cfg.ae.sample_rate / cfg.ae.hop_length) + 1
    t_compressed = max(1, raw_latent_len // cfg.ttl.chunk_compress_factor)

    text_emb, style_ttl = ttl.encode_conditions(text_ids, text_mask, ref_z_ttl, ref_mask_ttl)

    latent_mask = torch.ones(1, 1, t_compressed, device=device)
    zt = torch.randn(1, cfg.ttl.compressed_dim, t_compressed, device=device)
    dt = 1.0 / args.steps
    for step in range(args.steps):
        t = torch.full((1,), step * dt, device=device)
        v = ttl.vector_field(zt, t, text_emb, style_ttl, latent_mask, text_mask)
        zt = zt + v * dt

    raw_latent_gen = decompress_and_denormalize(ae, zt, cfg.ttl.chunk_compress_factor, cfg.ae.ldim, cfg.ttl.normalizer_scale)
    wav_out = ae.decode(raw_latent_gen).cpu().squeeze(0).numpy()

    if args.peak_normalize:
        peak = max(abs(wav_out.max()), abs(wav_out.min()))
        if peak > 1e-6:
            wav_out = wav_out * (0.95 / peak)

    Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_wav, wav_out, cfg.ae.sample_rate)
    print(f"wrote {args.out_wav} ({wav_out.shape[-1] / cfg.ae.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
