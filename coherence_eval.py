"""Objective progress check for the text-to-latent generator during fine-tuning.

For the first N eval sentences, synthesize speech (each sentence's own real clip as
the style reference and its real duration, so we compare like-for-like against the
ground-truth recording), then measure two alignment-free "is this articulate speech"
proxies against the real clip:
  - spectral flatness  (~0 = structured speech/harmonics, ~1 = white noise/buzz)
  - syllable modulation in the 2-10 Hz band (higher = clearer speech rhythm)
and one content proxy:
  - mel log-energy envelope correlation vs the real clip (needs matching duration).

None of these is a substitute for listening, but together they track whether the
generator is converging toward speech, and the saved wavs let you spot-check by ear.

    python -m training.coherence_eval \
        --eval_filelist data/eval.txt --root_dir data/wavs \
        --config supertonic-3-model/onnx/tts.json \
        --tokenizer runs/ported/tokenizer_ur.json \
        --autoencoder_ckpt runs/ae_ft/ckpt_stage1_fitted.pt \
        --ttl_ckpt runs/ttl_ft/ckpt_10000.pt \
        --n 4 --out_dir tts_outputs/eval_10k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from training.ckpt_utils import load_state_dict_grow_vocab
from training.config import load_model_config
from training.datasets import load_audio, load_filelist
from training.latent_utils import decompress_and_denormalize, normalize_and_compress
from training.modules.autoencoder import SpeechAutoencoder
from training.modules.text_to_latent import TextToLatentModel
from training.text import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_filelist", required=True)
    p.add_argument("--root_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--autoencoder_ckpt", required=True)
    p.add_argument("--ttl_ckpt", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--out_dir", default="tts_outputs/eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def speech_metrics(x: np.ndarray, sr: int):
    w = torch.hann_window(1024)
    f = torch.stft(torch.tensor(x), n_fft=1024, hop_length=256, window=w, return_complex=True)
    P = (f.abs() ** 2 + 1e-10).numpy()
    gm = np.exp(np.mean(np.log(P), axis=0)); am = np.mean(P, axis=0)
    flatness = float(np.mean(gm / am))
    e = np.log(am); e = e - e.mean()
    fr = sr / 256
    E = np.abs(np.fft.rfft(e)); mf = np.fft.rfftfreq(len(e), 1 / fr)
    band = (mf >= 2) & (mf <= 10)
    mod = float(E[band].max() / (E.max() + 1e-9))
    return flatness, mod, am


def envelope_corr(am_a, am_b):
    n = min(len(am_a), len(am_b))
    a = np.log(am_a[:n]); b = np.log(am_b[:n])
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_model_config(args.config)
    tok = CharTokenizer.from_dict(__import__("json").loads(Path(args.tokenizer).read_text()))

    ae = SpeechAutoencoder(cfg.ae).to(device)
    ae.load_state_dict(torch.load(args.autoencoder_ckpt, map_location=device)["generator"]); ae.eval()
    ttl = TextToLatentModel(cfg.ttl, tok.vocab_size).to(device)
    load_state_dict_grow_vocab(ttl, torch.load(args.ttl_ckpt, map_location=device)["model"]); ttl.eval()

    entries = load_filelist(args.eval_filelist)[: args.n]
    flats, mods, corrs = [], [], []
    for i, (wav_rel, text, lang) in enumerate(entries):
        real, sr = load_audio(str(Path(args.root_dir) / wav_rel))
        if sr != cfg.ae.sample_rate:
            real = torchaudio.functional.resample(real, sr, cfg.ae.sample_rate)
        dur = real.shape[-1] / cfg.ae.sample_rate

        raw_ref = ae.encoder(real.unsqueeze(0).to(device))
        ref_z = normalize_and_compress(ae, raw_ref, cfg.ttl.chunk_compress_factor, cfg.ttl.normalizer_scale)
        ref_mask = torch.ones(1, 1, ref_z.shape[-1], device=device)
        text_ids = torch.tensor([tok.encode(text, lang)], dtype=torch.long, device=device)
        text_mask = torch.ones(1, 1, text_ids.shape[1], device=device)
        text_emb, style_ttl = ttl.encode_conditions(text_ids, text_mask, ref_z, ref_mask)

        t_comp = max(1, (int(dur * cfg.ae.sample_rate / cfg.ae.hop_length) + 1) // cfg.ttl.chunk_compress_factor)
        latent_mask = torch.ones(1, 1, t_comp, device=device)
        zt = torch.randn(1, cfg.ttl.compressed_dim, t_comp, device=device)
        dt = 1.0 / args.steps
        for s in range(args.steps):
            t = torch.full((1,), s * dt, device=device)
            zt = zt + ttl.vector_field(zt, t, text_emb, style_ttl, latent_mask, text_mask) * dt
        raw_gen = decompress_and_denormalize(ae, zt, cfg.ttl.chunk_compress_factor, cfg.ae.ldim, cfg.ttl.normalizer_scale)
        syn = ae.decode(raw_gen).cpu().squeeze(0).numpy()
        peak = max(abs(syn.max()), abs(syn.min()))
        if peak > 1e-6:
            syn = syn * (0.95 / peak)
        sf.write(out_dir / f"eval_{i}.wav", syn, cfg.ae.sample_rate)

        fl, md, am_syn = speech_metrics(syn, cfg.ae.sample_rate)
        _, _, am_real = speech_metrics(real.numpy(), cfg.ae.sample_rate)
        cr = envelope_corr(am_syn, am_real)
        flats.append(fl); mods.append(md); corrs.append(cr)
        print(f"  eval_{i}: flatness={fl:.4f} syllable-mod={md:.3f} env-corr-vs-real={cr:.3f}  ({dur:.1f}s)")

    print(f"\nMEAN over {len(entries)}: flatness={np.mean(flats):.4f} "
          f"syllable-mod={np.mean(mods):.3f} env-corr={np.mean(corrs):.3f}")
    print("(reference targets: real speech syllable-mod ~0.97, env-corr of copy-synth ~0.97)")


if __name__ == "__main__":
    main()
