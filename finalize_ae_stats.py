"""Persist latent_mean/latent_std into an autoencoder checkpoint by refitting them
to the checkpoint's own (fine-tuned) encoder over a sample of real audio.

Intermediate stage-1 checkpoints (ckpt_5000.pt etc.) carry the *ported* model's
original latent stats, which don't describe the newly fine-tuned encoder's output
distribution at all. Stage 2/3 refit them in memory but never save them, so
inference (which reloads the AE checkpoint) denormalizes with the wrong stats and
feeds the decoder mis-scaled latents -> noise. Run this once to bake correct,
consistent stats into a checkpoint used by stage 2, stage 3, AND synthesis.

    python -m training.finalize_ae_stats \
        --in_ckpt runs/ae_ft/ckpt_5000.pt --out_ckpt runs/ae_ft/ckpt_stage1_fitted.pt \
        --filelist data/train.txt --root_dir data/wavs \
        --config supertonic-3-model/onnx/tts.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.config import load_model_config
from training.datasets import TextAudioDataset, load_filelist, text_audio_collate
from training.modules.autoencoder import SpeechAutoencoder
from training.text import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_ckpt", required=True)
    p.add_argument("--out_ckpt", required=True)
    p.add_argument("--filelist", required=True)
    p.add_argument("--root_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_audio_seconds", type=float, default=18.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_model_config(args.config)

    ae = SpeechAutoencoder(cfg.ae).to(device)
    ckpt = torch.load(args.in_ckpt, map_location=device)
    ae.load_state_dict(ckpt["generator"])
    ae.eval()

    # a throwaway tokenizer just to satisfy TextAudioDataset's signature; we ignore text here
    entries = load_filelist(args.filelist)
    tok = CharTokenizer.build_from_texts([e[1] for e in entries[:50]], [e[2] for e in entries[:50]])
    dataset = TextAudioDataset(args.filelist, args.root_dir, tok, cfg.ae.sample_rate, args.max_audio_seconds)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=text_audio_collate, drop_last=True,
    )

    print(f"pre-fit  latent_std mean-of-channels = {ae.latent_std.mean().item():.4f} (fitted={bool(ae.latent_stats_fitted)})")
    latents = []
    it = iter(loader)
    for i in range(args.batches):
        try:
            wav, wav_lengths, _, _ = next(it)
        except StopIteration:
            it = iter(loader)
            wav, wav_lengths, _, _ = next(it)
        raw = ae.encoder(wav.to(device))
        latents.append(raw.cpu())
    ae.fit_latent_stats(latents)
    print(f"post-fit latent_mean mean-of-channels = {ae.latent_mean.mean().item():+.4f}")
    print(f"post-fit latent_std  mean-of-channels = {ae.latent_std.mean().item():.4f} (fitted={bool(ae.latent_stats_fitted)})")

    # keep the rest of the checkpoint intact (mpd/mrd/opt) so it's still resumable if wanted
    ckpt["generator"] = ae.state_dict()
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out_ckpt)
    print(f"wrote {args.out_ckpt}")


if __name__ == "__main__":
    main()
