"""Stage 2: train the text-to-latent module via conditional flow matching with
context-sharing batch expansion (Section 3.2, Section 4.2). Requires a speech
autoencoder checkpoint from `train_autoencoder.py` (frozen, used only to produce
target latents on the fly).

Example:
    python -m training.train_text_to_latent \
        --filelist data/train.txt --root_dir data/wavs \
        --autoencoder_ckpt runs/autoencoder/ckpt_final.pt --out_dir runs/ttl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.config import load_model_config
from training.datasets import TextAudioDataset, load_filelist, text_audio_collate
from training.latent_utils import ChannelNormalizer, compress_latent, sample_reference_crop
from training.modules.autoencoder import SpeechAutoencoder
from training.modules.layers import sequence_mask
from training.modules.text_to_latent import TextToLatentModel, flow_matching_loss
from training.text import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--filelist", required=True)
    p.add_argument("--root_dir", required=True)
    p.add_argument("--autoencoder_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer", default=None, help="existing tokenizer.json; built from the filelist otherwise")
    p.add_argument("--max_audio_seconds", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr_halve_every", type=int, default=300_000)
    p.add_argument("--iters", type=int, default=700_000)
    p.add_argument("--calibrate_batches", type=int, default=50, help="batches used to fit latent normalizer stats")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_tokenizer(args) -> CharTokenizer:
    if args.tokenizer:
        return CharTokenizer.from_dict(json.loads(Path(args.tokenizer).read_text()))
    entries = load_filelist(args.filelist)
    texts = [e[1] for e in entries]
    langs = [e[2] for e in entries]
    return CharTokenizer.build_from_texts(texts, langs)


@torch.no_grad()
def encode_batch(ae: SpeechAutoencoder, cfg, wav, wav_lengths, device):
    wav = wav.to(device)
    latent = ae.encoder(wav)  # (B, ldim, T)
    latent_lengths = (wav_lengths // cfg.ae.hop_length + 1).clamp(max=latent.shape[-1]).to(device)
    z1 = compress_latent(latent, cfg.ttl.chunk_compress_factor)
    z_lengths = (latent_lengths // cfg.ttl.chunk_compress_factor).clamp(min=1)
    return z1, z_lengths


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_model_config(args.config)
    tokenizer = build_tokenizer(args)
    (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer.to_dict()))

    ae = SpeechAutoencoder(cfg.ae).to(device)
    ae_ckpt = torch.load(args.autoencoder_ckpt, map_location=device)
    ae.load_state_dict(ae_ckpt["generator"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    dataset = TextAudioDataset(args.filelist, args.root_dir, tokenizer, cfg.ae.sample_rate, args.max_audio_seconds)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=text_audio_collate,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = TextToLatentModel(cfg.ttl, tokenizer.vocab_size).to(device)
    normalizer = ChannelNormalizer(cfg.ttl.compressed_dim, cfg.ttl.normalizer_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_halve_every, gamma=0.5)

    frame_rate = cfg.ae.sample_rate / cfg.ae.hop_length / cfg.ttl.chunk_compress_factor

    def infinite(loader):
        while True:
            yield from loader

    data_iter = infinite(loader)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        normalizer.load_state_dict(ckpt["normalizer"])
        step = ckpt["step"]
    else:
        print(f"Calibrating latent normalizer over {args.calibrate_batches} batches...")
        calib_latents = []
        for _ in range(args.calibrate_batches):
            wav, wav_lengths, _, _ = next(data_iter)
            z1, _ = encode_batch(ae, cfg, wav, wav_lengths, device)
            calib_latents.append(z1.cpu())
        normalizer.fit(calib_latents)
        print("Calibration done. mean/std shape:", normalizer.mean.shape)

    model.train()
    while step < args.iters:
        wav, wav_lengths, text_padded, text_lengths = next(data_iter)
        text_padded = text_padded.to(device)
        text_mask = sequence_mask(text_lengths.to(device), text_padded.shape[1])

        z1_raw, z_lengths = encode_batch(ae, cfg, wav, wav_lengths, device)
        z1 = normalizer(z1_raw)
        latent_mask = sequence_mask(z_lengths, z1.shape[-1])
        ref_latent, ref_mask, ref_time_mask = sample_reference_crop(z1, z_lengths, frame_rate)

        loss = flow_matching_loss(
            model,
            z1,
            latent_mask,
            text_padded,
            text_mask,
            ref_latent,
            ref_mask,
            ref_time_mask,
            n_expand=cfg.ttl.n_batch_expand,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            print(f"step {step} | loss {loss.item():.4f} | lr {sched.get_last_lr()[0]:.2e}")
        if step > 0 and step % args.ckpt_every == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "normalizer": normalizer.state_dict(),
                },
                out_dir / f"ckpt_{step}.pt",
            )
        step += 1

    torch.save(
        {"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "normalizer": normalizer.state_dict()},
        out_dir / "ckpt_final.pt",
    )


if __name__ == "__main__":
    main()
