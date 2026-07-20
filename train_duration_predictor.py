"""Stage 3: train the utterance-level duration predictor (Section 3.3, Section 4.2).
Requires the same frozen speech autoencoder used in stage 2, plus the tokenizer
produced by `train_text_to_latent.py` (pass --tokenizer to reuse it).

Example:
    python -m training.train_duration_predictor \
        --filelist data/train.txt --root_dir data/wavs \
        --autoencoder_ckpt runs/autoencoder/ckpt_final.pt \
        --tokenizer runs/ttl/tokenizer.json --out_dir runs/dp

Example (fine-tuning Supertone's real, pretrained duration predictor for a new
language -- see training/README.md "Fine-tuning for a new language"):
    python -m training.train_duration_predictor \
        --filelist data/train.txt --root_dir data/wavs \
        --config supertonic-3-model/onnx/tts.json \
        --tokenizer runs/ported/tokenizer.json --autoencoder_ckpt runs/autoencoder_ft/ckpt_final.pt \
        --init_ckpt runs/ported/dp_ported.pt --out_dir runs/dp_ft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.ckpt_utils import load_state_dict_grow_vocab
from training.config import load_model_config
from training.datasets import TextAudioDataset, load_filelist, text_audio_collate
from training.latent_utils import normalize_and_compress
from training.modules.autoencoder import SpeechAutoencoder
from training.modules.duration_predictor import DurationPredictor, duration_loss
from training.modules.layers import sequence_mask
from training.text import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--filelist", required=True)
    p.add_argument("--root_dir", required=True)
    p.add_argument("--autoencoder_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument(
        "--max_audio_seconds",
        type=float,
        default=10.0,
        help="clips longer than this are truncated. IMPORTANT: the duration target is the "
        "(post-truncation) wav length, so any clip longer than this trains a WRONG, too-short "
        "duration. Set it >= your longest utterance so no clip is truncated.",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--calibrate_batches", type=int, default=20)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", default=None, help="full training-state checkpoint to continue an interrupted run")
    p.add_argument(
        "--init_ckpt",
        default=None,
        help="warm-start model weights only (e.g. runs/ported/dp_ported.pt from port_onnx_weights.py); "
        "fresh optimizer, step starts at 0. Ignored if --resume is set. The tokenizer used here must match "
        "the one the checkpoint was ported/trained with (pass --tokenizer runs/ported/tokenizer.json).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sample_dp_reference(z1: torch.Tensor, lengths: torch.Tensor):
    """Reference segment randomly selected within the 5%-95% span of each
    utterance (Section 4.2): a single contiguous crop with both endpoints inside
    that span. Returns ref_latent (B, C, T_ref) and ref_mask (B, 1, T_ref).
    """
    b, c, _ = z1.shape
    device = z1.device
    starts, ends = torch.zeros(b, dtype=torch.long), torch.zeros(b, dtype=torch.long)
    for i in range(b):
        n = max(int(lengths[i].item()), 1)
        lo, hi = int(0.05 * n), max(int(0.95 * n), int(0.05 * n) + 1)
        a = torch.randint(lo, hi, (1,)).item()
        b_ = torch.randint(lo, hi, (1,)).item()
        starts[i], ends[i] = min(a, b_), max(a, b_) + 1
    crop_lens = (ends - starts).clamp(min=1)
    t_ref = int(crop_lens.max().item())
    ref_latent = z1.new_zeros(b, c, t_ref)
    for i in range(b):
        s, l = int(starts[i]), int(crop_lens[i])
        ref_latent[i, :, :l] = z1[i, :, s : s + l]
    ref_mask = sequence_mask(crop_lens.to(device), t_ref)
    return ref_latent, ref_mask


@torch.no_grad()
def encode_batch_raw(ae: SpeechAutoencoder, cfg, wav, wav_lengths, device):
    wav = wav.to(device)
    raw_latent = ae.encoder(wav)
    latent_lengths = (wav_lengths // cfg.ae.hop_length + 1).clamp(max=raw_latent.shape[-1]).to(device)
    return raw_latent, latent_lengths


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_model_config(args.config)
    if args.tokenizer:
        tokenizer = CharTokenizer.from_dict(json.loads(Path(args.tokenizer).read_text()))
    else:
        entries = load_filelist(args.filelist)
        tokenizer = CharTokenizer.build_from_texts([e[1] for e in entries], [e[2] for e in entries])
    (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer.to_dict()))

    ae = SpeechAutoencoder(cfg.ae).to(device)
    ae.load_state_dict(torch.load(args.autoencoder_ckpt, map_location=device)["generator"])
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

    model = DurationPredictor(cfg.dp, tokenizer.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def infinite(loader):
        while True:
            yield from loader

    data_iter = infinite(loader)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
    else:
        if args.init_ckpt:
            load_state_dict_grow_vocab(model, torch.load(args.init_ckpt, map_location=device)["model"])
        if not bool(ae.latent_stats_fitted):
            print(f"ae.latent_mean/std not yet fitted; calibrating over {args.calibrate_batches} batches...")
            calib_latents = []
            for _ in range(args.calibrate_batches):
                wav, wav_lengths, _, _ = next(data_iter)
                raw_latent, _ = encode_batch_raw(ae, cfg, wav, wav_lengths, device)
                calib_latents.append(raw_latent.cpu())
            ae.fit_latent_stats(calib_latents)

    model.train()
    while step < args.iters:
        wav, wav_lengths, text_padded, text_lengths = next(data_iter)
        text_padded = text_padded.to(device)
        text_mask = sequence_mask(text_lengths.to(device), text_padded.shape[1])

        raw_latent, latent_lengths = encode_batch_raw(ae, cfg, wav, wav_lengths, device)
        z1 = normalize_and_compress(ae, raw_latent, cfg.dp.chunk_compress_factor, cfg.dp.normalizer_scale)
        z_lengths = (latent_lengths // cfg.dp.chunk_compress_factor).clamp(min=1)
        ref_latent, ref_mask = sample_dp_reference(z1, z_lengths)

        target_duration = (wav_lengths.float() / cfg.ae.sample_rate).to(device)
        pred_duration = model(text_padded, text_mask, ref_latent, ref_mask)
        loss = duration_loss(pred_duration, target_duration)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            print(f"step {step} | L1 {loss.item():.4f}s")
        if step > 0 and step % args.ckpt_every == 0:
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()}, out_dir / f"ckpt_{step}.pt")

        step += 1

    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()}, out_dir / "ckpt_final.pt")


if __name__ == "__main__":
    main()
