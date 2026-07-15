"""Stage 1: train the speech autoencoder (Section 3.1, Section 4.2).

Example:
    python -m training.train_autoencoder \
        --filelist data/train.txt --root_dir data/wavs --out_dir runs/autoencoder
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.config import load_model_config
from training.datasets import AutoencoderDataset, autoencoder_collate
from training.losses import (
    MultiResolutionMelLoss,
    discriminator_lsgan_loss,
    feature_matching_loss,
    generator_lsgan_loss,
)
from training.modules.autoencoder import SpeechAutoencoder
from training.modules.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--filelist", required=True)
    p.add_argument("--root_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default=None, help="path to tts.json; defaults to the paper's 44M config")
    p.add_argument("--segment_seconds", type=float, default=0.19)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--iters", type=int, default=1_500_000)
    p.add_argument("--lambda_recon", type=float, default=45.0)
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--lambda_fm", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_ckpt(path, step, generator, mpd, mrd, opt_g, opt_d):
    torch.save(
        {
            "step": step,
            "generator": generator.state_dict(),
            "mpd": mpd.state_dict(),
            "mrd": mrd.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_model_config(args.config)
    segment_samples = int(round(args.segment_seconds * cfg.ae.sample_rate / cfg.ae.hop_length)) * cfg.ae.hop_length

    dataset = AutoencoderDataset(args.filelist, args.root_dir, cfg.ae.sample_rate, segment_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=autoencoder_collate,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    generator = SpeechAutoencoder(cfg.ae).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    mel_loss_fn = MultiResolutionMelLoss(cfg.ae.sample_rate).to(device)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()), lr=args.lr, betas=(0.8, 0.99)
    )

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        mpd.load_state_dict(ckpt["mpd"])
        mrd.load_state_dict(ckpt["mrd"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt["step"]

    def infinite(loader):
        while True:
            yield from loader

    data_iter = infinite(loader)
    generator.train()
    mpd.train()
    mrd.train()

    while step < args.iters:
        wav = next(data_iter).to(device)
        recon, _latent = generator(wav)
        min_len = min(wav.shape[-1], recon.shape[-1])
        wav, recon = wav[..., :min_len], recon[..., :min_len]

        # --- discriminator step ---
        with torch.no_grad():
            recon_detached = recon
        mpd_real, _ = mpd(wav)
        mpd_fake, _ = mpd(recon_detached.detach())
        mrd_real, _ = mrd(wav)
        mrd_fake, _ = mrd(recon_detached.detach())
        loss_d = discriminator_lsgan_loss(mpd_real, mpd_fake) + discriminator_lsgan_loss(mrd_real, mrd_fake)

        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # --- generator step ---
        mpd_real, mpd_real_feats = mpd(wav)
        mpd_fake, mpd_fake_feats = mpd(recon)
        mrd_real, mrd_real_feats = mrd(wav)
        mrd_fake, mrd_fake_feats = mrd(recon)

        loss_recon = mel_loss_fn(recon, wav)
        loss_adv = generator_lsgan_loss(mpd_fake) + generator_lsgan_loss(mrd_fake)
        loss_fm = feature_matching_loss(mpd_real_feats, mpd_fake_feats) + feature_matching_loss(
            mrd_real_feats, mrd_fake_feats
        )
        loss_g = args.lambda_recon * loss_recon + args.lambda_adv * loss_adv + args.lambda_fm * loss_fm

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step % args.log_every == 0:
            print(
                f"step {step} | G {loss_g.item():.4f} (recon {loss_recon.item():.4f} "
                f"adv {loss_adv.item():.4f} fm {loss_fm.item():.4f}) | D {loss_d.item():.4f}"
            )
        if step > 0 and step % args.ckpt_every == 0:
            save_ckpt(out_dir / f"ckpt_{step}.pt", step, generator, mpd, mrd, opt_g, opt_d)

        step += 1

    save_ckpt(out_dir / "ckpt_final.pt", step, generator, mpd, mrd, opt_g, opt_d)


if __name__ == "__main__":
    main()
