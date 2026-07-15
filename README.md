# SupertonicTTS training code

A from-scratch PyTorch training pipeline for [SupertonicTTS](https://arxiv.org/abs/2503.23108)
(Kim et al., Supertone Inc.), reconstructed from:

- `papers/2503.23108v3.pdf` — the paper (architecture: Section 3 + Appendix A; training
  setup / hyperparameters: Section 4.2; losses: Eq. 1-6, Appendix B.1).
- `supertonic-3-model/onnx/tts.json` — the config shipped with the public
  [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) ONNX release,
  used to confirm exact dimensions and one undocumented detail (see "Assumptions" below).
- The four ONNX graphs (`onnx/*.onnx`) in that release, inspected for I/O shapes to
  confirm module boundaries.

No pretrained weights are used or loadable here — the public release only ships ONNX
inference graphs, not PyTorch training weights, so this is a full reimplementation.

## Three training stages, matching the paper's three modules

```
1. Speech autoencoder        train_autoencoder.py        (Section 3.1)
2. Text-to-latent module     train_text_to_latent.py     (Section 3.2, needs a stage-1 checkpoint)
3. Duration predictor        train_duration_predictor.py (Section 3.3, needs a stage-1 checkpoint)
```

Stage 1 must finish first: stages 2 and 3 freeze the trained speech-autoencoder
*encoder* and use it on the fly to turn training audio into target latents.

### Data format

A pipe-delimited filelist, one utterance per line:

```
relative/path/to/audio.wav|transcript text|en
```

`lang` is optional (defaults to `en`); any of the 31 codes in `training/text.py`
(`AVAILABLE_LANGS`) works — it's wrapped around the text as `<lang>...</lang>`,
matching the released model's raw-character, G2P-free input scheme.

### Quickstart

```bash
pip install -r training/requirements.txt

python -m training.train_autoencoder \
  --filelist data/train.txt --root_dir data/wavs --out_dir runs/ae \
  --batch_size 128 --iters 1500000        # paper: 4x RTX4090, 1.5M steps

python -m training.train_text_to_latent \
  --filelist data/train.txt --root_dir data/wavs \
  --autoencoder_ckpt runs/ae/ckpt_final.pt --out_dir runs/ttl \
  --batch_size 64 --iters 700000          # paper: 4x RTX4090, 700k steps

python -m training.train_duration_predictor \
  --filelist data/train.txt --root_dir data/wavs \
  --autoencoder_ckpt runs/ae/ckpt_final.pt \
  --tokenizer runs/ttl/tokenizer.json --out_dir runs/dp \
  --batch_size 128 --iters 3000           # paper: 1x RTX4090, 3k steps
```

Pass `--config supertonic-3-model/onnx/tts.json` to any script to train at the
larger, publicly released model scale (~99M params) instead of the paper's
44M-parameter research checkpoint (the default, no `--config` needed).

## Layout

```
training/
  config.py                    architecture dataclasses; paper defaults, or load tts.json
  text.py                      character tokenizer (built from your corpus at train time)
  latent_utils.py              temporal latent (de)compression, channel normalizer, ref-crop sampling
  losses.py                    multi-res mel L1, LSGAN adv losses, feature matching (autoencoder GAN)
  datasets.py                  filelist-based Dataset/collate for both training regimes
  modules/
    layers.py                  ConvNeXt block, rotary attention, style-token pooling, time embedding
    autoencoder.py              mel(+linear)-spec -> ConvNeXt latent encoder/decoder (WaveNeXt head)
    discriminators.py           multi-period + multi-resolution discriminators
    text_to_latent.py           text encoder, style/reference encoder, VF estimator, CFG masking,
                                 flow-matching loss with context-sharing batch expansion
    duration_predictor.py       DP text/style encoders + 2-layer duration MLP
  train_autoencoder.py
  train_text_to_latent.py
  train_duration_predictor.py
```

## Assumptions made where the paper/config were ambiguous

The paper and config are unusually detailed, but a few implementation choices
aren't fully pinned down publicly. Flagged in code comments at each site too:

- **Mel-spectrogram encoder input.** The paper's Section 3.1.1 describes a plain
  228-band mel input, but the released config's `ae.encoder.idim` (1253) doesn't
  match `n_mels` (228). `1253 = 228 + 1025 = n_mels + (n_fft//2 + 1)`, so the
  released model's encoder appears to concatenate the mel spectrogram with the
  log-linear-magnitude spectrogram. `AEConfig.use_linear_spec` implements this and
  is auto-detected when loading `tts.json`; it's off by default at paper scale.
- **Rotary position scale in text/latent cross-attention** (`vf.text_cond_layer.rotary_scale=10`).
  Text (character-rate) and latent (compressed, ~14 Hz) sequences run at very
  different effective rates. `RotaryEmbedding` divides the *key* (text) position
  by this scale so text and latent positions land on comparable units — a common
  trick for RoPE-based cross-modal alignment, but not spelled out in the paper.
- **`StyleTokenLayer`** (the NANSY++-style timbre-token block): implemented as two
  independent multi-head-attention-pooling passes (learnable seed queries attending
  over the reference sequence) producing `style_key` and `style_value`, matching
  Appendix A.2.1/A.3.1's description of two cross-attention layers with learnable
  queries. Exact internal projection widths are inferred from `tts.json`.
- **`UncondMasker`** (classifier-free guidance dropout): the config exposes
  `prob_both_uncond` and `prob_text_uncond` separately; implemented as mutually
  exclusive per-sample sampling (both-uncond, text-only-uncond, or fully
  conditioned) replacing embeddings with learnable null vectors, per the paper's
  "conditioning variables are replaced with learnable parameters."
- **Reference-segment sampling.** Section 3.2.4 confirms the reference crop used
  during text-to-latent training is drawn from *within the same utterance's own
  target latent* (not a separately encoded clip), and that span is excluded from
  the flow-matching loss via a mask — this is exactly what `sample_reference_crop`
  + the `ref_time_mask` argument to `flow_matching_loss` implement.

None of these affect the overall parameter count enormously (a from-scratch build at
the paper's stated dimensions comes to ~71M vs. the paper's reported 44M — same
architecture, some projection widths inferred rather than exact), but if you have
access to Supertone's actual training source, cross-check these five spots first.
# SuperTonic-TTS-training
