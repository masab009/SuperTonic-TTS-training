# SupertonicTTS training code

A from-scratch PyTorch training pipeline for [SupertonicTTS](https://arxiv.org/abs/2503.23108)
(Kim et al., Supertone Inc.), reconstructed from:

- `papers/2503.23108v3.pdf` — the paper (architecture: Section 3 + Appendix A; training
  setup / hyperparameters: Section 4.2; losses: Eq. 1-6, Appendix B.1).
- `supertonic-3-model/onnx/tts.json` — the config shipped with the public
  [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) ONNX release,
  used to confirm exact dimensions.
- The four ONNX graphs (`onnx/*.onnx`) in that release, traced node-by-node (not just
  matched by weight shape) to pin down every place the paper's prose was ambiguous or
  wrong — op order, activation functions, padding mode, residual connections, and
  attention scaling were all corrected this way. See "Assumptions" below.

No pretrained weights ship with this repo — the public release only ships ONNX
inference graphs, not PyTorch training checkpoints. `port_onnx_weights.py` recovers
what *can* be recovered from those graphs (see "Weight porting" below); the rest is a
full reimplementation trained from scratch.

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

### Quickstart (training a fresh model from scratch)

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

## Fine-tuning Supertone's released model for a new language

The recommended path if you want a model that actually speaks a new language, rather
than training one from nothing: port what's publicly recoverable from
Supertone/supertonic-3, then continue training each stage on your new-language data.

```bash
pip install -r training/requirements-port.txt   # adds onnx + onnxruntime

# 1. Recover every weight that's actually present in the public ONNX graphs.
python -m training.port_onnx_weights \
  --onnx_dir supertonic-3-model/onnx --out_dir runs/ported \
  --voice_style supertonic-3-model/voice_styles/F1.json

# 2. Stage 1: the speech autoencoder's ENCODER isn't public (see "Weight porting"), so
#    fine-tune a new one against Supertone's real, pretrained, frozen decoder instead of
#    training both from scratch -- converges far faster than a full 1.5M-step run.
python -m training.train_autoencoder \
  --filelist data/train.txt --root_dir data/wavs --out_dir runs/ae_ft \
  --config supertonic-3-model/onnx/tts.json \
  --init_ckpt runs/ported/autoencoder_ported.pt --freeze_decoder \
  --iters 50000

# 3. Stage 2: fine-tune the (mostly-ported) text-to-latent module. Reuse the tokenizer
#    port_onnx_weights.py built from the real model's unicode_indexer.json -- it's what
#    the ported char-embedding table's rows actually mean, and reusing it keeps every
#    known character's pretrained embedding aligned instead of starting the vocab over.
python -m training.train_text_to_latent \
  --filelist data/train.txt --root_dir data/wavs \
  --config supertonic-3-model/onnx/tts.json \
  --tokenizer runs/ported/tokenizer.json \
  --autoencoder_ckpt runs/ae_ft/ckpt_final.pt \
  --init_ckpt runs/ported/ttl_ported.pt --out_dir runs/ttl_ft \
  --iters 50000

# 4. Stage 3: same idea for the duration predictor.
python -m training.train_duration_predictor \
  --filelist data/train.txt --root_dir data/wavs \
  --config supertonic-3-model/onnx/tts.json \
  --tokenizer runs/ported/tokenizer.json \
  --autoencoder_ckpt runs/ae_ft/ckpt_final.pt \
  --init_ckpt runs/ported/dp_ported.pt --out_dir runs/dp_ft \
  --iters 2000
```

Notes:

- `--init_ckpt` warm-starts model weights only (fresh optimizer, step 0) — use it to
  start from a ported or otherwise pretrained checkpoint. `--resume` (all three
  scripts) instead restores a full training run, including optimizer/scheduler state
  and step count, to continue one that was interrupted.
- `--freeze_decoder` (stage 1 only) keeps the decoder's parameters out of the
  optimizer entirely; only pass it together with `--init_ckpt`/`--resume` pointing at
  a checkpoint that actually has a trained decoder in it.
- Iteration counts above are starting points, not tuned targets — a few 10s of
  thousands of steps is a reasonable place to start for fine-tuning vs. the paper's
  full pretraining run, but watch your own loss curves.
- The character vocabulary you fine-tune with must be able to represent the new
  language's script. If it uses characters absent from `unicode_indexer.json`
  entirely, they'll fall back to the pad id (character 0) at encode time; there's no
  vocab-extension path here.

### Smoke-tested, not benchmarked

This exact port → fine-tune sequence (all three stages, `--init_ckpt`, `--freeze_decoder`,
`--resume`) has been run end-to-end against the real `supertonic-3-model/onnx` release on
a handful of real utterances for a few hundred steps, confirmed to produce finite losses,
saveable/resumable checkpoints, and a valid Euler-sampled synthesis through the ported
vocoder. It has **not** been trained to convergence or evaluated for audio quality — that
depends entirely on the language and dataset you fine-tune with.

## Weight porting (`port_onnx_weights.py`)

```bash
python -m training.port_onnx_weights \
    --onnx_dir supertonic-3-model/onnx --out_dir runs/ported \
    [--voice_style supertonic-3-model/voice_styles/F1.json]
```

Resolves every initializer in the four released ONNX graphs back to this codebase's
module names (handling both named initializers and the anonymous ones PyTorch's ONNX
exporter emits for some `nn.Linear` weights, tracing the consuming graph node to
recover their original path), copies what matches, and — unless `--skip_verify` is
passed — cross-checks the result against `onnxruntime` on identical input.

What gets ported, and how closely the result currently matches onnxruntime
(max|diff| over a full fp32 forward pass on one test sentence):

| Source graph | Destination | Match |
|---|---|---|
| `vocoder.onnx` | `SpeechAutoencoder.decoder` + `latent_mean`/`latent_std` | exact (~0.003) |
| `duration_predictor.onnx` | `DurationPredictor.text_encoder` + `.estimator` | exact (~4e-6) |
| `text_encoder.onnx` | `TextToLatentModel.text_encoder` | close (~0.4 on a ~1.9-range output, correlation >0.9) but not yet float32-exact |
| `vector_estimator.onnx` | `TextToLatentModel.vector_field`, except each block's rotary text-conditioning cross-attention | diverges as expected — that one sub-module isn't ported, see below |

**Not recoverable** — absent from every public ONNX graph, so these stay randomly
initialized regardless of porting and must be trained from scratch:

- `SpeechAutoencoder.encoder` (mel → latent) — this is why stage-1 fine-tuning
  (`--freeze_decoder`) exists; the decoder is real, the encoder has to be trained
  to match it.
- `TextToLatentModel.style_encoder` / `DurationPredictor.style_encoder` (reference
  audio → style tokens) — at inference the released model takes precomputed style
  vectors as direct graph inputs (see `voice_styles/*.json`), so no reference-encoder
  weights are ever exported.
- Each `VFBlock`'s rotary text-conditioning cross-attention (`main_blocks.*.attn`) —
  present in the graph but resolved through enough dynamic-shape ONNX ops that
  reverse-engineering its exact wiring wasn't done here. It trains fine from
  scratch alongside everything else during fine-tuning; it's just not warm-started.

## Testing

```bash
pip install pytest
pytest training/tests/
```

Unit tests cover shape/gradient-flow correctness of every module (attention variants,
ConvNeXt blocks including causal/masked behavior, the autoencoder, both flow-matching
and duration losses, the tokenizer, dataset/collate padding) and config loading against
the real `tts.json`. They use small hand-built configs, not the full model scale, so the
whole suite runs in well under a second and needs no audio data or GPU. Cross-checking
against the real released model (`test_config.py`'s `tts.json` test aside) is a separate,
slower step — see "Weight porting" above — since it requires the `supertonic-3-model`
release and `onnxruntime` to be present locally.

## Layout

```
training/
  config.py                    architecture dataclasses; paper defaults, or load tts.json
  text.py                      character tokenizer (built from your corpus, or from the
                                released model's unicode_indexer.json via port_onnx_weights.py)
  latent_utils.py               temporal latent (de)compression, channel normalizer, ref-crop sampling
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
  port_onnx_weights.py          recover pretrained weights from the public ONNX release
  tests/                        pytest unit tests (see "Testing")
```

## Assumptions made where the paper/config were ambiguous

The paper and config are unusually detailed, but a few implementation choices aren't
fully pinned down by prose alone. Everything below was cross-checked (not just
shape-matched) against the actual released ONNX graphs via `port_onnx_weights.py`,
which is what caught most of these — the paper's descriptions of module *boundaries*
(what's in the text encoder vs. the VF estimator, etc.) are accurate, but several
*internal* details it either doesn't specify or states loosely turned out to differ
from what the shipped model actually computes:

- **Padding is edge/replicate, not zero.** Every `Pad` node feeding a depthwise or
  streaming conv in the released model (`ConvNeXtBlock1D`'s dwconv, the decoder's
  input conv, the decoder's output head conv) uses `mode="edge"` — replicating the
  boundary value — not PyTorch's zero-padding default. Getting this wrong doesn't
  change output shape, only values, so it's an easy thing to silently get wrong.
- **`ConvNeXtBlock1D` masks three times per block, not once.** The mask is applied to
  the input before the depthwise conv, again to the conv's output, and again after the
  residual add — and the residual branch itself is the *masked* input, not the raw
  one. A single mask multiply after the whole block (the more obvious way to write it)
  is not equivalent once any batch has real padding.
- **The self-attention FFN uses ReLU, not GELU**, and masks its input before *each*
  of its two conv layers (VITS/Glow-TTS's `FFN`, used by `RelPosTransformerEncoder`).
  The relative-position self-attention itself (`RelativePositionSelfAttention`) also
  scales the relative-position logits by the same `1/sqrt(k_channels)` factor as the
  main attention scores — scaling only the main scores and leaving the relative term
  unscaled is a subtle bug that still produces a plausible-looking (but wrong)
  attention distribution.
- **Two outer residual connections the paper's architecture figures don't show
  explicitly**: `TextEncoder` and `DPTextEncoder` (the duration predictor's sentence
  encoder) both add the ConvNeXt stack's output back in *after* the self-attention
  block, before the result is projected/pooled further. Easy to miss because each
  sub-layer already has its own internal residual.
- **`ConditionCrossAttention`** (the tanh-bounded style/reference cross-attention used
  throughout) scales attention scores by `1/sqrt(hidden)` — the full pre-head-split
  projection width — not `1/sqrt(head_dim)` as standard multi-head attention would.
  Also: `TextEncoder.proj_out` has no learned weight in the released model (it's
  `nn.Identity()` here, confirmed by tracing — the "proj_out" node in the graph is
  only a mask multiply), while `DPTextEncoder.proj_out` does have a real (bias-free)
  weight. Two modules with the same name in the paper's prose, two different
  realities in the shipped graph.
- **`DurationPredictor` concatenates text-then-style (not style-then-text)** before
  the final MLP, and that MLP predicts **log-duration** — its output needs `.exp()`
  before it's a duration in seconds. Getting either of these wrong produces a
  finite-but-nonsense duration, not a crash, so it's worth knowing this if you're
  cross-checking numbers.
- **Rotary position scale in text/latent cross-attention**
  (`vf.text_cond_layer.rotary_scale=10`). Text (character-rate) and latent
  (compressed, ~14 Hz) sequences run at very different effective rates.
  `RotaryEmbedding` divides the *key* (text) position by this scale so text and latent
  positions land on comparable units — a common trick for RoPE-based cross-modal
  alignment, but not spelled out in the paper.
- **`StyleTokenLayer`** (the NANSY++-style timbre-token block): implemented as two
  independent multi-head-attention-pooling passes (learnable seed queries attending
  over the reference sequence) producing `style_key` and `style_value`, matching
  Appendix A.2.1/A.3.1's description of two cross-attention layers with learnable
  queries. The duration predictor's reference encoder only uses the value branch
  (`style_key_dim=0`) — that branch's attention module isn't instantiated at all in
  that case, rather than being built and left unused.
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
- **Tokenizer id 0 is not padding when using the released vocab.** `CharTokenizer`
  reserves id 0 for padding when building its own vocab from a training corpus, but
  `unicode_indexer.json` (the released model's real vocab table) assigns id 0 to an
  actual character. The embedding layers here deliberately don't use
  `nn.Embedding`'s `padding_idx` — masking is fully position-based (`text_mask`), so
  `padding_idx` bought nothing except a landmine: it would have permanently zeroed
  the gradient for whatever real character happens to own id 0 in the released vocab.

None of these affect the overall parameter count enormously (a from-scratch build at
the paper's stated dimensions comes to ~71M vs. the paper's reported 44M — same
architecture, some projection widths inferred rather than exact), but if you have
access to Supertone's actual training source, cross-check these spots first.
