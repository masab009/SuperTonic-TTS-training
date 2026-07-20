"""Ports recoverable weights from Supertone/supertonic-3's public ONNX release
into this codebase's PyTorch modules, then verifies the port by comparing
outputs against onnxruntime on identical input.

    python -m training.port_onnx_weights \
        --onnx_dir supertonic-3-model/onnx --out_dir runs/ported \
        [--voice_style supertonic-3-model/voice_styles/F1.json]

What gets ported (see training/README.md "Weight porting" for full detail), and how
closely `verify()` below matches onnxruntime on a fixed test input (max|diff|, single
fp32 forward pass):
  - vocoder.onnx           -> SpeechAutoencoder.decoder + latent_mean/latent_std   [~0.003, ~exact]
  - text_encoder.onnx      -> TextToLatentModel.text_encoder                      [~0.4 on a ~1.9-range
                               output; loads cleanly and correlates >0.9 with onnxruntime, but doesn't
                               yet reach the vocoder/duration_predictor's float32-exact match -- if you
                               find the remaining gap, it's most likely in ConditionCrossAttention or
                               RelativePositionSelfAttention, see training/README.md "Assumptions"]
  - vector_estimator.onnx  -> TextToLatentModel.vector_field, EXCEPT each block's
                               rotary text-conditioning cross-attention ("attn"),
                               which needs more reverse-engineering than was done
                               here and is left randomly initialized            [partially verified,
                               diverges as expected because of the unported attn]
  - duration_predictor.onnx -> DurationPredictor.text_encoder + estimator MLP     [~0.000004, exact]

NOT recoverable -- absent from every public ONNX graph, so these stay randomly
initialized regardless of porting and must be trained from scratch:
  - SpeechAutoencoder.encoder (mel -> latent)
  - TextToLatentModel.style_encoder (reference audio -> style_ttl)
  - DurationPredictor.style_encoder
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import torch

from training.config import load_model_config
from training.modules.autoencoder import SpeechAutoencoder
from training.modules.duration_predictor import DurationPredictor
from training.modules.text_to_latent import TextToLatentModel
from training.text import CharTokenizer


# --------------------------------------------------------------------------
# ONNX graph -> qualified weight name resolution
# --------------------------------------------------------------------------


# empirically observed prefixes for graph-node-traced (anonymous-initializer) names
# that don't match their named-initializer siblings' canonical "tts.*" path
_NODE_NAME_ALIASES = {
    "vector_estimator.vector_field.": "tts.ttl.vector_field.",
    "speech_prompted_text_encoder.": "tts.ttl.speech_prompted_text_encoder.",
    "decoder.": "tts.ae.decoder.",
}


def load_onnx_arrays(path: str) -> dict[str, np.ndarray]:
    """Every initializer, keyed by its *qualified* module path. Named initializers
    (e.g. "tts.ae.decoder.convnext.0.gamma") are used as-is. Anonymous ones
    (e.g. "onnx::MatMul_3680", emitted for some nn.Linear weights during export)
    are resolved by tracing the graph to find the node that consumes them --
    that node's `name` attribute retains the full original module path
    (e.g. "/speech_prompted_text_encoder/attention1/W_query/linear/MatMul").

    Named initializers keep the model's true nested attribute path (always "tts.*"),
    but anonymous ones are resolved via *graph node* names, which are taken from a
    shorter export-time alias for a few submodules -- e.g. vector_estimator.onnx's
    node names read "vector_estimator/vector_field/..." (missing the "tts.ttl."
    that the same submodule's named-initializer siblings use), and
    text_encoder.onnx's read "speech_prompted_text_encoder/..." (same issue, no
    graph-name wrapper this time). Both are normalized back to the canonical
    "tts.*" form here via `_NODE_NAME_ALIASES` so every caller can assume the same
    convention regardless of which graph or resolution path a name came from.
    """
    m = onnx.load(path)
    inits = {i.name: i for i in m.graph.initializer}
    consumer = {}
    for node in m.graph.node:
        for idx, inp in enumerate(node.input):
            if inp in inits and (inp.startswith("onnx::") or inp.startswith("/")):
                consumer.setdefault(inp, (node.name, idx))

    resolved = {}
    for name, init in inits.items():
        arr = onnx.numpy_helper.to_array(init)
        if name.startswith("onnx::") or name.startswith("/"):
            entry = consumer.get(name)
            if entry is None:
                continue
            node_name, input_idx = entry
            # a node can have BOTH its weight and bias resolved through this anonymous
            # path (e.g. decoder.embed's Conv) -- always suffixing ".weight" would collide
            # the two under one key, silently dropping whichever loads second. Conv/Gemm's
            # 3rd input (index 2) is always the bias; everything else here (including
            # single-input cases like PReLU's slope) is a "weight" in our module naming.
            suffix = "bias" if input_idx >= 2 else "weight"
            qualified = node_name.strip("/").rsplit("/", 1)[0].replace("/", ".") + f".{suffix}"
        else:
            qualified = name
        if not qualified.startswith("tts."):
            if ".tts." in qualified:
                qualified = qualified[qualified.index("tts.") :]
            else:
                for alias, canonical in _NODE_NAME_ALIASES.items():
                    if qualified.startswith(alias):
                        qualified = canonical + qualified[len(alias) :]
                        break
        resolved[qualified] = arr
    return resolved


# --------------------------------------------------------------------------
# generic, shape-aware tensor assignment
# --------------------------------------------------------------------------


@dataclass
class PortReport:
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    optional_missing: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded)} missing={len(self.missing)} "
            f"optional_missing={len(self.optional_missing)} shape_mismatch={len(self.shape_mismatch)}"
        )


def try_copy(dst: torch.nn.Parameter, weights: dict, key: str, report: PortReport, optional: bool = False) -> None:
    if key not in weights:
        (report.optional_missing if optional else report.missing).append(key)
        return
    src = torch.from_numpy(weights[key]).float()
    # Conv1d(kernel=1) weight (out,in,1) -> Linear weight (out,in)
    src_sq = src.squeeze(-1) if src.dim() == dst.dim() + 1 and src.shape[-1] == 1 else src
    with torch.no_grad():
        if src_sq.shape == dst.shape:
            dst.copy_(src_sq)
            report.loaded.append(key)
        elif src_sq.dim() == 2 and src_sq.T.shape == dst.shape:
            dst.copy_(src_sq.T)
            report.loaded.append(f"{key} (transposed)")
        elif src_sq.numel() == dst.numel():
            dst.copy_(src_sq.reshape(dst.shape))
            report.loaded.append(f"{key} (reshaped)")
        else:
            report.shape_mismatch.append((key, tuple(src.shape), tuple(dst.shape)))


# --------------------------------------------------------------------------
# per-submodule porters
# --------------------------------------------------------------------------


def port_convnext_stack(stack, weights, prefix, report, causal=False):
    blocks = stack.blocks if hasattr(stack, "blocks") else stack
    for i, block in enumerate(blocks):
        p = f"{prefix}.convnext.{i}"
        dw = f"{p}.dwconv.net" if causal else f"{p}.dwconv"
        try_copy(block.dwconv.weight, weights, f"{dw}.weight", report)
        try_copy(block.dwconv.bias, weights, f"{dw}.bias", report)
        try_copy(block.norm.weight, weights, f"{p}.norm.norm.weight", report)
        try_copy(block.norm.bias, weights, f"{p}.norm.norm.bias", report)
        try_copy(block.pwconv1.weight, weights, f"{p}.pwconv1.weight", report)
        try_copy(block.pwconv1.bias, weights, f"{p}.pwconv1.bias", report)
        try_copy(block.pwconv2.weight, weights, f"{p}.pwconv2.weight", report)
        try_copy(block.pwconv2.bias, weights, f"{p}.pwconv2.bias", report)
        try_copy(block.gamma, weights, f"{p}.gamma", report)


def port_rel_pos_encoder(enc, weights, prefix, report):
    n = len(enc.attn_layers)
    for i in range(n):
        ap = f"{prefix}.attn_layers.{i}"
        attn = enc.attn_layers[i]
        for name in ("conv_q", "conv_k", "conv_v", "conv_o"):
            try_copy(getattr(attn, name).weight, weights, f"{ap}.{name}.weight", report)
            try_copy(getattr(attn, name).bias, weights, f"{ap}.{name}.bias", report)
        try_copy(attn.emb_rel_k, weights, f"{ap}.emb_rel_k", report)
        try_copy(attn.emb_rel_v, weights, f"{ap}.emb_rel_v", report)

        try_copy(enc.norm_layers_1[i].norm.weight, weights, f"{prefix}.norm_layers_1.{i}.norm.weight", report)
        try_copy(enc.norm_layers_1[i].norm.bias, weights, f"{prefix}.norm_layers_1.{i}.norm.bias", report)
        ffn = enc.ffn_layers[i]
        try_copy(ffn["conv_1"].weight, weights, f"{prefix}.ffn_layers.{i}.conv_1.weight", report)
        try_copy(ffn["conv_1"].bias, weights, f"{prefix}.ffn_layers.{i}.conv_1.bias", report)
        try_copy(ffn["conv_2"].weight, weights, f"{prefix}.ffn_layers.{i}.conv_2.weight", report)
        try_copy(ffn["conv_2"].bias, weights, f"{prefix}.ffn_layers.{i}.conv_2.bias", report)
        try_copy(enc.norm_layers_2[i].norm.weight, weights, f"{prefix}.norm_layers_2.{i}.norm.weight", report)
        try_copy(enc.norm_layers_2[i].norm.bias, weights, f"{prefix}.norm_layers_2.{i}.norm.bias", report)


def port_cross_attn(attn, weights, prefix, report):
    for name in ("W_query", "W_key", "W_value", "out_fc"):
        m = getattr(attn, name)
        try_copy(m.linear.weight, weights, f"{prefix}.{name}.linear.weight", report)
        try_copy(m.linear.bias, weights, f"{prefix}.{name}.linear.bias", report, optional=True)


def port_text_encoder(text_encoder, weights, report):
    prefix = "tts.ttl.text_encoder"
    try_copy(text_encoder.embed.weight, weights, f"{prefix}.text_embedder.char_embedder.weight", report)
    port_convnext_stack(text_encoder.convnext, weights, f"{prefix}.convnext", report)
    port_rel_pos_encoder(text_encoder.attn_encoder, weights, f"{prefix}.attn_encoder", report)
    port_cross_attn(text_encoder.attention1, weights, "tts.ttl.speech_prompted_text_encoder.attention1", report)
    port_cross_attn(text_encoder.attention2, weights, "tts.ttl.speech_prompted_text_encoder.attention2", report)
    try_copy(text_encoder.norm.weight, weights, "tts.ttl.speech_prompted_text_encoder.norm.norm.weight", report)
    try_copy(text_encoder.norm.bias, weights, "tts.ttl.speech_prompted_text_encoder.norm.norm.bias", report)
    # proj_out is nn.Identity() -- ground truth confirms the released graph has no learned
    # weight there (see TextEncoder's proj_out docstring in modules/text_to_latent.py)


def port_dp_text_encoder(dp_text_encoder, weights, report):
    prefix = "tts.dp.sentence_encoder"
    try_copy(dp_text_encoder.embed.weight, weights, f"{prefix}.text_embedder.char_embedder.weight", report)
    try_copy(dp_text_encoder.sentence_token, weights, f"{prefix}.sentence_token", report)
    port_convnext_stack(dp_text_encoder.convnext, weights, f"{prefix}.convnext", report)
    port_rel_pos_encoder(dp_text_encoder.attn_encoder, weights, f"{prefix}.attn_encoder", report)
    try_copy(dp_text_encoder.proj_out.weight, weights, f"{prefix}.proj_out.net.weight", report)


def port_duration_estimator(estimator, weights, report):
    prefix = "tts.dp.predictor"
    lin0, act, lin1 = estimator[0], estimator[1], estimator[2]
    try_copy(lin0.weight, weights, f"{prefix}.layers.0.weight", report)
    try_copy(lin0.bias, weights, f"{prefix}.layers.0.bias", report)
    try_copy(act.weight, weights, f"{prefix}.activation.weight", report)
    try_copy(lin1.weight, weights, f"{prefix}.layers.1.weight", report)
    try_copy(lin1.bias, weights, f"{prefix}.layers.1.bias", report)


def port_vf_estimator(vf, weights, report):
    prefix = "tts.ttl.vector_field"
    try_copy(vf.proj_in.weight, weights, f"{prefix}.proj_in.net.weight", report)
    try_copy(vf.time_encoder.mlp[0].weight, weights, f"{prefix}.time_encoder.mlp.0.linear.weight", report)
    try_copy(vf.time_encoder.mlp[0].bias, weights, f"{prefix}.time_encoder.mlp.0.linear.bias", report)
    try_copy(vf.time_encoder.mlp[2].weight, weights, f"{prefix}.time_encoder.mlp.2.linear.weight", report)
    try_copy(vf.time_encoder.mlp[2].bias, weights, f"{prefix}.time_encoder.mlp.2.linear.bias", report)

    for m, block in enumerate(vf.main_blocks):
        base = 6 * m
        port_convnext_stack(block.dilated_convnext, weights, f"{prefix}.main_blocks.{base}", report)
        try_copy(block.time_linear.weight, weights, f"{prefix}.main_blocks.{base + 1}.linear.linear.weight", report)
        try_copy(block.time_linear.bias, weights, f"{prefix}.main_blocks.{base + 1}.linear.linear.bias", report)
        port_convnext_stack(block.convnext_1, weights, f"{prefix}.main_blocks.{base + 2}", report)
        report.optional_missing.append(f"{prefix}.main_blocks.{base + 3}.attn.* (rotary text-cond, not ported)")
        port_convnext_stack(block.convnext_2, weights, f"{prefix}.main_blocks.{base + 4}", report)
        port_cross_attn(block.attention, weights, f"{prefix}.main_blocks.{base + 5}.attention", report)
        try_copy(block.attention_norm.weight, weights, f"{prefix}.main_blocks.{base + 5}.norm.norm.weight", report)
        try_copy(block.attention_norm.bias, weights, f"{prefix}.main_blocks.{base + 5}.norm.norm.bias", report)

    port_convnext_stack(vf.last_convnext, weights, f"{prefix}.last_convnext", report)
    try_copy(vf.proj_out.weight, weights, f"{prefix}.proj_out.net.weight", report)


def port_decoder(decoder, weights, report):
    prefix = "tts.ae.decoder"
    # embed's BatchNorm is fused into in_conv at export time -- see LatentDecoder's
    # docstring in modules/autoencoder.py -- so there's no separate ".embed.weight/bias"
    # (BN scale/shift) to load, only the (BN-folded) conv weight/bias themselves.
    try_copy(decoder.in_conv.weight, weights, f"{prefix}.embed.net.weight", report)
    try_copy(decoder.in_conv.bias, weights, f"{prefix}.embed.net.bias", report)
    port_convnext_stack(decoder.blocks, weights, prefix, report, causal=True)
    try_copy(decoder.out_bn.weight, weights, f"{prefix}.final_norm.norm.weight", report)
    try_copy(decoder.out_bn.bias, weights, f"{prefix}.final_norm.norm.bias", report)
    try_copy(decoder.out_bn.running_mean, weights, f"{prefix}.final_norm.norm.running_mean", report, optional=True)
    try_copy(decoder.out_bn.running_var, weights, f"{prefix}.final_norm.norm.running_var", report, optional=True)
    try_copy(decoder.head_conv.weight, weights, f"{prefix}.head.layer1.net.weight", report)
    try_copy(decoder.head_conv.bias, weights, f"{prefix}.head.layer1.net.bias", report)
    try_copy(decoder.head_act.weight, weights, f"{prefix}.head.act.weight", report)
    try_copy(decoder.head_proj.weight, weights, f"{prefix}.head.layer2.weight", report)


# --------------------------------------------------------------------------
# orchestration + verification
# --------------------------------------------------------------------------


def port_all(cfg, vocab_size: int, onnx_dir: str):
    reports = {}

    ae = SpeechAutoencoder(cfg.ae)
    vocoder_weights = load_onnx_arrays(f"{onnx_dir}/vocoder.onnx")
    r = PortReport()
    port_decoder(ae.decoder, vocoder_weights, r)
    if "tts.ae.latent_mean" in vocoder_weights:
        ae.latent_mean.copy_(torch.from_numpy(vocoder_weights["tts.ae.latent_mean"]).float())
        ae.latent_std.copy_(torch.from_numpy(vocoder_weights["tts.ae.latent_std"]).float())
        ae.latent_stats_fitted.fill_(True)
        r.loaded += ["tts.ae.latent_mean", "tts.ae.latent_std"]
    reports["vocoder (decoder)"] = r

    ttl = TextToLatentModel(cfg.ttl, vocab_size)
    text_encoder_weights = load_onnx_arrays(f"{onnx_dir}/text_encoder.onnx")
    r = PortReport()
    port_text_encoder(ttl.text_encoder, text_encoder_weights, r)
    reports["text_encoder"] = r

    vf_weights = load_onnx_arrays(f"{onnx_dir}/vector_estimator.onnx")
    r = PortReport()
    port_vf_estimator(ttl.vector_field, vf_weights, r)
    try_copy(ttl.uncond_masker.text_special_token, vf_weights, "tts.ttl.uncond_masker.text_special_token", r)
    try_copy(
        ttl.uncond_masker.style_value_special_token,
        vf_weights,
        "tts.ttl.uncond_masker.style_value_special_token",
        r,
    )
    reports["vector_estimator"] = r

    dp = DurationPredictor(cfg.dp, vocab_size)
    dp_weights = load_onnx_arrays(f"{onnx_dir}/duration_predictor.onnx")
    r = PortReport()
    port_dp_text_encoder(dp.text_encoder, dp_weights, r)
    port_duration_estimator(dp.estimator, dp_weights, r)
    reports["duration_predictor"] = r

    return ae, ttl, dp, reports


def load_voice_style(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(Path(path).read_text())
    ttl_dims = d["style_ttl"]["dims"]
    dp_dims = d["style_dp"]["dims"]
    style_ttl = np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(ttl_dims)
    style_dp = np.array(d["style_dp"]["data"], dtype=np.float32).reshape(dp_dims)
    return style_ttl, style_dp


def _max_abs_diff(a: torch.Tensor, b: np.ndarray) -> float:
    return float((a.detach().numpy() - b).__abs__().max())


def verify(cfg, tokenizer, ae, ttl, dp, onnx_dir, voice_style_path=None):
    import onnxruntime as ort

    print("\n=== Verifying against onnxruntime ===")
    torch.manual_seed(0)
    np.random.seed(0)
    ae.eval()  # BatchNorm1d must use its trained running stats, not batch-of-1 statistics
    text = "Hello, this is a verification sentence."
    text_ids = np.array([tokenizer.encode(text, "en")], dtype=np.int64)
    text_len = text_ids.shape[1]
    text_mask = np.ones((1, 1, text_len), dtype=np.float32)

    if voice_style_path:
        style_ttl_np, style_dp_np = load_voice_style(voice_style_path)
    else:
        style_ttl_np = np.random.randn(1, cfg.ttl.n_style, cfg.ttl.style_dim).astype(np.float32)
        style_dp_np = np.random.randn(1, cfg.dp.n_style, cfg.dp.style_value_dim).astype(np.float32)

    # --- text_encoder ---
    sess = ort.InferenceSession(f"{onnx_dir}/text_encoder.onnx", providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"text_ids": text_ids, "style_ttl": style_ttl_np, "text_mask": text_mask})[0]
    ttl.eval()
    with torch.no_grad():
        torch_out = ttl.text_encoder(torch.from_numpy(text_ids), torch.from_numpy(text_mask), torch.from_numpy(style_ttl_np))
    print(f"text_encoder      max|diff| = {_max_abs_diff(torch_out, onnx_out):.6f}  (shapes {tuple(torch_out.shape)} vs {onnx_out.shape})")

    # --- vector_estimator (expected to diverge: TextCondBlock rotary attn not ported) ---
    sess = ort.InferenceSession(f"{onnx_dir}/vector_estimator.onnx", providers=["CPUExecutionProvider"])
    latent_len = 12
    noisy_latent = np.random.randn(1, cfg.ttl.compressed_dim, latent_len).astype(np.float32)
    latent_mask = np.ones((1, 1, latent_len), dtype=np.float32)
    current_step = np.array([10.0], dtype=np.float32)
    total_step = np.array([32.0], dtype=np.float32)
    onnx_out = sess.run(
        None,
        {
            "noisy_latent": noisy_latent,
            "text_emb": torch_out.numpy(),
            "style_ttl": style_ttl_np,
            "latent_mask": latent_mask,
            "text_mask": text_mask,
            "current_step": current_step,
            "total_step": total_step,
        },
    )[0]
    with torch.no_grad():
        t = current_step / total_step
        vf_out = ttl.vector_field(
            torch.from_numpy(noisy_latent),
            torch.from_numpy(t),
            torch_out,
            torch.from_numpy(style_ttl_np),
            torch.from_numpy(latent_mask),
            torch.from_numpy(text_mask),
        )
    print(
        f"vector_estimator   max|diff| = {_max_abs_diff(vf_out, onnx_out):.6f}  "
        "(expected to diverge: TextCondBlock rotary cross-attn is not ported)"
    )

    # --- duration_predictor ---
    # NOTE: style_dp is fed directly here (bypassing dp.style_encoder, which is never
    # ported -- see module docstring), so this only exercises text_encoder + estimator.
    sess = ort.InferenceSession(f"{onnx_dir}/duration_predictor.onnx", providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"text_ids": text_ids, "style_dp": style_dp_np, "text_mask": text_mask})[0]
    dp.eval()
    with torch.no_grad():
        text_emb = dp.text_encoder(torch.from_numpy(text_ids), torch.from_numpy(text_mask))
        style_flat = torch.from_numpy(style_dp_np).reshape(1, -1)
        log_duration = dp.estimator(torch.cat([text_emb, style_flat], dim=-1)).squeeze(-1)
        pred = log_duration.exp()
    print(f"duration_predictor max|diff| = {abs(float(pred.item()) - float(onnx_out.reshape(-1)[0])):.6f}  ({pred.item():.3f}s vs {float(onnx_out.reshape(-1)[0]):.3f}s)")

    # --- vocoder (decoder only; encoder isn't public so this exercises the ported half) ---
    sess = ort.InferenceSession(f"{onnx_dir}/vocoder.onnx", providers=["CPUExecutionProvider"])
    from training.latent_utils import decompress_and_denormalize

    np.random.seed(0)  # re-seed so this stage's input doesn't depend on how many random
    # draws the earlier stages happened to make -- keeps each printed diff independently reproducible
    compressed = np.random.randn(1, cfg.ttl.compressed_dim, latent_len).astype(np.float32) * 0.25
    onnx_out = sess.run(None, {"latent": compressed})[0]
    with torch.no_grad():
        raw24 = decompress_and_denormalize(ae, torch.from_numpy(compressed), cfg.ttl.chunk_compress_factor, cfg.ae.ldim, cfg.ttl.normalizer_scale)
        wav = ae.decoder(raw24)
    min_len = min(wav.shape[-1], onnx_out.shape[-1])
    print(f"vocoder (decoder)  max|diff| = {_max_abs_diff(wav[..., :min_len], onnx_out[..., :min_len]):.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default=None, help="defaults to <onnx_dir>/tts.json")
    p.add_argument("--voice_style", default=None, help="e.g. supertonic-3-model/voice_styles/F1.json")
    p.add_argument("--skip_verify", action="store_true")
    args = p.parse_args()

    config_path = args.config or f"{args.onnx_dir}/tts.json"
    cfg = load_model_config(config_path)

    indexer_path = Path(args.onnx_dir) / "unicode_indexer.json"
    if indexer_path.exists():
        indexer = json.loads(indexer_path.read_text())
        char2id = {chr(cp): idx for cp, idx in enumerate(indexer) if idx >= 0}
        tokenizer = CharTokenizer(char2id=char2id)
    else:
        tokenizer = CharTokenizer.build_from_texts(["Hello world."], ["en"])

    ae, ttl, dp, reports = port_all(cfg, tokenizer.vocab_size, args.onnx_dir)

    print("=== Port report ===")
    for name, r in reports.items():
        print(f"{name}: {r.summary()}")
        for key, src_shape, dst_shape in r.shape_mismatch:
            print(f"  SHAPE MISMATCH: {key}  onnx={src_shape} model={dst_shape}")
        for key in r.missing:
            print(f"  MISSING: {key}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": ae.state_dict()}, out_dir / "autoencoder_ported.pt")
    torch.save({"model": ttl.state_dict()}, out_dir / "ttl_ported.pt")
    torch.save({"model": dp.state_dict()}, out_dir / "dp_ported.pt")
    (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer.to_dict()))
    print(f"\nSaved ported checkpoints to {out_dir}/")

    if not args.skip_verify:
        verify(cfg, tokenizer, ae, ttl, dp, args.onnx_dir, args.voice_style)


if __name__ == "__main__":
    main()
