"""Checkpoint-loading helpers shared by the fine-tuning entry points."""
from __future__ import annotations

import torch


def load_state_dict_grow_vocab(model: torch.nn.Module, state_dict: dict) -> None:
    """Loads `state_dict` into `model`, tolerating a dim-0 size mismatch (e.g. an
    embedding table sized for a smaller vocab than the fine-tuning tokenizer's --
    see CharTokenizer.extend_with_texts). Every other tensor shape must match
    exactly. Rows/entries beyond the checkpoint's dim-0 size are left at the
    model's own fresh initialization.
    """
    model_state = model.state_dict()
    missing = state_dict.keys() - model_state.keys()
    unexpected = model_state.keys() - state_dict.keys()
    if missing or unexpected:
        raise RuntimeError(f"key mismatch loading checkpoint: missing={missing} unexpected={unexpected}")

    with torch.no_grad():
        for key, ckpt_tensor in state_dict.items():
            target = model_state[key]
            if ckpt_tensor.shape == target.shape:
                target.copy_(ckpt_tensor)
            elif ckpt_tensor.dim() == target.dim() and ckpt_tensor.shape[1:] == target.shape[1:]:
                n = min(ckpt_tensor.shape[0], target.shape[0])
                target[:n].copy_(ckpt_tensor[:n])
            else:
                raise RuntimeError(
                    f"incompatible shape for {key}: checkpoint={tuple(ckpt_tensor.shape)} model={tuple(target.shape)}"
                )
