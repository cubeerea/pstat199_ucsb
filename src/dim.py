"""
Difference-in-Means (DIM) refusal direction computation.
Adapted from llm-activation-control/llama_many_layers.py (lines 450-560).

Algorithm:
  For each layer k:
    1. Capture residual stream at last token of chat-template suffix
    2. L2-normalize each prompt's activation vector
    3. ref_dir[k] = mean(harmful_normed[k]) - mean(harmless_normed[k])
  Returns per-layer directions as a tensor of shape (n_layers, d_model).

NOTE: this is the per-layer DIM baseline. The global vector r_bar is computed
in experiments/01_persistence_verification.py after identifying the window W.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer

from src.hooks import add_hooks, get_capture_pre_hook


def get_chat_suffix_last_token_idx(tokenizer: AutoTokenizer, device: str = "cpu") -> int:
    """
    Returns the index of the last token in the chat-template turn suffix.
    This is the extraction point for DIM (matches llama_many_layers.py logic).

    For Gemma-2: apply_chat_template with add_generation_prompt=True produces
    tokens ending in e.g. [..suffix.., <start_of_turn>, "model", newline].
    We use -1 (last token) as a reliable extraction point.
    """
    return -1


@torch.no_grad()
def collect_activations(
    model,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    layer_modules: dict,
    batch_size: int = 8,
    device: str | None = None,
) -> dict[int, Tensor]:
    """
    Run forward passes and collect residual stream activations at the last prompt token,
    for each layer in layer_modules.

    Args:
        layer_modules: dict mapping layer_idx -> layernorm module object to hook.
        Returns: dict layer_idx -> tensor of shape (n_prompts, d_model)
    """
    if device is None:
        device = next(model.parameters()).device

    all_acts: dict[int, list[Tensor]] = {k: [] for k in layer_modules}

    for i in range(0, len(instructions), batch_size):
        batch = instructions[i : i + batch_size]
        chats = [
            [{"role": "user", "content": instr}] for instr in batch
        ]
        inputs_str = tokenizer.apply_chat_template(
            chats,
            padding=True,
            truncation=False,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(inputs_str, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        cache: dict[int, Tensor] = {}
        pre_hooks = [
            (module, get_capture_pre_hook(layer_idx=layer_idx, cache=cache, positions=[-1]))
            for layer_idx, module in layer_modules.items()
        ]

        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            model(**inputs)

        for k in layer_modules:
            # cache[k]: (batch, 1, d_model) → squeeze seq dim
            all_acts[k].append(cache[k].squeeze(1).float().cpu())

    return {k: torch.cat(v, dim=0) for k, v in all_acts.items()}


def compute_dim(
    harmful_acts: dict[int, Tensor],
    harmless_acts: dict[int, Tensor],
) -> dict[int, Tensor]:
    """
    Compute per-layer DIM refusal directions.
    Direction points from harmless → harmful (matches paper convention).
    Directions are NOT normalized here; call /||r|| if needed.

    Returns: dict layer_idx -> tensor of shape (d_model,)
    """
    ref_dirs: dict[int, Tensor] = {}
    for k in harmful_acts:
        harm = harmful_acts[k]
        harm_normed = harm / (harm.norm(dim=-1, keepdim=True) + 1e-8)
        harm_mean = harm_normed.mean(dim=0)

        beni = harmless_acts[k]
        beni_normed = beni / (beni.norm(dim=-1, keepdim=True) + 1e-8)
        beni_mean = beni_normed.mean(dim=0)

        ref_dirs[k] = harm_mean - beni_mean
    return ref_dirs


def apply_pid_to_dirs(
    ref_dirs: dict[int, Tensor],
    kp: float = 0.9,
    ki: float = 0.01,
    kd: float = 0.01,
) -> dict[int, Tensor]:
    """
    Apply PID recurrence to per-layer DIM directions, matching angular_steering.ipynb
    Cell 47 inline implementation.

    Integral = prefix sum of ref_dirs across layers.
    Derivative = first difference (ref_dirs[k] - ref_dirs[k-1]; ref_dirs[0] for k=0).

    Returns modified steering directions dict.
    """
    layers = sorted(ref_dirs.keys())
    stack = torch.stack([ref_dirs[k] for k in layers])  # (n_layers, d_model)

    prefix_sum = torch.cumsum(stack, dim=0)
    shifted = stack.roll(1, dims=0)
    shifted[0] = stack[0]
    diff = stack - shifted

    steered = kp * stack + ki * prefix_sum + kd * diff

    return {k: steered[i] for i, k in enumerate(layers)}


def global_vector_from_window(
    ref_dirs: dict[int, Tensor],
    window: list[int],
) -> Tensor:
    """
    Compute the global refusal vector r_bar as the mean of per-layer directions
    within the persistence window W.

    Args:
        ref_dirs: per-layer DIM directions
        window: list of layer indices in the persistence window

    Returns: tensor of shape (d_model,)
    """
    vecs = torch.stack([ref_dirs[k] for k in window])
    r_bar = vecs.mean(dim=0)
    return r_bar
