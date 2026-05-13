"""
Residual stream hook utilities for HuggingFace Gemma-2.
Adapted from llm-activation-control/angular_steering.py (add_hooks, pre/post hook factories).
No vLLM, no TransformerLens.
"""
import contextlib
import functools
from typing import Callable

import torch
from torch import Tensor


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: list[tuple[torch.nn.Module, Callable]],
    module_forward_hooks: list[tuple[torch.nn.Module, Callable]],
    **kwargs,
):
    """
    Context manager for temporarily attaching forward hooks to a model.
    Hooks are removed even if an exception is raised.
    """
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            handles.append(module.register_forward_pre_hook(functools.partial(hook, **kwargs)))
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(functools.partial(hook, **kwargs)))
        yield
    finally:
        for h in handles:
            h.remove()


def get_capture_pre_hook(
    layer_idx: int,
    cache: dict,
    positions: list[int] | None = None,
):
    """
    Pre-hook that captures the INPUT of a module at specified token positions.
    Use on layernorm sub-modules to capture the residual stream before normalization.
    Stored in cache[layer_idx] as a tensor of shape (batch, len(positions), d_model).
    """
    def hook_fn(module, input):
        activation: Tensor = input[0]
        if positions is not None:
            cache[layer_idx] = activation[:, positions, :].detach().clone()
        else:
            cache[layer_idx] = activation.detach().clone()

    return hook_fn


def get_capture_output_hook(
    layer_idx: int,
    cache: dict,
    positions: list[int] | None = None,
):
    """
    Output hook that captures the OUTPUT of a module (e.g. a full decoder layer).
    Handles tuple outputs — takes the first element (hidden_states).
    Use for resid_post: hook the decoder layer itself to capture the full block output.
    Stored in cache[layer_idx] as a tensor of shape (batch, len(positions), d_model).
    """
    def hook_fn(module, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        if positions is not None:
            cache[layer_idx] = activation[:, positions, :].detach().clone()
        else:
            cache[layer_idx] = activation.detach().clone()

    return hook_fn


def get_actadd_output_hook(
    steering_vector: Tensor,
    scale: float = 1.0,
):
    """
    Output hook that adds a steering vector to the residual stream (plain ActAdd).
    Applied at every token position during generation.
    """
    def hook_fn(module, input, output):
        nonlocal steering_vector
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        v = steering_vector.to(activation.device, dtype=activation.dtype)
        activation = activation + scale * v

        if isinstance(output, tuple):
            return (activation, *output[1:])
        return activation

    return hook_fn


def get_gemma2_resid_module_names(model) -> dict[int, dict[str, str]]:
    """
    Returns per-layer module name mapping for Gemma-2's residual stream positions.

    Gemma-2 architecture:
      resid_mid  = INPUT of post_attention_layernorm  (post-attn, pre-FFN)
      resid_post = INPUT of post_feedforward_layernorm (post-FFN, full block output)

    Usage:
      module_names = get_gemma2_resid_module_names(model)
      module_names[k]["mid"]  -> "model.model.layers.{k}.post_attention_layernorm"
      module_names[k]["post"] -> "model.model.layers.{k}.post_feedforward_layernorm"
    """
    n_layers = model.config.num_hidden_layers
    return {
        k: {
            "mid": f"model.model.layers.{k}.post_attention_layernorm",
            "post": f"model.model.layers.{k}.post_feedforward_layernorm",
        }
        for k in range(n_layers)
    }
