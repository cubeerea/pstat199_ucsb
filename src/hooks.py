"""
Residual stream hook utilities for HuggingFace Gemma-2.
Adapted from llm-activation-control/angular_steering.py (add_hooks, pre/post hook factories).
No vLLM, no TransformerLens.
"""
import contextlib
import functools
from typing import Callable

import torch
from jaxtyping import Float
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
    Pre-hook that captures the INPUT of a layernorm module (= residual stream value)
    at specified token positions. If positions is None, captures all positions.
    Stored in cache[layer_idx] as a tensor.
    """
    def hook_fn(module, input):
        activation: Float[Tensor, "batch seq d_model"] = input[0]
        if positions is not None:
            cache[layer_idx] = activation[:, positions, :].detach().clone()
        else:
            cache[layer_idx] = activation.detach().clone()

    return hook_fn


def get_actadd_output_hook(
    steering_vector: Float[Tensor, "d_model"],
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
