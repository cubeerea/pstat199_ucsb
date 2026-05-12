"""
Week 1 Day 1 smoke test per CLAUDE.md §3.2.

Verifies:
  - Gemma-2-2B-it loads on the available device
  - forward_hook plumbing works on model.model.layers[k]
  - Residual stream shape is accessible

Usage:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --small  # same thing, just makes the flag available
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Fast iteration mode")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    model_id = "google/gemma-2-2b-it"
    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"Model loaded. Layers: {n_layers}, d_model: {d_model}")

    # Verify hook plumbing on residual stream
    captured = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            captured["shape"] = output[0].shape
        else:
            captured["shape"] = output.shape

    mid_layer = n_layers // 2
    h = model.model.layers[mid_layer].register_forward_hook(hook)

    prompt = "Hello, how are you?"
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)

    h.remove()

    assert "shape" in captured, "Hook never fired — plumbing broken"
    print(f"Layer {mid_layer} output shape: {captured['shape']}")
    print()

    # Verify pre-hook on post_attention_layernorm (residual mid extraction point)
    captured_pre = {}

    def pre_hook(module, input):
        if isinstance(input, tuple):
            captured_pre["shape"] = input[0].shape
        else:
            captured_pre["shape"] = input.shape

    h2 = model.model.layers[mid_layer].post_attention_layernorm.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        _ = model(**inputs)
    h2.remove()

    assert "shape" in captured_pre, "Pre-hook on post_attention_layernorm never fired"
    print(f"resid_mid (pre post_attention_layernorm input) shape: {captured_pre['shape']}")
    print()

    print("Smoke test PASSED")
    print()
    print("Next step: run experiments/01_persistence_verification.py --small")


if __name__ == "__main__":
    main()
