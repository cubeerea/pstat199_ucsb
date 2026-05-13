"""
Week 2, Days 4-5: GCG adversarial attack evaluation.

Appends a pre-computed universal GCG suffix (Zou et al., NeurIPS 2023) to every
AdvBench harmful prompt, then re-runs all four conditions:
  1. No steering  + GCG suffix
  2. Per-layer PID + GCG suffix
  3. Global PID   + GCG suffix
  4. Global PID + Anti-windup + GCG suffix

Do NOT run GCG optimization — it's compute-prohibitive for this timeline.
Use the released universal suffix from llm-attacks/llm-attacks.

The suffix is fetched automatically from the llm-attacks GitHub repo.
Pass --suffix to override with your own string.

Prerequisites:
  - experiments/01_persistence_verification.py (artifacts/refusal_vector_global.pt,
    artifacts/persistence_window.json, artifacts/refusal_vectors_per_layer.pt)

Outputs:
  results/gcg_attack_asr.json

Usage:
  python experiments/04_gcg_attack.py --small
  python experiments/04_gcg_attack.py
  python experiments/04_gcg_attack.py --suffix "YOUR_GCG_SUFFIX_STRING"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.controllers import (
    GlobalPIDController,
    GlobalPIDControllerAntiWindup,
    PerLayerPIDController,
)
from src.data import load_data
from src.eval import compute_asr
from src.hooks import add_hooks, get_actadd_output_hook

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"

# URL to fetch the universal GCG suffix from Zou et al.'s released transfer experiments.
# The JSON contains a list of optimized control strings; we use the first entry.
GCG_TRANSFER_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks"
    "/main/data/transfer_exps/transfer.json"
)


def fetch_gcg_suffix(url: str) -> str:
    """
    Fetch the universal GCG suffix from the llm-attacks release.
    Returns the first control string found in the transfer JSON.
    """
    print(f"Fetching GCG suffix from {url} ...")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # The transfer.json structure: top-level keys are model names or "controls" list
        # Try common key names
        for key in ("controls", "control", "suffix", "suffixes"):
            if key in data:
                val = data[key]
                suffix = val[0] if isinstance(val, list) else val
                print(f"  Found suffix under key '{key}' (length {len(suffix)} chars)")
                return suffix
        # Fallback: search nested dicts for any string value > 50 chars
        def _find_long_str(obj, depth=0):
            if depth > 4:
                return None
            if isinstance(obj, str) and len(obj) > 50:
                return obj
            if isinstance(obj, list):
                for item in obj:
                    r = _find_long_str(item, depth + 1)
                    if r:
                        return r
            if isinstance(obj, dict):
                for v in obj.values():
                    r = _find_long_str(v, depth + 1)
                    if r:
                        return r
            return None
        suffix = _find_long_str(data)
        if suffix:
            print(f"  Found suffix via deep search (length {len(suffix)} chars)")
            return suffix
        raise ValueError("Could not locate suffix string in transfer.json")
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch GCG suffix: {e}\n"
            "Pass --suffix manually. You can find the universal suffix at:\n"
            "  https://github.com/llm-attacks/llm-attacks/tree/main/data/transfer_exps\n"
            "Or copy the 'controls' string from the transfer.json file."
        ) from e


def apply_suffix(instructions: list[str], suffix: str) -> list[str]:
    """Append the GCG suffix to every instruction."""
    return [f"{instr} {suffix}" for instr in instructions]


def generate_completions(
    model, tokenizer, instructions, fwd_hooks, batch_size, max_new_tokens, device
):
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    gen_cfg.pad_token_id = tokenizer.pad_token_id

    completions = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i : i + batch_size]
        chats = [[{"role": "user", "content": instr}] for instr in batch]
        inputs_str = tokenizer.apply_chat_template(
            chats, padding=True, truncation=False, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(inputs_str, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            gen_toks = model.generate(**inputs, generation_config=gen_cfg)
            gen_toks = gen_toks[:, inputs["input_ids"].shape[-1]:]

        for tok in gen_toks:
            completions.append(tokenizer.decode(tok, skip_special_tokens=True).strip())

    return completions


def main():
    parser = argparse.ArgumentParser(
        description="GCG adversarial attack evaluation across all four steering conditions."
    )
    parser.add_argument("--small", action="store_true",
                        help="Shorthand for --n-test 10 --batch-size 4 --max-new-tokens 64")
    parser.add_argument("--device", default=None,
                        help="Override device (e.g. cuda, cuda:0, mps, cpu)")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Number of test prompts (default: all 104)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Generation batch size (default: 16)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max tokens per prompt (default: 256)")
    parser.add_argument("--kp", type=float, default=0.9)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--kd", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--suffix", type=str, default=None,
                        help="GCG suffix string to append. If omitted, fetched automatically.")
    args = parser.parse_args()

    n_test = args.n_test if args.n_test is not None else (10 if args.small else None)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (64 if args.small else 256)
    batch_size = args.batch_size if args.batch_size is not None else (4 if args.small else 16)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # ── GCG suffix ────────────────────────────────────────────────────────────
    suffix = args.suffix if args.suffix else fetch_gcg_suffix(GCG_TRANSFER_URL)
    print(f"GCG suffix preview: {suffix[:80]}{'...' if len(suffix) > 80 else ''}")

    # ── Load artifacts ────────────────────────────────────────────────────────
    per_layer_path = ARTIFACTS_DIR / "refusal_vectors_per_layer.pt"
    global_path = ARTIFACTS_DIR / "refusal_vector_global.pt"
    window_path = ARTIFACTS_DIR / "persistence_window.json"
    for p in [per_layer_path, global_path, window_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run experiments/01_persistence_verification.py first."
            )

    ref_dirs = {k: v.to(device) for k, v in torch.load(per_layer_path).items()}
    r_bar = torch.load(global_path).to(device)
    with open(window_path) as f:
        window_data = json.load(f)
    window = window_data["window"]

    print(f"Window W = layers {window[0]}-{window[-1]} ({len(window)} layers) | "
          f"Kp={args.kp} Ki={args.ki} Kd={args.kd} scale={args.scale} | model: {MODEL_ID}")

    _, harmful_test, _, _ = load_data(n_test=n_test)
    attacked_prompts = apply_suffix(harmful_test, suffix)
    print(f"Test set: {len(attacked_prompts)} prompts + GCG suffix")
    print(f"Example: {attacked_prompts[0][:120]}...")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)

    module_dict = dict(model.named_modules())
    n_layers = model.config.num_hidden_layers
    results = {}

    # ── Condition 1: No steering + GCG ───────────────────────────────────────
    print("\n[1/4] No steering + GCG")
    comps = generate_completions(
        model, tokenizer, attacked_prompts, [], batch_size, max_new_tokens, device
    )
    asr = compute_asr(comps)
    results["no_steering_gcg"] = asr
    print(f"  ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")

    # ── Condition 2: Per-layer PID + GCG ─────────────────────────────────────
    print("\n[2/4] Per-layer PID + GCG")
    ctrl_pl = PerLayerPIDController(ref_dirs, kp=args.kp, ki=args.ki, kd=args.kd)
    hooks_pl = [
        (
            module_dict[f"model.layers.{k}.post_attention_layernorm"],
            get_actadd_output_hook(ctrl_pl.steering_dirs[k].to(device), scale=args.scale),
        )
        for k in range(n_layers)
        if k in ctrl_pl.steering_dirs
    ]
    comps = generate_completions(
        model, tokenizer, attacked_prompts, hooks_pl, batch_size, max_new_tokens, device
    )
    asr = compute_asr(comps)
    results["perlayer_pid_gcg"] = {**asr, "kp": args.kp, "ki": args.ki, "kd": args.kd, "scale": args.scale}
    print(f"  ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")

    # ── Condition 3: Global PID + GCG ─────────────────────────────────────────
    print("\n[3/4] Global PID + GCG")
    ctrl_g = GlobalPIDController(r_bar=r_bar, kp=args.kp, ki=args.ki, kd=args.kd, window=window)
    sdirs_g = ctrl_g.precompute_steering_dirs()
    hooks_g = [
        (
            module_dict[f"model.layers.{k}.post_attention_layernorm"],
            get_actadd_output_hook(sdirs_g[k].to(device), scale=args.scale),
        )
        for k in window
        if f"model.layers.{k}.post_attention_layernorm" in module_dict
    ]
    comps = generate_completions(
        model, tokenizer, attacked_prompts, hooks_g, batch_size, max_new_tokens, device
    )
    asr = compute_asr(comps)
    results["global_pid_gcg"] = {**asr, "kp": args.kp, "ki": args.ki, "kd": args.kd, "scale": args.scale, "window": window}
    print(f"  ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")

    # ── Condition 4: Global PID + Anti-windup + GCG ───────────────────────────
    print("\n[4/4] Global PID + Anti-windup + GCG")
    ctrl_aw = GlobalPIDControllerAntiWindup(r_bar=r_bar, kp=args.kp, ki=args.ki, kd=args.kd, window=window)
    sdirs_aw = ctrl_aw.precompute_steering_dirs()
    hooks_aw = [
        (
            module_dict[f"model.layers.{k}.post_attention_layernorm"],
            get_actadd_output_hook(sdirs_aw[k].to(device), scale=args.scale),
        )
        for k in window
        if f"model.layers.{k}.post_attention_layernorm" in module_dict
    ]
    comps = generate_completions(
        model, tokenizer, attacked_prompts, hooks_aw, batch_size, max_new_tokens, device
    )
    asr = compute_asr(comps)
    results["global_pid_antiwindup_gcg"] = {**asr, "kp": args.kp, "ki": args.ki, "kd": args.kd, "scale": args.scale, "window": window}
    print(f"  ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nGCG attack ASR summary (suffix length: {len(suffix)} chars):")
    for cond, res in results.items():
        print(f"  {cond:<35} ASR={res['asr']:.3f}  ({res['n_success']}/{res['n_total']})")

    results["_meta"] = {
        "suffix_length": len(suffix),
        "suffix_preview": suffix[:80],
        "kp": args.kp, "ki": args.ki, "kd": args.kd, "scale": args.scale,
        "window": window,
        "model": MODEL_ID,
        "n_test": len(attacked_prompts),
    }

    out_path = RESULTS_DIR / "gcg_attack_asr.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
