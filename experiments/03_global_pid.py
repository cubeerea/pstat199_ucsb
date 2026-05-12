"""
Week 2, Days 1-3: Global PID controller — Hank's contribution.

Prerequisites:
  - experiments/01_persistence_verification.py (needs artifacts/refusal_vector_global.pt,
    artifacts/persistence_window.json)
  - experiments/02_baseline_perlayer_pid.py (for comparison numbers)

Outputs:
  results/global_pid_asr.json

Usage:
  python experiments/03_global_pid.py --small
  python experiments/03_global_pid.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.controllers import GlobalPIDController, GlobalPIDControllerAntiWindup
from src.data import load_data
from src.eval import compute_asr
from src.hooks import add_hooks, get_actadd_output_hook

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"
KP, KI, KD = 0.9, 0.01, 0.01


def generate_completions(model, tokenizer, instructions, fwd_hooks, batch_size, max_new_tokens, device):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    n_test = 10 if args.small else None
    max_new_tokens = 64 if args.small else 256
    batch_size = 4 if args.small else 16

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device} | n_test: {n_test or 'all'} | model: {MODEL_ID}")

    # Load global vector and window
    global_path = ARTIFACTS_DIR / "refusal_vector_global.pt"
    window_path = ARTIFACTS_DIR / "persistence_window.json"
    if not global_path.exists():
        raise FileNotFoundError(f"{global_path} not found. Run 01_persistence_verification.py first.")
    r_bar = torch.load(global_path).to(device)
    with open(window_path) as f:
        window_data = json.load(f)
    window = window_data["window"]
    print(f"r_bar shape: {r_bar.shape}, norm: {r_bar.norm():.4f}")
    print(f"Window W = layers {window[0]}-{window[-1]} ({len(window)} layers), "
          f"mean cosine: {window_data['mean_cosine']:.3f} [{window_data['verdict']}]")

    _, harmful_test, _, _ = load_data(n_test=n_test)
    print(f"Test set: {len(harmful_test)} harmful prompts")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=device
    )
    model.eval()
    model.requires_grad_(False)

    module_dict = dict(model.named_modules())
    results = {}

    # Load per-layer PID baseline for comparison
    baseline_path = RESULTS_DIR / "baseline_perlayer_pid_asr.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        results["no_steering"] = baseline.get("no_steering")
        results["perlayer_pid"] = baseline.get("perlayer_pid")

    # ── Global PID ────────────────────────────────────────────────────────────
    print("\n[1/2] Global PID")
    ctrl = GlobalPIDController(r_bar=r_bar, kp=KP, ki=KI, kd=KD, window=window)
    steering_dirs = ctrl.precompute_steering_dirs()

    fwd_hooks = [
        (
            module_dict[f"model.layers.{k}.post_attention_layernorm"],
            get_actadd_output_hook(steering_dirs[k].to(device), scale=1.0),
        )
        for k in window
        if f"model.layers.{k}.post_attention_layernorm" in module_dict
    ]
    completions = generate_completions(
        model, tokenizer, harmful_test, fwd_hooks, batch_size, max_new_tokens, device
    )
    asr = compute_asr(completions)
    results["global_pid"] = {**asr, "kp": KP, "ki": KI, "kd": KD, "window": window}
    print(f"  Global PID ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")

    # ── Global PID + Anti-windup ───────────────────────────────────────────────
    print("\n[2/2] Global PID + Anti-windup")
    ctrl_aw = GlobalPIDControllerAntiWindup(r_bar=r_bar, kp=KP, ki=KI, kd=KD, window=window)
    steering_dirs_aw = ctrl_aw.precompute_steering_dirs()

    fwd_hooks_aw = [
        (
            module_dict[f"model.layers.{k}.post_attention_layernorm"],
            get_actadd_output_hook(steering_dirs_aw[k].to(device), scale=1.0),
        )
        for k in window
        if f"model.layers.{k}.post_attention_layernorm" in module_dict
    ]
    completions_aw = generate_completions(
        model, tokenizer, harmful_test, fwd_hooks_aw, batch_size, max_new_tokens, device
    )
    asr_aw = compute_asr(completions_aw)
    results["global_pid_antiwindup"] = {**asr_aw, "kp": KP, "ki": KI, "kd": KD, "window": window}
    print(f"  Global PID + AW ASR: {asr_aw['asr']:.3f} ({asr_aw['n_success']}/{asr_aw['n_total']})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nAll conditions:")
    for cond, res in results.items():
        if res:
            print(f"  {cond:<30} ASR={res['asr']:.3f}")

    out_path = RESULTS_DIR / "global_pid_asr.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
