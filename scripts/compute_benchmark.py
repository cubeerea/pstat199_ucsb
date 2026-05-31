"""
Phase 5: Wall-clock compute comparison (CLAUDE.md §4.8).

Measures per-forward-pass latency for three conditions over 20 trials:
  1. No steering
  2. Per-layer PID (hooks on all 26 layers)
  3. Global PID    (hooks on window layers only)

Outputs:
  results/compute_benchmark.json  — mean ± std in ms/pass + speedup ratio

Usage:
  python scripts/compute_benchmark.py
  python scripts/compute_benchmark.py --n-trials 50 --scale 20.0
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.controllers import GlobalPIDController, PerLayerPIDController
from src.data import load_data
from src.hooks import add_hooks, get_actadd_output_hook

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_ID = "google/gemma-2-2b-it"


def time_condition(model, tokenizer, prompt, hooks, device, n_trials):
    """Measure mean ± std forward-pass latency in milliseconds over n_trials."""
    chat = [{"role": "user", "content": prompt}]
    inputs_str = tokenizer.apply_chat_template(
        [chat], padding=False, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(inputs_str, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    for _ in range(3):
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=hooks):
            with torch.no_grad():
                model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=hooks):
            with torch.no_grad():
                model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    return mean_ms, std_ms


def main():
    parser = argparse.ArgumentParser(description="Per-forward-pass latency benchmark.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale to apply to steering hooks (use calibrated value from 07)")
    parser.add_argument("--kp", type=float, default=0.9)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--kd", type=float, default=0.01)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device} | n_trials: {args.n_trials} | scale: {args.scale}")

    per_layer_path = ARTIFACTS_DIR / "refusal_vectors_per_layer.pt"
    global_path = ARTIFACTS_DIR / "refusal_vector_global.pt"
    window_path = ARTIFACTS_DIR / "persistence_window.json"
    for p in [per_layer_path, global_path, window_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run 01_persistence_verification.py --commit first.")

    ref_dirs = {k: v.to(device) for k, v in torch.load(per_layer_path).items()}
    r_bar = torch.load(global_path).to(device)
    with open(window_path) as f:
        window_data = json.load(f)
    window = window_data["window"]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    module_dict = dict(model.named_modules())

    _, harmful_test, _, _ = load_data(n_test=1)
    probe = harmful_test[0]

    # Build hook lists
    pl_ctrl = PerLayerPIDController(ref_dirs, kp=args.kp, ki=args.ki, kd=args.kd)
    hooks_perlayer = [
        (module_dict[f"model.layers.{k}.post_attention_layernorm"],
         get_actadd_output_hook(pl_ctrl.steering_dirs[k].to(device), scale=args.scale))
        for k in range(n_layers) if k in pl_ctrl.steering_dirs
    ]

    g_ctrl = GlobalPIDController(r_bar=r_bar, kp=args.kp, ki=args.ki, kd=args.kd, window=window)
    sdirs_g = g_ctrl.precompute_steering_dirs()
    hooks_global = [
        (module_dict[f"model.layers.{k}.post_attention_layernorm"],
         get_actadd_output_hook(sdirs_g[k].to(device), scale=args.scale))
        for k in window if f"model.layers.{k}.post_attention_layernorm" in module_dict
    ]

    print(f"\nBenchmarking {args.n_trials} trials each...")
    print(f"  Per-layer hooks: {len(hooks_perlayer)}  |  Global hooks: {len(hooks_global)}")
    conditions = [
        ("no_steering",  []),
        ("perlayer_pid", hooks_perlayer),
        ("global_pid",   hooks_global),
    ]

    results = {}
    for label, hooks in conditions:
        mean_ms, std_ms = time_condition(model, tokenizer, probe, hooks, device, args.n_trials)
        results[label] = {"mean_ms": mean_ms, "std_ms": std_ms, "n_trials": args.n_trials}
        print(f"  {label:<20}  {mean_ms:.2f} ± {std_ms:.2f} ms")

    base = results["no_steering"]["mean_ms"]
    for label in ("perlayer_pid", "global_pid"):
        ratio = results[label]["mean_ms"] / base
        results[label]["overhead_vs_nosteer"] = ratio
        print(f"  {label} overhead vs no-steer: {ratio:.3f}×")

    if "perlayer_pid" in results and "global_pid" in results:
        speedup = results["perlayer_pid"]["mean_ms"] / results["global_pid"]["mean_ms"]
        results["_speedup_global_vs_perlayer"] = speedup
        print(f"\n  Global PID is {speedup:.2f}× {'faster' if speedup > 1 else 'slower'} than per-layer PID")

    results["_meta"] = {
        "device": device, "scale": args.scale,
        "n_window_layers": len(window), "n_all_layers": n_layers,
        "kp": args.kp, "ki": args.ki, "kd": args.kd,
        "model": MODEL_ID,
    }

    out_path = RESULTS_DIR / "compute_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
