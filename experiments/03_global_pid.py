"""
Week 2, Days 1-3: Global PID controller — Hank's contribution.

Prerequisites:
  - experiments/01_persistence_verification.py (needs artifacts/refusal_vector_global.pt,
    artifacts/persistence_window.json)
  - experiments/02_baseline_perlayer_pid.py (for comparison numbers)

Outputs:
  results/global_pid_asr[_TAG].json   — ASR + per-layer diagnostics
  figures/activation_norm_vs_layer_NNN.png  — Plot A (§4.7)
  figures/iterm_magnitude_vs_layer_NNN.png  — Plot B (§4.7)

Usage:
  python experiments/03_global_pid.py --small
  python experiments/03_global_pid.py
  python experiments/03_global_pid.py --kp 1.0 --ki 0.05 --kd 0.0 --scale 0.5
  python experiments/03_global_pid.py --ki 0.1 --tag ki010   # windup-regime probe
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.controllers import GlobalPIDController, GlobalPIDControllerAntiWindup
from src.data import load_data
from src.eval import compute_asr
from src.hooks import add_hooks, get_actadd_output_hook, get_capture_output_hook

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"
CAPTURE_N = 32  # prompts for diagnostic capture pass (not generation)


# ── Figure helpers (mirror 01_persistence_verification.py convention) ────────

def next_run_id(directory: Path, prefix: str) -> int:
    existing = list(directory.glob(f"{prefix}_???.png"))
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


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


def capture_activation_norms(
    model, tokenizer, prompts, module_dict, n_layers, window, steer_hooks, batch_size, device
):
    """
    Single forward pass (no generate) over `prompts` to capture per-layer
    ||h^(k)|| at the last token position. Steer_hooks are registered BEFORE
    the capture hooks so capture sees the post-injection activations.

    Returns dict: layer_idx (all layers) -> mean norm over prompts.
    """
    # Pre-build capture hooks for ALL layers (not just window), capture hooks only
    act_sums = {k: 0.0 for k in range(n_layers)}
    act_counts = {k: 0 for k in range(n_layers)}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        chats = [[{"role": "user", "content": instr}] for instr in batch]
        inputs_str = tokenizer.apply_chat_template(
            chats, padding=True, truncation=False, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(inputs_str, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Build per-batch capture cache
        batch_cache: dict[int, torch.Tensor] = {}
        cap_hooks = [
            (module_dict[f"model.layers.{k}"],
             get_capture_output_hook(k, batch_cache, positions=[-1]))
            for k in range(n_layers)
            if f"model.layers.{k}" in module_dict
        ]

        # Steer hooks (on post_attention_layernorm) fire before capture hooks
        # (on the decoder layer) because they're on different modules.
        all_hooks = steer_hooks + cap_hooks

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=all_hooks):
            with torch.no_grad():
                model(**inputs)

        for k, act in batch_cache.items():
            # act: (batch, 1, d_model) — last token position
            norms = act.squeeze(1).norm(dim=-1)  # (batch,)
            act_sums[k] += float(norms.sum())
            act_counts[k] += len(batch)

    return {k: act_sums[k] / max(act_counts[k], 1) for k in range(n_layers)}


def main():
    parser = argparse.ArgumentParser(
        description="Global PID steering on Gemma-2-2B-it (Hank's contribution)."
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
                        help="Max tokens to generate per prompt (default: 256)")
    parser.add_argument("--kp", type=float, default=0.9,
                        help="Proportional gain (default: 0.9)")
    parser.add_argument("--ki", type=float, default=0.01,
                        help="Integral gain (default: 0.01)")
    parser.add_argument("--kd", type=float, default=0.01,
                        help="Derivative gain (default: 0.01)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Steering vector scale / alpha (default: 1.0)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix for output filenames, e.g. ki010 → global_pid_asr_ki010.json")
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

    print(f"Device: {device} | n_test: {n_test or 'all'} | "
          f"Kp={args.kp} Ki={args.ki} Kd={args.kd} scale={args.scale} | model: {MODEL_ID}")

    global_path = ARTIFACTS_DIR / "refusal_vector_global.pt"
    window_path = ARTIFACTS_DIR / "persistence_window.json"
    if not global_path.exists():
        raise FileNotFoundError(
            f"{global_path} not found. Run 01_persistence_verification.py first "
            f"(needs a non-empty persistence window)."
        )
    r_bar = torch.load(global_path).to(device)
    with open(window_path) as f:
        window_data = json.load(f)
    window = window_data["window"]
    if not window:
        raise ValueError(
            "artifacts/persistence_window.json has an empty window. "
            "Re-run: python experiments/01_persistence_verification.py --threshold 0.7"
        )
    print(f"r_bar shape: {r_bar.shape}, norm: {r_bar.norm():.4f}")
    print(f"Window W = layers {window[0]}-{window[-1]} ({len(window)} layers), "
          f"mean cosine: {window_data['mean_cosine']:.3f} [{window_data['verdict']}]")

    _, harmful_test, _, _ = load_data(n_test=n_test)
    print(f"Test set: {len(harmful_test)} harmful prompts")

    # Subsample for diagnostic capture (cheap single forward pass)
    capture_prompts = harmful_test[:min(CAPTURE_N, len(harmful_test))]
    if args.small:
        capture_prompts = harmful_test[:min(4, len(harmful_test))]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    module_dict = dict(model.named_modules())
    results = {}

    baseline_path = RESULTS_DIR / "baseline_perlayer_pid_asr.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        results["no_steering"] = baseline.get("no_steering")
        results["perlayer_pid"] = baseline.get("perlayer_pid")

    # ── Global PID ────────────────────────────────────────────────────────────
    print("\n[1/2] Global PID")
    ctrl = GlobalPIDController(r_bar=r_bar, kp=args.kp, ki=args.ki, kd=args.kd, window=window)
    steering_dirs = ctrl.precompute_steering_dirs()

    # Log term magnitudes (deterministic from recurrence — no forward pass needed)
    print(f"  P-term norms by layer: {[f'{ctrl.p_norms[k]:.4f}' for k in sorted(window)]}")
    print(f"  I-term norms by layer: {[f'{ctrl.i_norms[k]:.4f}' for k in sorted(window)]}")
    print(f"  AW clamp value: 2·||r̄||·Ki = {2*float(r_bar.norm())*args.ki:.4f} "
          f"(I-term exceeds this? {any(ctrl.i_norms[k] > 2*float(r_bar.norm())*args.ki for k in window)})")

    fwd_hooks = [
        (
            module_dict[f"model.layers.{k}.post_attention_layernorm"],
            get_actadd_output_hook(steering_dirs[k].to(device), scale=args.scale),
        )
        for k in window
        if f"model.layers.{k}.post_attention_layernorm" in module_dict
    ]

    # Diagnostic capture pass (separate from generation)
    steer_hooks_for_capture = fwd_hooks  # on post_attn_ln, fires before decoder-layer output hooks
    act_norms_global = capture_activation_norms(
        model, tokenizer, capture_prompts, module_dict, n_layers, window,
        steer_hooks_for_capture, batch_size=batch_size if args.small else 8, device=device
    )

    completions = generate_completions(
        model, tokenizer, harmful_test, fwd_hooks, batch_size, max_new_tokens, device
    )
    asr = compute_asr(completions)
    results["global_pid"] = {**asr, "kp": args.kp, "ki": args.ki, "kd": args.kd,
                              "scale": args.scale, "window": window,
                              "p_norms": ctrl.p_norms, "i_norms": ctrl.i_norms,
                              "d_norms": ctrl.d_norms, "integral_norms": ctrl.integral_norms,
                              "act_norms_by_layer": act_norms_global}
    print(f"  Global PID ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")

    # ── Global PID + Anti-windup ──────────────────────────────────────────────
    print("\n[2/2] Global PID + Anti-windup")
    ctrl_aw = GlobalPIDControllerAntiWindup(
        r_bar=r_bar, kp=args.kp, ki=args.ki, kd=args.kd, window=window
    )
    steering_dirs_aw = ctrl_aw.precompute_steering_dirs()

    fwd_hooks_aw = [
        (
            module_dict[f"model.layers.{k}.post_attention_layernorm"],
            get_actadd_output_hook(steering_dirs_aw[k].to(device), scale=args.scale),
        )
        for k in window
        if f"model.layers.{k}.post_attention_layernorm" in module_dict
    ]

    act_norms_aw = capture_activation_norms(
        model, tokenizer, capture_prompts, module_dict, n_layers, window,
        fwd_hooks_aw, batch_size=batch_size if args.small else 8, device=device
    )

    completions_aw = generate_completions(
        model, tokenizer, harmful_test, fwd_hooks_aw, batch_size, max_new_tokens, device
    )
    asr_aw = compute_asr(completions_aw)
    results["global_pid_antiwindup"] = {**asr_aw, "kp": args.kp, "ki": args.ki, "kd": args.kd,
                                         "scale": args.scale, "window": window,
                                         "p_norms": ctrl_aw.p_norms, "i_norms": ctrl_aw.i_norms,
                                         "d_norms": ctrl_aw.d_norms,
                                         "integral_norms": ctrl_aw.integral_norms,
                                         "act_norms_by_layer": act_norms_aw}
    print(f"  Global PID + AW ASR: {asr_aw['asr']:.3f} ({asr_aw['n_success']}/{asr_aw['n_total']})")

    # ── Unsteered capture (no baseline JSON) ─────────────────────────────────
    if "no_steering" not in results or results["no_steering"] is None:
        print("\n  (no_steering not in baseline_perlayer_pid_asr.json — skipping unsteered ASR)")
    act_norms_nosteer = capture_activation_norms(
        model, tokenizer, capture_prompts, module_dict, n_layers, window,
        [], batch_size=batch_size if args.small else 8, device=device
    )
    results["no_steering_act_norms"] = act_norms_nosteer

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nAll conditions:")
    for cond, res in results.items():
        if res and isinstance(res, dict) and "asr" in res:
            print(f"  {cond:<30} ASR={res['asr']:.3f}")

    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"global_pid_asr{tag_suffix}.json"
    # Convert int keys to str for JSON
    def jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [jsonify(x) for x in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    layers_all = list(range(n_layers))

    # -- Plot A: activation norm vs layer (§4.7 Plot A) -----------------------
    run_id_a = next_run_id(FIGURES_DIR, "activation_norm_vs_layer")
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions_for_plot_a = {
        "no_steering": (act_norms_nosteer, "C0", "--"),
        "global_pid": (act_norms_global, "C1", "-"),
        "global_pid_antiwindup": (act_norms_aw, "C2", ":"),
    }
    if results.get("perlayer_pid") and "act_norms_by_layer" in results.get("perlayer_pid", {}):
        conditions_for_plot_a["perlayer_pid"] = (
            results["perlayer_pid"]["act_norms_by_layer"], "C3", "-."
        )

    for label, (norms_dict, color, ls) in conditions_for_plot_a.items():
        ys = [norms_dict.get(k, float("nan")) for k in layers_all]
        ax.plot(layers_all, ys, label=label, color=color, linestyle=ls, linewidth=1.5)

    if window:
        ax.axvspan(window[0], window[-1], alpha=0.08, color="grey", label=f"Window W=[{window[0]},{window[-1]}]")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean ||h^(k)|| at last token")
    ax.set_title(
        f"Activation norm vs layer — {MODEL_ID}\n"
        f"Kp={args.kp} Ki={args.ki} Kd={args.kd} scale={args.scale} | Run #{run_id_a:03d}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_a_path = FIGURES_DIR / f"activation_norm_vs_layer_{run_id_a:03d}.png"
    fig.savefig(fig_a_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_a_path}")

    # -- Plot B: I-term magnitude vs layer (§4.7 Plot B) ----------------------
    run_id_b = next_run_id(FIGURES_DIR, "iterm_magnitude_vs_layer")
    aw_clamp = args.ki * 2.0 * float(r_bar.norm())

    fig, ax = plt.subplots(figsize=(10, 5))
    layers_w = sorted(window)

    p_vals = [ctrl.p_norms[k] for k in layers_w]
    i_vals_plain = [ctrl.i_norms[k] for k in layers_w]
    i_vals_aw = [ctrl_aw.i_norms[k] for k in layers_w]
    d_vals = [ctrl.d_norms[k] for k in layers_w]

    ax.plot(layers_w, p_vals, label="P-term", color="C0", linewidth=2)
    ax.plot(layers_w, i_vals_plain, label="I-term (vanilla)", color="C1", linewidth=2)
    ax.plot(layers_w, i_vals_aw, label="I-term (anti-windup)", color="C2", linewidth=2, linestyle="--")
    ax.plot(layers_w, d_vals, label="D-term", color="C3", linewidth=1.5, linestyle=":")
    ax.axhline(aw_clamp, color="C2", linestyle="-.", linewidth=1,
               label=f"AW clamp = Ki·2·||r̄|| = {aw_clamp:.4f}")

    ax.set_xlabel("Layer index (in window W)")
    ax.set_ylabel("Term norm  ||·||")
    ax.set_title(
        f"PID term magnitudes vs layer — {MODEL_ID}\n"
        f"Kp={args.kp} Ki={args.ki} Kd={args.kd} | Run #{run_id_b:03d}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_b_path = FIGURES_DIR / f"iterm_magnitude_vs_layer_{run_id_b:03d}.png"
    fig.savefig(fig_b_path, dpi=150)
    plt.close(fig)

    sidecar = {
        "run_id": run_id_b, "model": MODEL_ID, "kp": args.kp, "ki": args.ki, "kd": args.kd,
        "scale": args.scale, "window": window, "r_bar_norm": float(r_bar.norm()),
        "aw_clamp": aw_clamp, "tag": args.tag,
    }
    sidecar_path = FIGURES_DIR / f"iterm_magnitude_vs_layer_{run_id_b:03d}.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Saved: {fig_b_path}  +  {sidecar_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
