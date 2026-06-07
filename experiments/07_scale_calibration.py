"""
Phase 2: Calibrate steering scale (α).

Sweeps scale ∈ {0.5, 1, 2, 5, 10, 20, 50, 100} using activation-norm captures
(no generation — fast). For each scale, measures how much the Global PID hooks
shift ||h^(k)|| at every layer vs. no-steering baseline.

Identifies:
  scale_min  — smallest scale where mean |Δ| inside window exceeds 5% of ||h_nosteer||
  scale_break — smallest scale where ||h^(final_layer)|| > 2× no-steering (blowup)
  recommended — (scale_min + scale_break) / 2 if scale_min < scale_break else None

Outputs:
  figures/scale_calibration.png   (multi-line: one line per scale)
  results/scale_calibration.json  (scale_min, scale_break, recommended, all norms)

Usage:
  python experiments/07_scale_calibration.py --small
  python experiments/07_scale_calibration.py
  python experiments/07_scale_calibration.py --scales 1 5 10 20 50
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.attacks import ATTACK_NAMES, apply_attack
from src.controllers import GlobalPIDController
from src.data import load_data
from src.hooks import add_hooks, get_actadd_output_hook, get_capture_output_hook

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"
SCALE_GRID_DEFAULT = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
DELTA_THRESHOLD = 0.05   # 5% relative divergence inside window
BLOWUP_RATIO = 2.0       # final-layer norm > 2× no-steer → blown up


def capture_norms(model, tokenizer, prompts, module_dict, n_layers, steer_hooks, batch_size, device):
    """Single forward pass over prompts, return mean ||h^(k)|| at last token per layer."""
    act_sums = {k: 0.0 for k in range(n_layers)}
    act_counts = {k: 0 for k in range(n_layers)}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        chats = [[{"role": "user", "content": p}] for p in batch]
        inputs_str = tokenizer.apply_chat_template(
            chats, padding=True, truncation=False, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(inputs_str, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        batch_cache: dict[int, torch.Tensor] = {}
        cap_hooks = [
            (module_dict[f"model.layers.{k}"],
             get_capture_output_hook(k, batch_cache, positions=[-1]))
            for k in range(n_layers)
            if f"model.layers.{k}" in module_dict
        ]

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=steer_hooks + cap_hooks):
            with torch.no_grad():
                model(**inputs)

        for k, act in batch_cache.items():
            norms = act.squeeze(1).norm(dim=-1)
            act_sums[k] += float(norms.sum())
            act_counts[k] += len(batch)

    return {k: act_sums[k] / max(act_counts[k], 1) for k in range(n_layers)}


def main():
    parser = argparse.ArgumentParser(description="Scale calibration sweep for Global PID.")
    parser.add_argument("--small", action="store_true",
                        help="--n-probe 4 --batch-size 4")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-probe", type=int, default=None,
                        help="Number of prompts for activation capture (default: 16)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=SCALE_GRID_DEFAULT,
                        help="Scale values to sweep")
    parser.add_argument("--kp", type=float, default=0.9)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--kd", type=float, default=0.01)
    parser.add_argument("--sign", type=int, choices=[1, -1], default=1,
                        help="Steering direction: 1=toward refusal, -1=away (sanity check)")
    parser.add_argument("--attack", choices=ATTACK_NAMES, default="none",
                        help="Attack to apply to probe prompts (default: none)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix for output filenames")
    args = parser.parse_args()

    n_probe = args.n_probe if args.n_probe is not None else (4 if args.small else 16)
    batch_size = args.batch_size if args.batch_size is not None else (4 if args.small else 8)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sign_label = "REVERSE (sanity check)" if args.sign == -1 else "forward"
    print(f"Device: {device} | n_probe: {n_probe} | scales: {args.scales} | "
          f"attack={args.attack} sign={sign_label}")

    global_path = ARTIFACTS_DIR / "refusal_vector_global.pt"
    window_path = ARTIFACTS_DIR / "persistence_window.json"
    for p in [global_path, window_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run experiments/01_persistence_verification.py --threshold 0.5 --commit"
            )
    r_bar = torch.load(global_path).to(device)
    with open(window_path) as f:
        window_data = json.load(f)
    window = window_data["window"]
    if not window:
        raise ValueError("Empty window. Re-run 01 with --threshold 0.5 --commit.")

    print(f"r_bar norm: {r_bar.norm():.4f}  |  Window W = {window[0]}–{window[-1]} ({len(window)} layers)")

    _, harmful_raw, _, _ = load_data(n_test=n_probe)
    probes = apply_attack(harmful_raw[:n_probe], args.attack)
    print(f"Probe set: {len(probes)} harmful prompts (attack={args.attack})")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    module_dict = dict(model.named_modules())
    layers_all = list(range(n_layers))

    # ── No-steering baseline ──────────────────────────────────────────────────
    print("\nCapturing no-steering baseline norms...")
    nosteer_norms = capture_norms(model, tokenizer, probes, module_dict, n_layers, [], batch_size, device)
    nosteer_final = nosteer_norms[n_layers - 1]
    print(f"  Final-layer norm (no steer): {nosteer_final:.2f}")

    # ── Scale sweep ───────────────────────────────────────────────────────────
    sweep_results = {}
    scale_min = None
    scale_break = None

    for scale in sorted(args.scales):
        ctrl = GlobalPIDController(r_bar=r_bar, kp=args.kp, ki=args.ki, kd=args.kd, window=window, sign=args.sign)
        sdirs = ctrl.precompute_steering_dirs()

        steer_hooks = [
            (module_dict[f"model.layers.{k}.post_attention_layernorm"],
             get_actadd_output_hook(sdirs[k].to(device), scale=scale))
            for k in window
            if f"model.layers.{k}.post_attention_layernorm" in module_dict
        ]

        norms = capture_norms(model, tokenizer, probes, module_dict, n_layers, steer_hooks, batch_size, device)
        final_norm = norms[n_layers - 1]

        # Mean relative Δ inside window
        deltas = [abs(norms[k] - nosteer_norms[k]) / max(nosteer_norms[k], 1e-6) for k in window]
        mean_delta_window = sum(deltas) / len(deltas)
        blowup = final_norm > BLOWUP_RATIO * nosteer_final

        print(f"  scale={scale:6.1f}  final_norm={final_norm:8.1f}  "
              f"mean_Δ_window={mean_delta_window:.4f}  blowup={blowup}")

        sweep_results[scale] = {
            "scale": scale,
            "mean_delta_window": mean_delta_window,
            "final_layer_norm": final_norm,
            "blowup": blowup,
            "norms_by_layer": {str(k): norms[k] for k in layers_all},
        }

        if scale_min is None and mean_delta_window >= DELTA_THRESHOLD:
            scale_min = scale
        if scale_break is None and blowup:
            scale_break = scale

    if scale_min is not None and scale_break is not None and scale_min < scale_break:
        recommended = (scale_min + scale_break) / 2.0
        print(f"\nscale_min={scale_min}  scale_break={scale_break}  → recommended={recommended:.1f}")
    elif scale_min is None:
        recommended = None
        print("\nWARNING: No scale produces >5% window divergence. Steering has no measurable effect.")
    elif scale_break is not None and scale_min >= scale_break:
        recommended = None
        print("\nWARNING: Blowup occurs at or before measurable divergence. No stable operating point.")
    else:
        recommended = scale_min
        print(f"\nscale_min={scale_min}  no blowup detected → recommended={recommended:.1f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "scale_min": scale_min,
        "scale_break": scale_break,
        "recommended": recommended,
        "delta_threshold": DELTA_THRESHOLD,
        "blowup_ratio": BLOWUP_RATIO,
        "nosteer_final_norm": nosteer_final,
        "window": window,
        "kp": args.kp, "ki": args.ki, "kd": args.kd,
        "sign": args.sign, "attack": args.attack,
        "n_probe": n_probe,
        "model": MODEL_ID,
        "sweep": sweep_results,
    }
    attack_tag = f"_{args.attack}" if args.attack != "none" else ""
    sign_tag = "_reverse" if args.sign == -1 else ""
    user_tag = f"_{args.tag}" if args.tag else ""
    json_path = RESULTS_DIR / f"scale_calibration{attack_tag}{sign_tag}{user_tag}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {json_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    colors = plt.cm.plasma([i / max(len(args.scales) - 1, 1) for i in range(len(args.scales))])

    # Top: ||h^(k)|| vs layer for each scale
    ax_top.plot(layers_all, [nosteer_norms[k] for k in layers_all],
                color="black", linewidth=2, linestyle="--", label="no_steering", zorder=10)
    for i, scale in enumerate(sorted(args.scales)):
        if scale not in sweep_results:
            continue
        norms_dict = {int(k): v for k, v in sweep_results[scale]["norms_by_layer"].items()}
        ys = [norms_dict.get(k, float("nan")) for k in layers_all]
        label = f"scale={scale}"
        if scale == scale_min:
            label += " ← min"
        if scale == scale_break:
            label += " ← break"
        ax_top.plot(layers_all, ys, color=colors[i], linewidth=1.5, label=label)

    if window:
        ax_top.axvspan(window[0], window[-1], alpha=0.08, color="grey",
                       label=f"W=[{window[0]},{window[-1]}]")
    ax_top.set_ylabel("Mean ||h^(k)||")
    ax_top.set_title(f"Activation norm vs layer — scale sweep  ({MODEL_ID})")
    ax_top.legend(fontsize=7, ncol=3)
    ax_top.grid(True, alpha=0.3)

    # Bottom: mean Δ inside window per scale (bar chart)
    scales_sorted = sorted(args.scales)
    deltas_bar = [sweep_results[s]["mean_delta_window"] for s in scales_sorted]
    bar_colors = ["C3" if sweep_results[s]["blowup"] else "C0" for s in scales_sorted]
    ax_bot.bar([str(s) for s in scales_sorted], deltas_bar, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax_bot.axhline(DELTA_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                   label=f"5% threshold")
    ax_bot.set_xlabel("Scale (α)")
    ax_bot.set_ylabel("Mean relative |Δ| inside window")
    ax_bot.set_title("Window divergence per scale  (red bars = blowup at final layer)")
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = FIGURES_DIR / f"scale_calibration{attack_tag}{sign_tag}{user_tag}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
