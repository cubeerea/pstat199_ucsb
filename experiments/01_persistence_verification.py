"""
Week 1, Days 1-2: Verify refusal feature persistence across layers in Gemma-2-2B-it.

Outputs (figures are numbered, artifacts are canonical/latest):
  figures/persistence_cosine_matrix_NNN.png   ← numbered, never overwritten
  figures/persistence_cosine_matrix_NNN.json  ← sidecar config for that run
  artifacts/persistence_window.json           ← latest run (used by 02/03)
  artifacts/refusal_vectors_per_layer.pt      ← latest run
  artifacts/refusal_vector_global.pt          ← latest run (if window found)
  artifacts/cosine_matrix.npy                 ← latest run

Decision criteria (per CLAUDE.md §4.1):
  - Window ≥ 8 layers, mean intra-window cosine ≥ 0.85 → GREEN
  - Window 5-7 OR cosine 0.75-0.85 → YELLOW
  - Window < 5 OR cosine < 0.75 → RED — escalate to Hank

Usage:
  python experiments/01_persistence_verification.py --small
  python experiments/01_persistence_verification.py
  python experiments/01_persistence_verification.py --extraction-point post_ffn --threshold 0.5
  python experiments/01_persistence_verification.py --n-train 512 --batch-size 32
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_data
from src.dim import collect_activations, compute_dim, global_vector_from_window


FIGURES_DIR = Path("figures")
ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"
COSINE_THRESHOLD = 0.8

EXTRACTION_MODULES = {
    "post_attn": "post_attention_layernorm",   # residual stream mid-block (post-attention, pre-FFN)
    "post_ffn":  "post_feedforward_layernorm", # full block output (post-FFN)
    "input":     "input_layernorm",            # block input (pre-attention)
}


def next_run_id(directory: Path, prefix: str) -> int:
    """Return the next available 3-digit run number by scanning existing files."""
    existing = list(directory.glob(f"{prefix}_???.png"))
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def identify_persistence_window(cosine_matrix: np.ndarray, threshold: float) -> list[int]:
    """Find the largest contiguous window W where all pairwise cosines ≥ threshold."""
    n = cosine_matrix.shape[0]
    best_window = []
    for start in range(n):
        for end in range(start + 1, n + 1):
            window = list(range(start, end))
            sub = cosine_matrix[np.ix_(window, window)]
            mask = ~np.eye(len(window), dtype=bool)
            if mask.any() and sub[mask].min() >= threshold:
                if len(window) > len(best_window):
                    best_window = window
    return best_window


def main():
    parser = argparse.ArgumentParser(
        description="Verify per-layer refusal direction persistence in Gemma-2-2B-it."
    )
    parser.add_argument("--small", action="store_true",
                        help="Shorthand for --n-train 10 --batch-size 4 (fast, CPU ok)")
    parser.add_argument("--device", default=None,
                        help="Override device (e.g. cuda, cuda:0, mps, cpu)")
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD,
                        help=f"Cosine threshold for window identification (default: {COSINE_THRESHOLD})")
    parser.add_argument("--n-train", type=int, default=None,
                        help="Number of prompts per class for DIM extraction (default: 256)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Activation collection batch size (default: 16)")
    parser.add_argument("--extraction-point", choices=list(EXTRACTION_MODULES.keys()),
                        default="post_attn",
                        help="Which residual stream position to extract from "
                             "(post_attn=post-attention, post_ffn=full block output, "
                             "input=block input). Default: post_attn")
    args = parser.parse_args()

    # --small sets defaults; explicit flags override
    n_train = args.n_train if args.n_train is not None else (10 if args.small else 256)
    batch_size = args.batch_size if args.batch_size is not None else (4 if args.small else 16)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}  |  n_train: {n_train}  |  extract: {args.extraction_point}  "
          f"|  threshold: {args.threshold}  |  model: {MODEL_ID}")

    print("Loading data...")
    harmful_train, _, harmless_train, _ = load_data(n_train=n_train)
    print(f"  harmful: {len(harmful_train)}, harmless: {len(harmless_train)}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    print(f"  n_layers={n_layers}, d_model={model.config.hidden_size}")

    module_name_suffix = EXTRACTION_MODULES[args.extraction_point]
    module_dict = dict(model.named_modules())
    layer_modules = {
        k: module_dict[f"model.layers.{k}.{module_name_suffix}"]
        for k in range(n_layers)
    }

    print("Collecting harmful activations...")
    harmful_acts = collect_activations(
        model, tokenizer, harmful_train, layer_modules, batch_size=batch_size, device=device
    )

    print("Collecting harmless activations...")
    harmless_acts = collect_activations(
        model, tokenizer, harmless_train, layer_modules, batch_size=batch_size, device=device
    )

    print("Computing DIM directions...")
    ref_dirs = compute_dim(harmful_acts, harmless_acts)

    layers = sorted(ref_dirs.keys())
    stack = torch.stack([ref_dirs[k] for k in layers])
    stack_normed = stack / (stack.norm(dim=-1, keepdim=True) + 1e-8)
    cosine_matrix = (stack_normed @ stack_normed.T).float().cpu().numpy()

    # ── Diagnostics ───────────────────────────────────────────────────────────
    off_diag = cosine_matrix[~np.eye(len(layers), dtype=bool)]
    adj = [cosine_matrix[i, i + 1] for i in range(len(layers) - 1)]
    print(f"\nCosine matrix diagnostics (n_layers={len(layers)}, extract={args.extraction_point}):")
    print(f"  Overall  — min: {off_diag.min():.3f} | mean: {off_diag.mean():.3f} | max: {off_diag.max():.3f}")
    print(f"  Adjacent — min: {min(adj):.3f} | mean: {np.mean(adj):.3f} | max: {max(adj):.3f}")
    print(f"  Adjacent cosines by layer pair:")
    for i, c in enumerate(adj):
        bar = "#" * int(max(0, c) * 20)
        print(f"    L{i:2d}–L{i+1:2d}: {c:+.3f}  {bar}")

    print(f"\n  Best window at various thresholds:")
    for t in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        w = identify_persistence_window(cosine_matrix, threshold=t)
        label = f"L{w[0]}–L{w[-1]} (width {len(w)})" if w else "none"
        marker = " ← selected" if abs(t - args.threshold) < 1e-6 else ""
        print(f"    threshold={t:.1f}: {label}{marker}")

    np.save(ARTIFACTS_DIR / "cosine_matrix.npy", cosine_matrix)
    print(f"\n  Raw matrix saved: {ARTIFACTS_DIR / 'cosine_matrix.npy'}")

    # ── Numbered figure ───────────────────────────────────────────────────────
    run_id = next_run_id(FIGURES_DIR, "persistence_cosine_matrix")
    out_fig = FIGURES_DIR / f"persistence_cosine_matrix_{run_id:03d}.png"
    out_cfg = FIGURES_DIR / f"persistence_cosine_matrix_{run_id:03d}.json"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cosine_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(
        f"Per-layer refusal direction cosine similarity\n"
        f"{MODEL_ID},  n={n_train},  extract={args.extraction_point},  threshold={args.threshold:.2f}\n"
        f"Run #{run_id:03d}"
    )
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_fig}")

    run_config = {
        "run_id": run_id,
        "model": MODEL_ID,
        "n_train": n_train,
        "batch_size": batch_size,
        "extraction_point": args.extraction_point,
        "threshold": args.threshold,
        "device": device,
    }
    with open(out_cfg, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved: {out_cfg}")

    # ── Identify window ───────────────────────────────────────────────────────
    window = identify_persistence_window(cosine_matrix, threshold=args.threshold)

    if len(window) >= 2:
        sub = cosine_matrix[np.ix_(window, window)]
        mask = ~np.eye(len(window), dtype=bool)
        mean_cosine = float(sub[mask].mean())
    else:
        mean_cosine = 0.0

    print(f"\nPersistence window W = layers {window[0] if window else 'N/A'} – "
          f"{window[-1] if window else 'N/A'}")
    print(f"Window width: {len(window)}  |  Mean intra-window cosine: {mean_cosine:.3f}")

    if len(window) >= 8 and mean_cosine >= 0.85:
        verdict = "GREEN — good to go"
    elif len(window) >= 5 and mean_cosine >= 0.75:
        verdict = "YELLOW — proceed with caution, note as limitation"
    else:
        verdict = "RED — window too weak. Escalate to Hank before proceeding."

    if args.small and args.n_train is None:
        print(f"Decision: {verdict}  "
              f"(NOTE: RED/YELLOW expected with --small; too few prompts for stable DIM.)")
    else:
        print(f"Decision: {verdict}")

    # ── Save canonical artifacts (used by 02/03) ──────────────────────────────
    window_data = {
        "window": window, "width": len(window), "mean_cosine": mean_cosine,
        "threshold": args.threshold, "verdict": verdict, "model": MODEL_ID,
        "n_train": n_train, "extraction_point": args.extraction_point, "run_id": run_id,
    }
    window_path = ARTIFACTS_DIR / "persistence_window.json"
    with open(window_path, "w") as f:
        json.dump(window_data, f, indent=2)
    print(f"Saved: {window_path}")

    per_layer_path = ARTIFACTS_DIR / "refusal_vectors_per_layer.pt"
    torch.save({k: ref_dirs[k].cpu() for k in layers}, per_layer_path)
    print(f"Saved: {per_layer_path}")

    if window:
        r_bar = global_vector_from_window(ref_dirs, window)
        global_path = ARTIFACTS_DIR / "refusal_vector_global.pt"
        torch.save(r_bar.cpu(), global_path)
        print(f"Saved: {global_path}  (norm: {r_bar.norm():.4f})")
    else:
        print("WARNING: No persistence window identified. r_bar not saved.")

    print(f"\nDone. Figure: {out_fig}  |  Window: {window_path}")


if __name__ == "__main__":
    main()
