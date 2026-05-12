"""
Week 1, Days 1-2: Verify refusal feature persistence across layers in Gemma-2-2B-it.

Outputs:
  figures/persistence_cosine_matrix.png
  artifacts/persistence_window.json
  artifacts/refusal_vectors_per_layer.pt
  artifacts/refusal_vector_global.pt

Decision criteria (per CLAUDE.md §4.1):
  - Window ≥ 8 layers, mean intra-window cosine ≥ 0.85 → green light
  - Window 5-7 OR cosine 0.75-0.85 → yellow
  - Window < 5 OR cosine < 0.75 → RED — escalate to Hank

Usage:
  python experiments/01_persistence_verification.py --small   # 10 prompts, CPU ok
  python experiments/01_persistence_verification.py           # full 256 prompts
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_data
from src.dim import collect_activations, compute_dim, global_vector_from_window
from src.hooks import get_gemma2_resid_module_names


FIGURES_DIR = Path("figures")
ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"
COSINE_THRESHOLD = 0.8


def identify_persistence_window(
    cosine_matrix: np.ndarray,
    threshold: float = COSINE_THRESHOLD,
) -> list[int]:
    """
    Find the largest contiguous window W where all pairwise cosines ≥ threshold.
    Returns the window as a sorted list of layer indices.
    """
    n = cosine_matrix.shape[0]
    best_window = []
    for start in range(n):
        for end in range(start + 1, n + 1):
            window = list(range(start, end))
            sub = cosine_matrix[np.ix_(window, window)]
            # mask diagonal (always 1.0)
            mask = ~np.eye(len(window), dtype=bool)
            if mask.any() and sub[mask].min() >= threshold:
                if len(window) > len(best_window):
                    best_window = window
    return best_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Use 10 prompts (fast, CPU)")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda:0, mps, cpu)")
    args = parser.parse_args()

    n_train = 10 if args.small else 256

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}  |  n_train: {n_train}  |  model: {MODEL_ID}")

    print("Loading data...")
    harmful_train, _, harmless_train, _ = load_data(n_train=n_train)
    print(f"  harmful: {len(harmful_train)}, harmless: {len(harmless_train)}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    model.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    print(f"  n_layers={n_layers}, d_model={model.config.hidden_size}")

    # Build layer_modules dict: hook post_attention_layernorm (resid_mid) per layer
    module_dict = dict(model.named_modules())
    layer_modules = {
        k: module_dict[f"model.layers.{k}.post_attention_layernorm"]
        for k in range(n_layers)
    }

    batch_size = 4 if args.small else 16

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

    # Stack and normalize for cosine similarity
    layers = sorted(ref_dirs.keys())
    stack = torch.stack([ref_dirs[k] for k in layers])  # (n_layers, d_model)
    stack_normed = stack / (stack.norm(dim=-1, keepdim=True) + 1e-8)

    cosine_matrix = (stack_normed @ stack_normed.T).float().cpu().numpy()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cosine_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"Per-layer refusal direction cosine similarity\n{MODEL_ID}, n={n_train}")
    fig.tight_layout()
    out_fig = FIGURES_DIR / "persistence_cosine_matrix.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_fig}")

    # ── Identify window ───────────────────────────────────────────────────────
    window = identify_persistence_window(cosine_matrix, threshold=COSINE_THRESHOLD)

    if len(window) >= 2:
        sub = cosine_matrix[np.ix_(window, window)]
        mask = ~np.eye(len(window), dtype=bool)
        mean_cosine = float(sub[mask].mean())
    else:
        mean_cosine = float(cosine_matrix[window[0], window[0]]) if window else 0.0

    print(f"\nPersistence window W = layers {window[0] if window else 'N/A'} – {window[-1] if window else 'N/A'}")
    print(f"Window width: {len(window)}  |  Mean intra-window cosine: {mean_cosine:.3f}")

    # Decision
    if len(window) >= 8 and mean_cosine >= 0.85:
        verdict = "GREEN — good to go"
    elif len(window) >= 5 and mean_cosine >= 0.75:
        verdict = "YELLOW — proceed with caution, note as limitation"
    else:
        verdict = "RED — window too weak. Escalate to Hank before proceeding."

    if args.small:
        print(f"Decision: {verdict}  (NOTE: RED/YELLOW expected with --small; 10 prompts are too few for stable DIM directions. Run full scale for real verdict.)")
    else:
        print(f"Decision: {verdict}")

    # ── Save window ───────────────────────────────────────────────────────────
    window_data = {
        "window": window,
        "width": len(window),
        "mean_cosine": mean_cosine,
        "threshold": COSINE_THRESHOLD,
        "verdict": verdict,
        "model": MODEL_ID,
        "n_train": n_train,
    }
    window_path = ARTIFACTS_DIR / "persistence_window.json"
    with open(window_path, "w") as f:
        json.dump(window_data, f, indent=2)
    print(f"Saved: {window_path}")

    # ── Save per-layer refusal vectors ────────────────────────────────────────
    per_layer_path = ARTIFACTS_DIR / "refusal_vectors_per_layer.pt"
    torch.save({k: ref_dirs[k].cpu() for k in layers}, per_layer_path)
    print(f"Saved: {per_layer_path}")

    # ── Compute and save global vector ────────────────────────────────────────
    if window:
        r_bar = global_vector_from_window(ref_dirs, window)
        global_path = ARTIFACTS_DIR / "refusal_vector_global.pt"
        torch.save(r_bar.cpu(), global_path)
        print(f"Saved: {global_path}")
        print(f"r_bar shape: {r_bar.shape}, norm: {r_bar.norm():.4f}")
    else:
        print("WARNING: No persistence window identified. r_bar not saved.")

    print("\nDone. Deliverable for Hank:")
    print(f"  Figure: {out_fig}")
    print(f"  Window: {window_path}")


if __name__ == "__main__":
    main()
