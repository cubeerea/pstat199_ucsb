"""
Ki ablation sweep replicating the paper's gain grid.

Fixed:  Kp ∈ {1.0, 1.5},  Kd = 0
Sweep:  Ki ∈ {0.0, 0.03, 0.05, 0.10, 0.13, 0.15}

Loads Gemma-2-2B-it and pythia-70m (PPL reference) once, iterates over
(Kp, Ki) pairs. Each condition is scored by ASR + mean perplexity under
pythia-70m so coherence collapse (windup → gibberish) is visible.

Outputs:
  results/sweep_kp{kp_tag}.json         — per-condition ASR + PPL + PID norms
  figures/asr_and_ppl_vs_ki_NNN.png     — 2-panel: ASR (top) + PPL (bottom)

Usage:
  python experiments/sweep_ki.py --small
  python experiments/sweep_ki.py                        # Kp=1.0 only
  python experiments/sweep_ki.py --kp 1.0 1.5          # both Kp values
  python experiments/sweep_ki.py --ki 0.0 0.05 0.10    # custom Ki grid
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.controllers import GlobalPIDController, GlobalPIDControllerAntiWindup
from src.data import load_data
from src.eval import compute_asr
from src.hooks import add_hooks, get_actadd_output_hook
from src.perplexity import compute_perplexity, load_reference_model

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"
KI_GRID_DEFAULT = [0.0, 0.03, 0.05, 0.10, 0.13, 0.15]
KP_DEFAULT = [1.0]
KD_FIXED = 0.0


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


def run_condition(label, ctrl_cls, r_bar, kp, ki, kd, window, module_dict,
                  model, tokenizer, harmful_test, batch_size, max_new_tokens, device,
                  ref_model, ref_tokenizer):
    ctrl = ctrl_cls(r_bar=r_bar, kp=kp, ki=ki, kd=kd, window=window)
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
    ppl = compute_perplexity(completions, ref_model, ref_tokenizer, device)
    return {
        **asr,
        "perplexity": ppl,
        "kp": kp, "ki": ki, "kd": kd, "window": window,
        "p_norms": {str(k): ctrl.p_norms[k] for k in sorted(window)},
        "i_norms": {str(k): ctrl.i_norms[k] for k in sorted(window)},
        "d_norms": {str(k): ctrl.d_norms[k] for k in sorted(window)},
        "integral_norms": {str(k): ctrl.integral_norms[k] for k in sorted(window)},
    }


def main():
    parser = argparse.ArgumentParser(description="Ki sweep (paper gain grid) on Gemma-2-2B-it.")
    parser.add_argument("--small", action="store_true",
                        help="--n-test 10 --batch-size 4 --max-new-tokens 64")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--kp", type=float, nargs="+", default=KP_DEFAULT,
                        help="Proportional gain(s) to sweep (default: 1.0)")
    parser.add_argument("--ki", type=float, nargs="+", default=KI_GRID_DEFAULT,
                        help="Ki values to sweep")
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

    print(f"Device: {device} | n_test: {n_test or 'all'} | Kd={KD_FIXED} (fixed)")
    print(f"Kp grid: {args.kp}")
    print(f"Ki grid: {args.ki}")
    print(f"Total conditions: {len(args.kp) * len(args.ki) * 2} (×2 for vanilla + AW)")

    # Load artifacts
    global_path = ARTIFACTS_DIR / "refusal_vector_global.pt"
    window_path = ARTIFACTS_DIR / "persistence_window.json"
    if not global_path.exists():
        raise FileNotFoundError(f"{global_path} not found. Run 01_persistence_verification.py first.")
    r_bar = torch.load(global_path)
    with open(window_path) as f:
        window_data = json.load(f)
    window = window_data["window"]
    if not window:
        raise ValueError(
            "Empty persistence window. Re-run: "
            "python experiments/01_persistence_verification.py --threshold 0.7"
        )
    print(f"\nr_bar norm: {r_bar.norm():.4f}  |  Window W = {window[0]}–{window[-1]} "
          f"({len(window)} layers, mean cos: {window_data['mean_cosine']:.3f})")
    r_bar = r_bar.to(device)

    # Load baseline ASR for reference
    baseline_no_steer = None
    baseline_perlayer = None
    baseline_path = RESULTS_DIR / "baseline_perlayer_pid_asr.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        baseline_no_steer = bl.get("no_steering", {}).get("asr")
        baseline_perlayer = bl.get("perlayer_pid", {}).get("asr")
        if baseline_no_steer is not None:
            print(f"Baseline no-steering ASR:   {baseline_no_steer:.3f}")
        if baseline_perlayer is not None:
            print(f"Baseline per-layer PID ASR: {baseline_perlayer:.3f}")

    # Load data + Gemma
    _, harmful_test, _, _ = load_data(n_test=n_test)
    print(f"\nTest set: {len(harmful_test)} harmful prompts")
    print("Loading Gemma-2-2B-it...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)
    module_dict = dict(model.named_modules())

    # Load pythia-70m reference model for perplexity
    print("Loading pythia-70m (PPL reference)...")
    ref_model, ref_tokenizer = load_reference_model(device=device)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    all_sweep_results = {}
    HDR = f"{'Ki':>6}  {'ASR_v':>7}  {'ASR_aw':>7}  {'PPL_v':>7}  {'PPL_aw':>7}  {'#deg(v/aw)':>12}  {'I_max/P':>8}"
    SEP = "-" * len(HDR)

    for kp in args.kp:
        kp_tag = f"kp{str(kp).replace('.', '')}"
        sweep_results = {}
        print(f"\n{'='*len(HDR)}")
        print(f"Kp = {kp}  (Kd = {KD_FIXED})")
        print(HDR)
        print(SEP)

        for ki in args.ki:
            ki_tag = f"ki{str(ki).replace('.', '')}"

            vanilla = run_condition(
                "vanilla", GlobalPIDController, r_bar, kp, ki, KD_FIXED, window,
                module_dict, model, tokenizer, harmful_test, batch_size, max_new_tokens, device,
                ref_model, ref_tokenizer,
            )
            aw = run_condition(
                "antiwindup", GlobalPIDControllerAntiWindup, r_bar, kp, ki, KD_FIXED, window,
                module_dict, model, tokenizer, harmful_test, batch_size, max_new_tokens, device,
                ref_model, ref_tokenizer,
            )

            sweep_results[ki_tag] = {"ki": ki, "global_pid": vanilla, "global_pid_antiwindup": aw}

            i_max = max(vanilla["i_norms"].values()) if ki > 0 and vanilla["i_norms"] else 0.0
            p_val = list(vanilla["p_norms"].values())[0] if vanilla["p_norms"] else 0.0
            ratio = i_max / p_val if p_val > 0 else 0.0
            v_ppl = vanilla["perplexity"]["mean"]
            aw_ppl = aw["perplexity"]["mean"]
            v_deg = vanilla["perplexity"]["n_degenerate"]
            aw_deg = aw["perplexity"]["n_degenerate"]
            print(
                f"{ki:>6.2f}  {vanilla['asr']:>7.3f}  {aw['asr']:>7.3f}"
                f"  {v_ppl:>7.1f}  {aw_ppl:>7.1f}"
                f"  {v_deg:>5}/{aw_deg:<5}  {ratio:>8.3f}"
            )

        all_sweep_results[kp_tag] = {"kp": kp, "kd": KD_FIXED, "ki_grid": args.ki,
                                      "window": window, "sweep": sweep_results}

        out_path = RESULTS_DIR / f"sweep_{kp_tag}.json"
        with open(out_path, "w") as f:
            json.dump(all_sweep_results[kp_tag], f, indent=2)
        print(f"\nSaved: {out_path}")

    # ── 2-panel figure: ASR (top) + PPL (bottom) ─────────────────────────────
    run_id = next_run_id(FIGURES_DIR, "asr_and_ppl_vs_ki")
    fig, (ax_asr, ax_ppl) = plt.subplots(2, 1, sharex=True, figsize=(9, 8))
    colors = ["C0", "C1", "C2", "C3"]

    for idx, kp in enumerate(args.kp):
        kp_tag = f"kp{str(kp).replace('.', '')}"
        sweep = all_sweep_results[kp_tag]["sweep"]
        ki_vals = args.ki

        vanilla_asrs = [sweep[f"ki{str(ki).replace('.', '')}"]["global_pid"]["asr"] for ki in ki_vals]
        aw_asrs      = [sweep[f"ki{str(ki).replace('.', '')}"]["global_pid_antiwindup"]["asr"] for ki in ki_vals]
        vanilla_ppls = [sweep[f"ki{str(ki).replace('.', '')}"]["global_pid"]["perplexity"]["mean"] for ki in ki_vals]
        aw_ppls      = [sweep[f"ki{str(ki).replace('.', '')}"]["global_pid_antiwindup"]["perplexity"]["mean"] for ki in ki_vals]

        c = colors[idx % len(colors)]
        ax_asr.plot(ki_vals, vanilla_asrs, "o-",  color=c, label=f"Kp={kp} vanilla",      linewidth=2)
        ax_asr.plot(ki_vals, aw_asrs,      "s--", color=c, label=f"Kp={kp} anti-windup",  linewidth=2, alpha=0.7)
        ax_ppl.plot(ki_vals, vanilla_ppls, "o-",  color=c, label=f"Kp={kp} vanilla",      linewidth=2)
        ax_ppl.plot(ki_vals, aw_ppls,      "s--", color=c, label=f"Kp={kp} anti-windup",  linewidth=2, alpha=0.7)

    if baseline_no_steer is not None:
        ax_asr.axhline(baseline_no_steer, color="grey", linestyle=":", linewidth=1.5,
                       label=f"no steering ({baseline_no_steer:.3f})")

    ax_asr.set_ylabel("Attack Success Rate (ASR)")
    ax_asr.set_title(
        f"ASR & PPL vs Ki — Global PID  |  {MODEL_ID}\n"
        f"Kd={KD_FIXED}, scale=1.0, W={window[0]}–{window[-1]}  |  Run #{run_id:03d}"
    )
    ax_asr.legend(fontsize=8)
    ax_asr.grid(True, alpha=0.3)

    ax_ppl.set_xlabel("Ki")
    ax_ppl.set_ylabel("Mean PPL (pythia-70m ref, log scale)")
    ax_ppl.set_yscale("log")
    ax_ppl.legend(fontsize=8)
    ax_ppl.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig_path = FIGURES_DIR / f"asr_and_ppl_vs_ki_{run_id:03d}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
