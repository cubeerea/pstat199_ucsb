"""
Week 2, Days 4-5: GCG adversarial attack evaluation.

Appends a pre-computed universal GCG suffix (Zou et al., arXiv:2307.15043,
Appendix B Table 2) to every AdvBench harmful prompt, then re-runs all four
conditions:
  1. No steering  + GCG suffix
  2. Per-layer PID + GCG suffix
  3. Global PID   + GCG suffix
  4. Global PID + Anti-windup + GCG suffix

The llm-attacks/llm-attacks repo does NOT commit pre-computed suffixes, so the
published suffix is hardcoded in src/attacks.py. Pass --attack to choose between
gcg (default) and affirmative (prefix attack), or none (direct attack).

Prerequisites:
  - experiments/01_persistence_verification.py (artifacts/refusal_vector_global.pt,
    artifacts/persistence_window.json, artifacts/refusal_vectors_per_layer.pt)

Outputs:
  results/gcg_attack_asr.json
  artifacts/completions/<condition>.jsonl  (if --save-completions set)

Usage:
  python experiments/04_gcg_attack.py --small
  python experiments/04_gcg_attack.py
  python experiments/04_gcg_attack.py --attack affirmative
  python experiments/04_gcg_attack.py --attack none
  python experiments/04_gcg_attack.py --scale 20.0 --save-completions gcg_scale20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.attacks import ATTACK_NAMES, apply_attack
from src.controllers import (
    GlobalPIDController,
    GlobalPIDControllerAntiWindup,
    PerLayerPIDController,
)
from src.data import load_data
from src.eval import compute_asr, is_jailbreak
from src.hooks import add_hooks, get_actadd_output_hook

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = Path("artifacts")
COMPLETIONS_DIR = ARTIFACTS_DIR / "completions"
RESULTS_DIR.mkdir(exist_ok=True)
COMPLETIONS_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"


def save_completions(path: Path, prompts: list[str], completions: list[str]) -> None:
    with open(path, "w") as f:
        for prompt, completion in zip(prompts, completions):
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "is_jailbreak": is_jailbreak(completion),
            }) + "\n")
    print(f"  Saved {len(completions)} completions → {path}")


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
    parser.add_argument("--attack", choices=ATTACK_NAMES, default="gcg",
                        help="Attack to apply (default: gcg)")
    parser.add_argument("--save-completions", metavar="TAG", default=None,
                        help="Save (prompt, completion, is_jailbreak) JSONL to "
                             "artifacts/completions/<TAG>_<condition>.jsonl")
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
    if not window:
        raise ValueError(
            "Empty persistence window. Re-run: "
            "python experiments/01_persistence_verification.py --threshold 0.5 --commit"
        )

    print(f"Window W = layers {window[0]}-{window[-1]} ({len(window)} layers) | "
          f"Kp={args.kp} Ki={args.ki} Kd={args.kd} scale={args.scale} | "
          f"attack={args.attack} | model: {MODEL_ID}")

    _, harmful_raw, _, _ = load_data(n_test=n_test)
    attacked_prompts = apply_attack(harmful_raw, args.attack)
    print(f"Test set: {len(attacked_prompts)} prompts (attack={args.attack})")
    if args.attack != "none":
        print(f"  Example: {attacked_prompts[0][:120]}...")

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

    tag = args.save_completions

    # ── Condition 1: No steering + attack ────────────────────────────────────
    print(f"\n[1/4] No steering + {args.attack}")
    comps = generate_completions(
        model, tokenizer, attacked_prompts, [], batch_size, max_new_tokens, device
    )
    asr = compute_asr(comps)
    results["no_steering"] = {**asr, "attack": args.attack}
    print(f"  ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")
    if tag:
        save_completions(COMPLETIONS_DIR / f"{tag}_no_steering.jsonl", harmful_raw, comps)

    # ── Condition 2: Per-layer PID + attack ──────────────────────────────────
    print(f"\n[2/4] Per-layer PID + {args.attack}")
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
    results["perlayer_pid"] = {**asr, "attack": args.attack, "kp": args.kp, "ki": args.ki, "kd": args.kd, "scale": args.scale}
    print(f"  ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")
    if tag:
        save_completions(COMPLETIONS_DIR / f"{tag}_perlayer_pid.jsonl", harmful_raw, comps)

    # ── Condition 3: Global PID + attack ─────────────────────────────────────
    print(f"\n[3/4] Global PID + {args.attack}")
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
    results["global_pid"] = {**asr, "attack": args.attack, "kp": args.kp, "ki": args.ki, "kd": args.kd, "scale": args.scale, "window": window}
    print(f"  ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")
    if tag:
        save_completions(COMPLETIONS_DIR / f"{tag}_global_pid.jsonl", harmful_raw, comps)

    # ── Condition 4: Global PID + Anti-windup + attack ────────────────────────
    print(f"\n[4/4] Global PID + Anti-windup + {args.attack}")
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
    results["global_pid_antiwindup"] = {**asr, "attack": args.attack, "kp": args.kp, "ki": args.ki, "kd": args.kd, "scale": args.scale, "window": window}
    print(f"  ASR: {asr['asr']:.3f} ({asr['n_success']}/{asr['n_total']})")
    if tag:
        save_completions(COMPLETIONS_DIR / f"{tag}_global_pid_aw.jsonl", harmful_raw, comps)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nAttack ({args.attack}) ASR summary:")
    for cond, res in results.items():
        print(f"  {cond:<35} ASR={res['asr']:.3f}  ({res['n_success']}/{res['n_total']})")

    results["_meta"] = {
        "attack": args.attack,
        "kp": args.kp, "ki": args.ki, "kd": args.kd, "scale": args.scale,
        "window": window,
        "model": MODEL_ID,
        "n_test": len(attacked_prompts),
    }

    attack_tag = f"_{args.attack}" if args.attack != "none" else ""
    out_path = RESULTS_DIR / f"attack_asr{attack_tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
