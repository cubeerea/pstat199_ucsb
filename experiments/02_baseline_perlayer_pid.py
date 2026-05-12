"""
Week 1, Days 3-5: Replicate the paper's per-layer PID baseline on Gemma-2-2B-it.

Prerequisites:
  - Run experiments/01_persistence_verification.py first (needs artifacts/refusal_vectors_per_layer.pt)

Outputs:
  results/baseline_perlayer_pid_asr.json

Usage:
  python experiments/02_baseline_perlayer_pid.py --small   # 10 prompts
  python experiments/02_baseline_perlayer_pid.py           # full 104 test prompts
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.controllers import PerLayerPIDController
from src.data import load_data
from src.eval import compute_asr
from src.hooks import add_hooks, get_actadd_output_hook

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-2-2b-it"

# PID gains — Path A defaults from angular_steering.ipynb Cell 47
# Escalated to Hank for confirmation; see notes/gains_gemma2.md
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
    parser.add_argument("--small", action="store_true", help="Use 10 test prompts")
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

    # Load pre-computed DIM directions
    per_layer_path = ARTIFACTS_DIR / "refusal_vectors_per_layer.pt"
    if not per_layer_path.exists():
        raise FileNotFoundError(
            f"{per_layer_path} not found. Run experiments/01_persistence_verification.py first."
        )
    ref_dirs = {k: v.to(device) for k, v in torch.load(per_layer_path).items()}
    print(f"Loaded per-layer refusal directions for {len(ref_dirs)} layers.")

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
    n_layers = model.config.num_hidden_layers

    results = {}

    # ── Condition 1: No steering ───────────────────────────────────────────────
    print("\n[1/2] Baseline — no steering")
    completions_base = generate_completions(
        model, tokenizer, harmful_test, fwd_hooks=[], batch_size=batch_size,
        max_new_tokens=max_new_tokens, device=device
    )
    asr_base = compute_asr(completions_base)
    results["no_steering"] = asr_base
    print(f"  No-steering ASR: {asr_base['asr']:.3f} ({asr_base['n_success']}/{asr_base['n_total']})")

    # ── Condition 2: Per-layer PID steering ───────────────────────────────────
    print("\n[2/2] Per-layer PID steering")
    controller = PerLayerPIDController(ref_dirs, kp=KP, ki=KI, kd=KD)
    steering_dirs = controller.steering_dirs

    fwd_hooks_pid = [
        (
            module_dict[f"model.layers.{k}.post_attention_layernorm"],
            get_actadd_output_hook(steering_dirs[k].to(device), scale=1.0),
        )
        for k in range(n_layers)
        if k in steering_dirs
    ]
    completions_pid = generate_completions(
        model, tokenizer, harmful_test, fwd_hooks=fwd_hooks_pid, batch_size=batch_size,
        max_new_tokens=max_new_tokens, device=device
    )
    asr_pid = compute_asr(completions_pid)
    results["perlayer_pid"] = {**asr_pid, "kp": KP, "ki": KI, "kd": KD}
    print(f"  Per-layer PID ASR: {asr_pid['asr']:.3f} ({asr_pid['n_success']}/{asr_pid['n_total']})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nDirect-attack ASR summary:")
    print(f"  No steering:   {results['no_steering']['asr']:.3f}")
    print(f"  Per-layer PID: {results['perlayer_pid']['asr']:.3f}  (Kp={KP}, Ki={KI}, Kd={KD})")

    out_path = RESULTS_DIR / "baseline_perlayer_pid_asr.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
