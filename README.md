# Global PID Steering for LLM Jailbreak Defense

**Senior Independent Research Project** | UCSB Statistics & Data Science | Hank Sha

> Gemma-2-2B-it · ActAdd + Difference-in-Means · PyTorch / HuggingFace

---

## Experimental Claim

> A single static **Global PID** controller, using a layer-averaged refusal vector $\bar{r}$ applied across a verified persistence window, matches the defense effect of Nguyen et al.'s **per-layer** PID baseline on AdvBench while preserving model capability where per-layer PID does not, at substantially lower compute.

Three sub-claims, each independently supported (see [Findings](#findings)):

1. **Defense parity** — Global PID matches per-layer PID's ASR reduction on direct AdvBench attacks.
2. **Capability dominance** — At identical gains, per-layer PID induces ~96% over-refusal on benign prompts; Global PID stays at ~4%.
3. **Stability via anti-windup** — The predicted integral windup pathology manifests empirically in vanilla Global PID at Ki ≥ 0.07; the anti-windup clamp resolves it without ASR cost.

---

## Background

Nguyen et al. (ICLR 2026) reframe activation steering as a control problem and show that ActAdd, DirAblate, and Mean-AcT are all instances of a proportional (P) controller. They propose **PID Steering**, which constructs a steering vector at every layer $k$ as:

$$u(k) = K_p\, r(k) + K_i \sum_{j=0}^{k-1} r(j) + K_d \bigl(r(k) - r(k-1)\bigr)$$

where $r(k)$ is the difference-in-means (DIM) refusal direction at layer $k$, computed from contrastive harmful vs. harmless prompts.

---

## Extension — Global PID

Replace the per-layer target $r(k)$ with a single static global vector $\bar{r}$, computed once by averaging DIM directions over a verified **persistence window** $W$:

$$\bar{r} = \frac{1}{|W|} \sum_{k \in W} r(k), \quad W = \{k : \text{refusal direction persists across layer } k\}$$

One PID controller then uses $\bar{r}$ as the (constant) error signal across all layers in $W$.

**Why this might work:** cross-layer feature superposition — refusal features persist across adjacent residual-stream layers in late-block transformers (Lindsey et al., Anthropic).

**Why it might fail:** with a constant error signal, the I-term accumulates against the same target at every layer, causing linear integral windup and potentially pushing activations out of distribution. A clamped anti-windup variant is included as the safety net.

**Scope (hard constraints from CLAUDE.md):**
- Model: **Gemma-2-2B-it only**
- Steering: **ActAdd + DIM only**
- Gain values: **inherited from paper** — no grid search

---

## Findings

All numbers below from runs at the calibrated operating scale **α = 50** on the validated persistence window **W = [19, 20, 21, 22, 23, 24, 25]** (7 layers, threshold = 0.5).

### 1. Persistence window (`figures/persistence_cosine_matrix_*.png`)
Per-layer DIM directions in Gemma-2-2B-it are locally coherent in the late block (L19–L25, mean intra-window cosine ≥ 0.5) but **not globally distributed across the network**. The first half of the model contributes effectively no refusal-direction signal.

### 2. Scale calibration (`figures/scale_calibration.png`)
Measurable activation perturbation (>5% relative window divergence) requires **α ≥ 50**. Below this, the steering vector is sub-threshold relative to the residual stream magnitude (~400 at L19). Critically, no coherence blowup is detected even at α = 100 — there is no upper bound from the residual-stream norm criterion within the swept range.

### 3. Ki sweep at α=50 (`figures/asr_and_ppl_vs_ki_*.png`)
**Windup hypothesis empirically validated.**

- ASR: pinned at 0/104 for both vanilla and anti-windup across Ki ∈ {0, 0.03, 0.05, 0.10, 0.13, 0.15} — Global PID reduces ASR from the no-steering baseline (1/104 = 0.96%) to zero at any Ki.
- PPL (vanilla): U-shape with explosion. PPL ≈ 18 → 11 (Ki = 0.05, minimum) → 45 (Ki = 0.15). The integral accumulates against a static target and drives activations OOD past Ki ≈ 0.07.
- PPL (anti-windup): bounded in [12, 18] across the entire Ki sweep. The clamp at `2·||r̄||·Ki` prevents the explosion exactly as predicted.

**Anti-windup is therefore necessary, not a marginal improvement.**

### 4. Capability eval at matched gains (`figures/capability_eval_*.png`)
At Kp = 0.9, Ki = 0.01, Kd = 0.01 — the same gains as the main ASR run — on 200 benign Alpaca prompts:

| Condition | Over-refusal rate | PPL | Verdict |
|---|---|---|---|
| no_steer | 2% | 37 | baseline |
| **perlayer** | **96%** | **7 (degenerate)** | **DISQUALIFIED** |
| global | 4% | 37 | ✓ |
| global + AW | 4% | 37 | ✓ |

**Per-layer PID lobotomizes the model at the paper's published gain regime.** The model refuses 96% of benign instructions ("Write a poem about autumn") because its 26 stacked steering hooks inject ~3.7× more total energy than Global PID's 7. Global PID maintains capability essentially unchanged from baseline.

### 5. Mechanism plots (`figures/iterm_magnitude_vs_layer_*.png`, `activation_norm_vs_layer_*.png`)
- I-term magnitude grows linearly across W under vanilla (0.006 → 0.043 over 7 layers); the AW clamp pins it flat at 0.012.
- D-term is identically zero across W by design (the static error signal makes `e(k) − e(k−1) = 0`).
- Activation norm in W diverges visibly from no-steering at α = 50 but stays bounded — no exponential growth.

---

## Repository Layout

```
pstat199_ucsb/
├── src/
│   ├── attacks.py            # GCG / affirmative-prefix attack strings (Zou et al. Appendix B)
│   ├── controllers.py        # PerLayer + Global PID + AntiWindup (all support sign=±1)
│   ├── data.py               # AdvBench harmful + Alpaca harmless loaders
│   ├── dim.py                # DIM computation, PID recurrence, global r_bar from W
│   ├── eval.py               # ASR + over-refusal substring scorer (JailbreakBench list)
│   ├── hooks.py              # HF register_forward_hook utilities
│   └── perplexity.py         # pythia-70m reference PPL + degeneracy detection
│
├── experiments/
│   ├── 01_persistence_verification.py   # Cosine matrix, identify window W, save r_bar
│   ├── 02_baseline_perlayer_pid.py      # Per-layer PID ASR baseline
│   ├── 03_global_pid.py                 # Global PID + AW + activation diagnostics
│   ├── 04_gcg_attack.py                 # GCG / affirmative attack evaluation
│   ├── 05_capability_eval.py            # Over-refusal on benign Alpaca prompts
│   ├── 06_sweep_ki.py                   # Ki sweep at fixed scale (windup test)
│   └── 07_scale_calibration.py          # Find α_min / α_break for steering visibility
│
├── scripts/
│   ├── smoke_test.py            # Model load + hook plumbing sanity check
│   └── compute_benchmark.py     # Wall-clock latency per forward pass (CLAUDE.md §4.8)
│
├── run_all_experiments.sh       # Sequential pipeline (logs to logs/)
├── requirements.txt             # Pinned dependencies
│
├── notes/
│   ├── repo_map.md              # Forked codebase map
│   ├── gains_gemma2.md          # PID gain audit
│   └── decisions.md             # Open questions and judgment calls (Q1–Q7)
│
├── artifacts/                   # persistence_window.json, refusal_vector_global.pt, completions/
├── results/                     # ASR JSON outputs per experiment
├── figures/                     # All paper figures (numbered runs)
│
├── llm-activation-control/      # Forked paper code — reference only
└── Mean-AcT/                    # Forked paper code — reference only
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Gemma-2 is a gated model — requires HF account + accepted license
# Visit https://huggingface.co/google/gemma-2-2b-it and click "Agree"
huggingface-cli login
```

---

## Running the Experiments

All experiment scripts share a uniform CLI surface:

```
--small                         Fast iteration defaults (n_test=10, batch=4, max_tokens=64)
--device {cuda,mps,cpu}         Auto-detected if omitted
--n-test N                      Test prompts (default: 104 harmful / 200 benign for cap eval)
--batch-size N                  Generation batch size
--max-new-tokens N              Generation length cap
--kp / --ki / --kd FLOAT        PID gains
--scale FLOAT                   Steering vector α (CALIBRATED VALUE: 50)
--sign {1,-1}                   1 = toward refusal; -1 = reverse-direction sanity check
--attack {none,gcg,affirmative} Attack augmentation
--save-completions TAG          Dump (prompt, completion, is_jailbreak) JSONL per condition
--tag SUFFIX                    Append to output filename
```

(Exceptions: `01` has its own DIM-extraction args; `07` uses `--scales` for the sweep grid.)

### Quick full pipeline

```bash
bash run_all_experiments.sh
```

Runs experiments 01–07 in order, logging each to `logs/experiment_<N>_<timestamp>.log`. Stops on first failure.

### Manual run

```bash
# 1. Sanity check (~15 s on MPS/CPU)
python scripts/smoke_test.py

# 2. Persistence — identify W and commit r_bar to canonical artifact path
python experiments/01_persistence_verification.py --threshold 0.5 --commit

# 3. Find the operating scale
python experiments/07_scale_calibration.py
# → records scale_min / scale_break / recommended in results/scale_calibration.json

# 4. Per-layer baseline + Global PID main run (use calibrated scale)
python experiments/02_baseline_perlayer_pid.py --scale 50
python experiments/03_global_pid.py --scale 50

# 5. Attack evaluation (gcg + affirmative prefix)
python experiments/04_gcg_attack.py --scale 50 --attack gcg
python experiments/04_gcg_attack.py --scale 50 --attack affirmative

# 6. Capability eval — matched gains to main run
python experiments/05_capability_eval.py --scale 20 --kp 0.9 --ki 0.01 --kd 0.01

# 7. Ki sweep at the operating scale (windup test)
python experiments/06_sweep_ki.py --scale 50 --kp 0.9

# 8. Wall-clock comparison
python scripts/compute_benchmark.py --scale 50

# Sanity check: steering away from refusal should INCREASE ASR
python experiments/03_global_pid.py --scale 50 --sign -1
```

### Artifact hygiene

`01_persistence_verification.py` writes **versioned** artifacts by default (`persistence_window_<thresh>_<extract>.json`). Pass `--commit` to overwrite the canonical `persistence_window.json` and `refusal_vector_global.pt` consumed by downstream experiments. This prevents the accidental-overwrite bug that produced the earlier degenerate results.

---

## Key result files

After a clean run, the headline numbers live in:

- `results/global_pid_asr.json` — main ASR table (no-steer, per-layer, global, global+AW)
- `results/capability_eval.json` — over-refusal table at matched gains
- `results/sweep_kp09.json` — Ki sweep diagnostics (ASR + PPL across Ki grid)
- `results/scale_calibration.json` — `scale_min`, `scale_break`, recommended α
- `results/compute_benchmark.json` — wall-clock latency per condition
- `results/attack_asr_gcg.json` / `attack_asr_affirmative.json` — under-attack ASR

Plus all numbered figures in `figures/` and per-condition completions in `artifacts/completions/`.

---

## Status

| Phase | Goal | Status |
|---|---|---|
| Persistence | Identify W in Gemma-2-2B-it | ✓ Complete (L19–L25, θ=0.5) |
| Baseline | Per-layer PID ASR replication | ✓ Complete |
| Global PID | Main controller + AW ablation | ✓ Complete |
| Attacks | GCG / affirmative-prefix runs | ✓ Complete |
| Capability | Benign Alpaca over-refusal | ✓ Complete (per-layer DISQUALIFIED) |
| Ki sweep | Windup hypothesis test | ✓ Complete (validated at α=50) |
| Compute | Wall-clock comparison | ✓ Complete |
| Writeup | Paper | In progress |

---

## Prior Work

| Work | Relevance |
|---|---|
| Nguyen et al. (ICLR 2026) — *Activation Steering with a Feedback Controller* ([arXiv:2510.04309](https://arxiv.org/abs/2510.04309)) | Foundation. Per-layer PID Steering. This repo forks their code. |
| Vu & Nguyen (NeurIPS 2025) — *Angular Steering* | Upstream of the forked codebase (`llm-activation-control/`). Not used. |
| Arditi et al. (2024) — *Refusal in LLMs is mediated by a single direction* | Motivation for the single-vector global approach. |
| Lindsey et al. (Anthropic) — *Crosscoders* | Cross-layer feature persistence — theoretical grounding for averaging over W. |
| Park et al. ([arXiv:2405.14860](https://arxiv.org/abs/2405.14860)) — *Linear Representation Hypothesis* | Geometric framing for linear steering. |
| Zou et al. (2023) — *Universal and Transferable Adversarial Attacks* ([arXiv:2307.15043](https://arxiv.org/abs/2307.15043)) | GCG suffix used in `src/attacks.py` (Appendix B). |

---

## Citation

If you use this work, please cite the paper it builds on:

```bibtex
@inproceedings{nguyen2026pidsteering,
  title     = {Activation Steering with a Feedback Controller},
  author    = {Dung V. Nguyen and Nhi Y. Pham and Hieu M. Vu and Lei Zhang and Tan M. Nguyen},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
