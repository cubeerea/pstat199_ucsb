# Global PID Steering for LLM Jailbreak Defense

**Senior Independent Research Project** | UCSB Statistics & Data Science | Hank Sha

> 4-week timeline · Gemma-2-2B-it · ActAdd + Difference-in-Means · PyTorch / HuggingFace

---

## Experimental Claim

> A single static **global** PID controller, using a layer-averaged refusal vector $\bar{r}$, achieves Attack Success Rate (ASR) on AdvBench within ±5 percentage points of Nguyen et al.'s **per-layer** PID baseline, at substantially lower compute, on Gemma-2-2B-it.

This is the single claim being tested. Everything in this repo either directly serves it or is supporting scaffolding.

---

## Background

Nguyen et al. (ICLR 2026) reframe activation steering as a dynamical systems problem and show that popular methods (ActAdd, DirAblate, Mean-AcT) are all instances of a proportional (P) controller. They propose **PID Steering**, which constructs a steering vector at every layer $k$ as:

$$u(k) = K_p\, r(k) + K_i \sum_{j=0}^{k-1} r(j) + K_d \bigl(r(k) - r(k-1)\bigr)$$

where $r(k)$ is the difference-in-means (DIM) refusal direction at layer $k$, computed from contrastive harmful vs. harmless prompts. The I term eliminates steady-state error; the D term damps overshoot.

---

## My Extension — Global PID

Replace the per-layer target $r(k)$ with a single static global vector $\bar{r}$, computed once by averaging DIM directions over a verified **persistence window** $W$:

$$\bar{r} = \frac{1}{|W|} \sum_{k \in W} r(k), \quad W = \{k : \text{refusal feature persists at layer } k\}$$

One PID controller then uses $\bar{r}$ as the (constant) error signal across all layers in $W$.

**Why this might work:** cross-layer feature superposition — refusal features tend to persist in the residual stream across adjacent layers (Lindsey et al. crosscoders, Anthropic).

**Why it might not:** with a constant error signal, the I term accumulates against the same target at every layer, causing linear integral windup and potentially pushing activations out of distribution. A clamped anti-windup variant is included as an ablation.

**Scope hard constraints:**
- Model: **Gemma-2-2B-it only**
- Steering: **ActAdd + DIM only** (no Angular Steering, no Mean-AcT)
- Gain values: **inherited from paper** — no grid search
- ISS proof for the global case: future work

---

## Repository Layout

```
pstat199_ucsb/
├── src/                          # Shared modules (no vLLM, no TransformerLens)
│   ├── data.py                   # AdvBench (harmful) + Alpaca (harmless) loaders
│   ├── dim.py                    # DIM computation, PID recurrence, global r_bar
│   ├── hooks.py                  # HF register_forward_hook/pre_hook utilities
│   ├── controllers.py            # PerLayerPIDController, GlobalPIDController, AntiWindup
│   └── eval.py                   # ASR string-match (JailbreakBench refusal phrases)
│
├── experiments/
│   ├── 01_persistence_verification.py   # Week 1 — cosine matrix, identify window W
│   ├── 02_baseline_perlayer_pid.py      # Week 1 — replicate paper's per-layer PID ASR
│   ├── 03_global_pid.py                 # Week 2 — global PID + anti-windup ablation
│   ├── 04_attacks.py                    # Week 2 — direct + GCG attack runners (planned)
│   └── 05_capability_eval.py            # Week 3 — AlpacaEval-style benign capability check (planned)
│
├── scripts/
│   └── smoke_test.py             # Model load + hook plumbing sanity check
│
├── notes/
│   ├── repo_map.md               # Codebase map, gain values, DIM/hook locations
│   ├── gains_gemma2.md           # PID gain audit — source of Kp, Ki, Kd values
│   └── decisions.md              # Open questions and judgment calls (Q1–Q7)
│
├── Claude.md                     # Full project brief — read this for complete spec
│
├── artifacts/                    # Gitignored — saved tensors (refusal_vector_global.pt, etc.)
├── results/                      # Gitignored — ASR JSON outputs
├── figures/                      # Gitignored — plots (cosine matrix, diagnostic plots)
│
├── llm-activation-control/       # Paper's original code — retained as reference, not modified
└── Mean-AcT/                     # Paper's original code — retained as reference, not modified
```

Each experiment script accepts `--small` for fast local iteration (10 prompts, runs on CPU/MPS in minutes) and the full flag-free invocation for GPU runs.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers accelerate datasets numpy pandas \
    matplotlib seaborn tqdm requests scikit-learn jaxtyping

# Gemma-2 is a gated model — requires HF account + accepted license
# Visit https://huggingface.co/google/gemma-2-2b-it and click "Agree"
hf auth login
```

---

## Running the Experiments

Run in order. Each script is self-contained and saves outputs to `artifacts/`, `results/`, `figures/`.

```bash
# 1. Verify model loads and hooks fire (~15 seconds on MPS/CPU)
python scripts/smoke_test.py

# 2. Persistence verification — identify window W, compute r_bar
python experiments/01_persistence_verification.py --small   # pipeline check
python experiments/01_persistence_verification.py           # full run (GPU recommended)

# 3. Per-layer PID baseline — replicates paper's ASR on AdvBench
python experiments/02_baseline_perlayer_pid.py --small
python experiments/02_baseline_perlayer_pid.py

# 4. Global PID + anti-windup ablation
python experiments/03_global_pid.py --small
python experiments/03_global_pid.py
```

After step 4, `results/global_pid_asr.json` will contain the three numbers that test the experimental claim:

```json
{
  "no_steering":          { "asr": ... },
  "perlayer_pid":         { "asr": ... },
  "global_pid":           { "asr": ... },
  "global_pid_antiwindup":{ "asr": ... }
}
```

---

## Status

| Week | Goal | Status |
|------|------|--------|
| 1 | Scaffold, smoke test, persistence verification, per-layer baseline | In progress |
| 2 | Global PID implementation, direct-attack ASR results | Planned |
| 3 | Anti-windup ablation, diagnostic plots, capability eval, compute comparison | Planned |
| 4 | Writeup — all experiments frozen | Planned |

---

## Prior Work

| Work | Relevance |
|------|-----------|
| Nguyen et al. (ICLR 2026) — *Activation Steering with a Feedback Controller* ([arXiv:2510.04309](https://arxiv.org/abs/2510.04309)) | Foundation. Per-layer PID Steering. This repo forks their code. |
| Vu & Nguyen (NeurIPS 2025) — *Angular Steering* | Upstream of the forked codebase (`llm-activation-control/`). Not used in my experiments. |
| Arditi et al. (2024) — *Refusal in LLMs is mediated by a single direction* | Motivation for the single-vector global approach. |
| Lindsey et al. (Anthropic) — *Crosscoders* | Cross-layer feature persistence — theoretical grounding for why averaging over $W$ might work. |
| Park et al. (arXiv:2405.14860) — *The Linear Representation Hypothesis* | Geometric framing for why linear steering works at all. |

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
