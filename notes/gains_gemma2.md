# PID Gain Values — Source Audit

## Status: ESCALATION REQUIRED

No Gemma-2-2B-it specific gain values exist anywhere in the repo.
The closest available values are listed below. Decision needed from Hank on which to inherit.

---

## Gains found in codebase

### Source 1 — `angular_steering.ipynb`, Cell 47 (primary eval pipeline)

Applied inline during direction extraction (not via the `PID_control` function):

```python
# Inline PID (active code, "PID_" method prefix):
refusal_dirs = 0.9*refusal_dirs + 0.01*prefix_sum + 0.01*diff_from_first
# → Kp = 0.9, Ki = 0.01, Kd = 0.01

# Commented alternative in same cell:
# PID_control(refusal_dirs, kp=1, ki=0.1, kd=0.1)

# PI variant ("PI_" prefix):
PID_control(refusal_dirs, kp=1, ki=0.1, kd=0.0)

# PD variant ("PD_" prefix):
PID_control(refusal_dirs, kp=1, ki=0.0, kd=0.1)
```

Active model in this notebook: `Qwen/Qwen2.5-7B-Instruct`.
No model-conditional branching on gains — same values used for all models including Gemma.

### Source 2 — `llama_many_layers.py`, lines 50–52 (causal-noise per-layer path)

```python
p_coe = 1.0
i_coe = 0.3
d_coe = 0.01
```

Active model: `meta-llama/Llama-3.1-8B-Instruct`. Llama-specific, not validated on Gemma.

### Source 3 — `angular_steering_causal.ipynb`, Cell 8 (causal-noise alternative)

```python
p_coe = 1.0
i_coe = 2.5
d_coe = 0.0   # effectively PI control, no derivative term
```

Active model: `Qwen/Qwen2.5-3B-Instruct`. The large i_coe=2.5 may be Qwen-specific.

---

## Recommendation for Hank

**Use Source 1 values (Kp=0.9, Ki=0.01, Kd=0.01).**

Reasons:
- These are the gains from the main Angular Steering notebook, which is the actual eval pipeline for the paper's Table 1 results.
- They are applied identically across all models without conditioning — closest to a universal "paper default."
- `angular_steering.ipynb` has `google/gemma-2-9b-it` in its model list (commented out), so these gains were intended to cover Gemma-2.
- Source 2's i_coe=0.3 is plausible too; Source 3's i_coe=2.5 looks Qwen-tuned and should be avoided.

---

## Open question

The paper's appendix (arXiv:2510.04309) may contain per-model gain tables not reflected in the code. If Hank has the PDF, check Table A.1 or the Hyperparameters section. If explicit Gemma-2-2B gains exist there, override Source 1.
