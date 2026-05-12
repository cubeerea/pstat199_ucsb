# CLAUDE.md — Global PID Steering Project

> **Audience:** Claude Code, executing inside Hank's fork of `dungnvnus/pid-steering`.
> **Goal:** Extend the per-layer PID Steering baseline (Nguyen et al., ICLR 2026) with a **Global PID controller** using a single layer-averaged refusal vector, and benchmark jailbreak defense on Gemma-2-2B-it against AdvBench.
> **Owner:** Hank Sha (UCSB Statistics & Data Science). Senior independent research project. **4-week timeline.**

---

## 0. Mission Statement (read this first, every session)

You are helping Hank ship a senior research project in 4 weeks from a cold start. Your job is to be the engineering force-multiplier: write working code, run experiments, surface failures fast, and never let scope creep.

**The single experimental claim we are testing:**

> A single static **global** PID controller, using a layer-averaged refusal vector $\bar{r}$, achieves Attack Success Rate (ASR) on AdvBench within ±5 percentage points of Nguyen et al.'s **per-layer** PID baseline, at substantially lower compute, on Gemma-2-2B-it.

Everything else is supporting evidence or future work. If a task does not directly serve this claim, defer it.

**Hard constraints:**
- Model: **Gemma-2-2B-it only.** Do not branch to other models without explicit approval.
- Steering paradigm: **ActAdd + difference-in-means only.** Do not touch Angular Steering or Mean-AcT.
- Gain selection: **inherit from the paper** (Path A). No grid search. No tuning. If the paper's gains for Gemma-2 are not recoverable, escalate to Hank immediately.
- Timeline: 4 weeks total. Weekly checkpoints in §6.

---

## 1. Background — What This Project Is

### 1.1 The paper we're extending

Nguyen et al., *Activation Steering with a Feedback Controller* (ICLR 2026, arXiv:2510.04309).

Their core method: at every transformer layer $k$, compute a steering control signal via PID:

$$u(k) = K_p \, r(k) + K_i \sum_{j=0}^{k-1} r(j) + K_d \big(r(k) - r(k-1)\big)$$

where $r(k)$ is the difference-in-means (DIM) error signal at layer $k$ between contrastive datasets (harmful vs. benign). The control signal $u(k)$ is added into the residual stream at layer $k$.

They prove input-to-state stability (ISS) and demonstrate empirically that PID reduces steady-state error and overshoot vs. plain proportional control (which is what ActAdd, DirAblate, and Mean-AcT all reduce to in their framing).

### 1.2 Hank's extension

Replace the per-layer target $r(k)$ with a **single static global vector** $\bar{r}$ computed once via layer averaging over a verified persistence window:

$$\bar{r} = \frac{1}{|W|} \sum_{k \in W} r(k), \quad W = \{k : \text{refusal feature persists at layer } k\}$$

Then apply one PID controller using $\bar{r}$ as the (static) target across all layers in $W$.

**Why this might work:** cross-layer superposition — features tend to persist in the residual stream across adjacent layers (Lindsey et al. crosscoders work, Anthropic; CLVQ-VAE).

**Why it might not:** static target means the integral term accumulates against the same error repeatedly. Risk of integral windup → activations pushed out-of-distribution → coherence collapse.

### 1.3 What we are NOT doing

- Not building on Angular Steering (their `llm-activation-control/` setup wraps Angular Steering — we strip that out and use plain ActAdd).
- Not re-deriving the ISS theorem for the global case (mention as future work in writeup).
- Not running adaptive attacks against our own defense.
- Not testing on multiple models.
- Not using TransformerLens — we stay on their HuggingFace stack to leverage their existing code.
- Not using vLLM or Ollama for any part of this — they do not expose hooks. Plain `transformers` + PyTorch `register_forward_hook` only.

---

## 2. First Tasks — Repo Reconnaissance

Before writing any code, you must understand what is already in the fork. Do this first, every fresh session if needed.

### 2.1 Map the repo

Run these and report back to Hank what you find:

```bash
# From repo root
find . -type f \( -name "*.py" -o -name "*.ipynb" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.sh" -o -name "README*" -o -name "requirements*.txt" -o -name "pyproject.toml" \) | grep -v __pycache__ | grep -v .git | head -100

# Focus on the subfolder we care about
ls -la llm-activation-control/
cat llm-activation-control/README.md 2>/dev/null || echo "No README in subfolder"
```

### 2.2 Find the gain values

The single most important thing to locate in their code: the actual $(K_p, K_i, K_d)$ values for Gemma-2-2B.

Search aggressively:

```bash
# Search for hardcoded gain values
grep -rn -E "K_?p|K_?i|K_?d|kp|ki|kd" llm-activation-control/ --include="*.py" --include="*.ipynb" --include="*.yaml" --include="*.json"

# Search for config files
find llm-activation-control/ -name "config*" -o -name "*gemma*" -o -name "*pid*"

# Search notebooks for gain assignments
grep -rn -E "= [0-9]+\.[0-9]+" llm-activation-control/ --include="*.ipynb" | grep -iE "p_gain|i_gain|d_gain|kp|ki|kd|proportional|integral|derivative" | head -30
```

**If you find them:** record them verbatim in `notes/gains_gemma2.md` with the exact source file and line number. These are your baseline.

**If you do not find them:**
1. Check the paper's appendix (Hank has the PDF — ask him to share or check arXiv).
2. Check if there's a `hydra` or `argparse` default somewhere that loads from a config we missed.
3. **STOP and escalate to Hank.** Do not invent values. Do not grid search.

### 2.3 Find the DIM computation

Their difference-in-means logic is the foundation for both their per-layer $r(k)$ and our global $\bar{r}$. Find it:

```bash
grep -rn -E "diff.in.means|difference.in.means|mean.diff|harmful.*benign|contrastive" llm-activation-control/ --include="*.py" --include="*.ipynb"
```

Report back: which file/function computes DIM, what layer(s) it operates on, and what format the result takes (tensor shape, normalization, etc.).

### 2.4 Find the hook injection point

Find where they actually modify the residual stream:

```bash
grep -rn -E "register_forward_hook|register_forward_pre_hook|hook_fn|residual" llm-activation-control/ --include="*.py" --include="*.ipynb"
```

Report which layer module they hook (likely something like `model.model.layers[k]` for Gemma-2), and whether they hook the input or output of the residual stream block.

### 2.5 Find the ASR evaluation

```bash
grep -rn -E "ASR|attack.success|refusal.match|jailbreak.*eval" llm-activation-control/ --include="*.py" --include="*.ipynb"
```

Report: what refusal-phrase list they use, what model/method scores ASR (string match vs. LLM judge), and where the eval dataset is loaded.

### 2.6 Output of reconnaissance

After §2.1–§2.5, write a single file `notes/repo_map.md` summarizing:

1. Entry points (which notebook/script runs the baseline end-to-end).
2. Gain values for Gemma-2 (with source).
3. DIM function location.
4. Hook function location.
5. ASR evaluation location.
6. Anything that is broken, missing, or unclear.

This file is your map. Update it as you learn more.

---

## 3. Environment Setup

### 3.1 Python environment

```bash
# From repo root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Then install whatever their `requirements.txt` or notebooks demand. Likely:

```bash
pip install torch transformers accelerate datasets numpy pandas matplotlib seaborn tqdm jupyter
```

For Gemma-2 access, Hank will need to:
1. Have a HuggingFace account.
2. Accept the Gemma-2 license at https://huggingface.co/google/gemma-2-2b-it.
3. Run `huggingface-cli login` with a token that has read access.

If you hit a 401/403 on model download, this is the cause. Stop and tell Hank.

### 3.2 Local-first verification

Hank wants to verify experiments run end-to-end locally before paying for Lightning AI GPU time. Local target: Mac (MPS) or Linux CPU. Gemma-2-2B is ~5GB in float16 — it fits in 16GB RAM but inference will be slow.

**Local smoke test (do this in Week 1 Day 1):**

```python
# scripts/smoke_test.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_id = "google/gemma-2-2b-it"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
model.eval()

print(f"Model loaded. Layers: {model.config.num_hidden_layers}, d_model: {model.config.hidden_size}")

# Verify hook plumbing
captured = {}
def hook(module, input, output):
    captured['shape'] = output[0].shape if isinstance(output, tuple) else output.shape

h = model.model.layers[10].register_forward_hook(hook)

prompt = "Hello, how are you?"
inputs = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    _ = model(**inputs)

h.remove()
print(f"Layer 10 output shape: {captured['shape']}")
print("Smoke test PASSED" if 'shape' in captured else "Smoke test FAILED")
```

If this runs cleanly on Hank's laptop, the hook plumbing works and we can scale to GPU later with one line change.

### 3.3 GPU later

Once local smoke test passes and experiments are working at 10-prompt scale: move to Lightning AI Studio (single A10G or L4, ~$0.50–1.00/hr). The same code runs by changing `device = "cuda"`.

Estimate: 40-60 GPU-hours for full project. Budget $50-100. Hank is also checking UCSB cluster access.

---

## 4. Implementation Plan

This section is the engineering backbone. Implement in this order. **Do not skip steps.**

### 4.1 Persistence verification (Week 1, Days 1-2)

Before any controller is built, verify that the refusal feature actually persists across layers in Gemma-2-2B. If it doesn't, the "global vector" framing collapses and we replan.

Script: `experiments/01_persistence_verification.py`

Pseudo-code:

```python
# 1. Load Gemma-2-2B-it.
# 2. Load 256 ALPACA benign prompts + 256 AdvBench harmful prompts.
# 3. For each prompt, run forward pass and capture the residual stream output
#    at the LAST TOKEN position for every layer k = 0...N-1.
# 4. For each layer k:
#       r(k) = mean(harmful_activations[k]) - mean(benign_activations[k])
# 5. Build cosine similarity matrix C[i,j] = cos(r(i), r(j)).
# 6. Plot C as a heatmap. Save to figures/persistence_cosine_matrix.png.
# 7. Identify the contiguous window W = [k_a, k_b] where C[i,j] > 0.8
#    for all i,j in W. Save W to notes/persistence_window.json.
# 8. Compute the global vector: r_bar = mean(r(k) for k in W).
# 9. Save r_bar as a .pt file: artifacts/refusal_vector_global.pt.
# 10. Also save individual r(k) tensors: artifacts/refusal_vectors_per_layer.pt.
```

**Decision criteria after running this:**

- Window width ≥ 8 layers, mean intra-window cosine ≥ 0.85 → green light, proceed.
- Window width 5-7 layers OR mean cosine 0.75-0.85 → yellow, proceed but note as limitation.
- Window width < 5 layers OR mean cosine < 0.75 → red. Stop. The averaging premise is weak. Escalate to Hank.

**Deliverable for Hank:** the cosine heatmap figure + a one-paragraph summary of what the window looks like. This figure will appear in his final writeup regardless of outcome.

### 4.2 Baseline replication (Week 1, Days 3-5)

**Do not build the global controller yet.** First, run their per-layer PID baseline on Gemma-2-2B with plain ActAdd (strip out Angular Steering if present). Confirm we can reproduce their ASR ballpark on AdvBench direct attacks.

Script: `experiments/02_baseline_perlayer_pid.py`

Tasks:
1. Use the DIM function found in §2.3 to compute $r(k)$ at every layer.
2. Apply per-layer PID with gains from §2.2.
3. Inject control signal $u(k)$ into the residual stream via the hook function from §2.4.
4. Generate completions for the 520 AdvBench harmful behaviors.
5. Score ASR using the refusal-match list from §2.5.
6. Output: `results/baseline_perlayer_pid_asr.json` containing `{n_prompts, n_attacks_succeeded, asr, mean_completion_length}`.

**Sanity check:** ASR with no steering on Gemma-2-2B-it should be very low (probably <10%) because Gemma-2 is already aligned. ASR under attack (no steering) should be higher. Their PID Steering claim is that ASR goes DOWN when you steer toward refusal.

If you cannot get a sensible baseline number in 2 days, escalate. Do not move on.

### 4.3 Global PID implementation (Week 2, Days 1-3)

Now the actual contribution.

Script: `experiments/03_global_pid.py`

The Global PID controller is conceptually simpler than per-layer:

```python
class GlobalPIDController:
    """
    Single PID controller using a static global refusal vector r_bar.
    Applied at every layer in the persistence window W.
    """
    def __init__(self, r_bar, K_p, K_i, K_d, window):
        self.r_bar = r_bar          # tensor, shape (d_model,)
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.window = window         # list of layer indices [k_a, ..., k_b]
        self.reset()

    def reset(self):
        self.integral = torch.zeros_like(self.r_bar)
        self.prev_error = None       # for derivative term

    def step(self, layer_idx, current_activation_at_last_token):
        # Error: how far is the current activation from being "refusing"?
        # In their setup, e(k) = r(k); in ours, e(k) = r_bar (constant).
        # But to make the PID meaningful, we use the projection of current state.
        # CONFIRM with Hank: do we use r_bar directly, or do we project current
        # activations onto r_bar and use that as error signal?
        e = self.r_bar  # placeholder — see open question Q1

        # P term
        p_term = self.K_p * e

        # I term — accumulates error across layers
        self.integral = self.integral + e
        i_term = self.K_i * self.integral

        # D term — first difference
        if self.prev_error is None:
            d_term = torch.zeros_like(e)
        else:
            d_term = self.K_d * (e - self.prev_error)
        self.prev_error = e.clone()

        u = p_term + i_term + d_term
        return u
```

**Open question Q1 (escalate to Hank before implementing):** Is the error signal $e(k)$ just $\bar{r}$ (constant), or is it the projection of the current residual stream onto $\bar{r}$ (state-dependent)? Look at how the paper defines $e(k)$ in the per-layer case and mirror that structure for the global case. The two choices produce very different dynamics:

- **Constant error** ($e(k) = \bar{r}$): the I term grows linearly with layer index, causing aggressive windup. This is the "naive" version and probably what Hank had in mind.
- **State-dependent error** ($e(k) = \bar{r} - \text{proj}_{\bar{r}}(h_k)$): the I term self-corrects as activations align with $\bar{r}$. This is the proper control-theoretic version.

Default to **constant error** unless Hank says otherwise — it matches his framing of "the I term will accumulate errors against the exact same target repeatedly" and lets us test his integral-windup hypothesis cleanly.

### 4.4 Anti-windup ablation (Week 3, Day 1)

Implement a simple clamp on the I term:

```python
# In GlobalPIDController.step():
i_term = self.K_i * self.integral
i_term = torch.clamp(i_term, min=-CLAMP_LIMIT, max=CLAMP_LIMIT)
```

`CLAMP_LIMIT` is `2.0 * ||r_bar||` as a starting point — tune if needed.

This is a single ablation, not the main method. Run both with and without anti-windup. Report both numbers in the results table.

### 4.5 GCG attack (Week 2, Days 4-5 — drop if Week 1 slipped)

Use pre-computed GCG suffixes from Zou et al.'s released set. **Do not run GCG optimization yourself** — it's compute-prohibitive for this timeline.

Source: https://github.com/llm-attacks/llm-attacks has released suffixes. Find a transferable suffix set for Gemma-2 or use universal suffixes.

Append the suffix to each AdvBench prompt, re-run ASR evaluation under all three conditions (no steering, per-layer PID, global PID, global PID + anti-windup).

**If Week 1 slipped past Day 5 of Week 1, skip GCG entirely.** Direct attack is enough for the paper.

### 4.6 Capability evaluation (Week 3, Days 2-3)

Run AlpacaEval-style evaluation on 200 benign instructions to verify the model isn't lobotomized. Use Claude or GPT-4 as judge if API access is available; otherwise, use a small open-source judge model or do a manual review of a 50-prompt subset.

Script: `experiments/05_capability_eval.py`

Output: win-rate / quality score for each steering condition vs. unsteered baseline.

**Decision rule:** if any steering condition drops AlpacaEval win-rate by >20pp relative to unsteered, that condition is broken regardless of ASR.

### 4.7 Diagnostics (Week 3, Days 4-5)

Two plots to support the writeup:

**Plot A: Activation norm vs. layer.** For each controller condition, plot $\|h^{(k)}\|$ across layers averaged over a held-out set. Bounded norm = empirical ISS. Exploding norm = instability.

**Plot B: Integral term magnitude vs. layer.** For the global PID specifically, plot $\|I\text{-term}\|$ across layers. Linear growth confirms Hank's windup hypothesis. Sublinear or bounded growth suggests anti-windup isn't needed.

These plots are figures 2 and 3 of the writeup. Make them publication-quality.

### 4.8 Compute comparison (Week 3 ongoing)

Track wall-clock per forward pass for each controller. The "computationally simpler" claim needs a number.

```python
import time
# Per layer over 100 prompts
t0 = time.time()
for prompt in prompts[:100]:
    run_with_perlayer_pid(prompt)
t_perlayer = (time.time() - t0) / 100

t0 = time.time()
for prompt in prompts[:100]:
    run_with_global_pid(prompt)
t_global = (time.time() - t0) / 100

print(f"Per-layer: {t_perlayer*1000:.2f} ms/prompt")
print(f"Global: {t_global*1000:.2f} ms/prompt")
print(f"Speedup: {t_perlayer/t_global:.2f}x")
```

---

## 5. File Layout (proposed)

```
pid-steering/                           # the fork root
├── llm-activation-control/             # untouched (their code)
├── Mean-AcT/                           # untouched
├── notes/                              # NEW — your scratchpad
│   ├── repo_map.md                     # output of §2
│   ├── gains_gemma2.md                 # gain values verbatim
│   ├── persistence_window.json
│   └── decisions.md                    # any judgment calls you made
├── experiments/                        # NEW — Hank's experiments
│   ├── 01_persistence_verification.py
│   ├── 02_baseline_perlayer_pid.py
│   ├── 03_global_pid.py
│   ├── 04_attacks.py                   # direct + GCG attack runners
│   └── 05_capability_eval.py
├── src/                                # NEW — shared code
│   ├── __init__.py
│   ├── controllers.py                  # PerLayerPID, GlobalPID
│   ├── hooks.py                        # residual stream hook utilities
│   ├── dim.py                          # difference-in-means
│   ├── eval.py                         # ASR scoring
│   └── data.py                         # ALPACA, AdvBench loaders
├── artifacts/                          # NEW — saved tensors
│   ├── refusal_vector_global.pt
│   ├── refusal_vectors_per_layer.pt
│   └── completions/
├── results/                            # NEW — JSON outputs
│   └── *.json
├── figures/                            # NEW — plots
│   └── *.png
├── scripts/
│   └── smoke_test.py                   # §3.2
└── CLAUDE.md                           # this file
```

Do not commit `artifacts/` or model weights to git — add to `.gitignore`.

---

## 6. Weekly Checkpoints

Hank has 4 weeks total. Each week ends Friday EOD with a concrete deliverable.

### Week 1 — Foundation
**By end of Week 1, you should have:**
- Working local smoke test on Gemma-2-2B-it (`scripts/smoke_test.py`).
- `notes/repo_map.md` complete.
- Gain values for Gemma-2 recovered and recorded.
- Persistence verification run; cosine heatmap saved; window $W$ identified.
- Per-layer PID baseline reproduces a sensible ASR number on AdvBench direct attack (even if just on 50 prompts).

**Red flags by end of Week 1:**
- Cannot find gain values → drop "Path A" assumption, replan with Hank.
- Persistence window < 5 layers → the global framing is in trouble.
- Cannot reproduce paper's baseline ASR within 10pp → something is wrong with the pipeline.

### Week 2 — Core method
**By end of Week 2:**
- Global PID controller implemented and running.
- Direct-attack ASR results for: (a) no steering, (b) per-layer PID, (c) global PID. Three numbers.
- GCG attack added if time permits.

### Week 3 — Diagnostics & ablations
**By end of Week 3:**
- Anti-windup ablation run.
- Activation-norm and integral-magnitude diagnostic plots done.
- Capability eval (AlpacaEval-style) run on at least 100 benign prompts.
- Compute comparison numbers.

### Week 4 — Writeup
**By end of Week 4:**
- All experiments frozen. No new methods.
- Final figures generated.
- Hank writes the paper. Your job in Week 4 is to answer questions and re-run small things — not to add features.

---

## 7. Open Questions — Always Surface These

When you encounter ambiguity, **do not invent an answer**. Surface it to Hank with the options and your recommendation. Track running questions in `notes/decisions.md`.

Current open questions:

- **Q1:** Error signal for Global PID — constant $\bar{r}$ or state-dependent projection? (See §4.3.) **Default: constant.**
- **Q2:** Where exactly in the residual stream to inject — pre or post layernorm? Match what the paper does. (See §2.4.)
- **Q3:** Token position for DIM computation and steering — last token, all tokens, generation tokens only? Match the paper.
- **Q4:** AdvBench split — use all 520 harmful behaviors or a subset? Recommend: all 520 for final numbers, 50 for iteration.
- **Q5:** ASR scoring — string-match vs. LLM judge? Recommend: string-match for primary, LLM-judge spot check on 50 samples to validate.

---

## 8. Working Style — Rules of Engagement

### 8.1 What "done" means

Every script in `experiments/` must:
- Run end-to-end without errors when invoked as `python experiments/0X_*.py`.
- Take a `--small` flag for fast iteration (10 prompts, no GPU needed).
- Save outputs to deterministic paths (`results/`, `figures/`, `artifacts/`).
- Log key numbers to stdout so Hank can verify by reading terminal output.

### 8.2 When to stop and ask

Stop and ask Hank before:
- Adding any model beyond Gemma-2-2B-it.
- Adding any steering paradigm beyond ActAdd + DIM.
- Tuning gain values.
- Spending more than 4 hours on a single bug.
- Making any non-trivial change to `llm-activation-control/` — prefer copying their code into `src/` and modifying there, leaving their code untouched.

### 8.3 Commit hygiene

- Small commits per experiment script.
- Commit messages: present-tense, imperative. "Add persistence verification" not "Added".
- Push at end of every working session so Hank can review.

### 8.4 When something fails

Failure modes in priority order:
1. Pipeline broken (can't even run) → fix immediately, this blocks everything.
2. Numbers look wrong (ASR negative, NaN losses, etc.) → debug, do not paper over.
3. Numbers look plausible but unexpected → record, do not "fix" by tweaking until they match expectations.
4. Method underperforms baseline → this is a real result, not a failure. Record it honestly. Hank can write about why it failed.

A **honest null result is publishable for a senior project**. A fabricated positive result is not. If Global PID is worse than per-layer PID, that's the finding.

---

## 9. Communication With Hank

- Hank is a UCSB senior with strong ML background. He knows the theory. Skip handholding on transformer internals or PID basics.
- He prefers concise, direct, technically precise communication. No hedging. No "I think maybe perhaps." Just facts and recommendations.
- When you have results, lead with the number. "Direct-attack ASR: 12% (global), 14% (per-layer), 67% (unsteered)." Then context.
- When you hit a blocker, state it in one sentence. Then options. Then your recommendation.

---

## 10. Reference

- Paper: arXiv:2510.04309 — *Activation Steering with a Feedback Controller* (Nguyen, Pham, Vu, Zhang, Nguyen — ICLR 2026)
- Their repo: https://github.com/dungnvnus/pid-steering (you are inside Hank's fork of this)
- Related — cross-layer features: https://transformer-circuits.pub/2024/crosscoders/index.html
- Related — refusal direction in LLMs: Arditi et al. 2024 (refusal is a single direction)
- Related — geometric view: Park et al. arXiv:2405.14860 (linear representation hypothesis geometry)
- AdvBench: https://github.com/llm-attacks/llm-attacks
- ALPACA: https://github.com/tatsu-lab/stanford_alpaca
- Gemma-2-2B-it: https://huggingface.co/google/gemma-2-2b-it

---

**End of CLAUDE.md.** Update this file as the project evolves. The §6 weekly checkpoints are the only deadlines that matter.