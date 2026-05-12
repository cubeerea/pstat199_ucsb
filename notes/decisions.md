# Open Questions & Decisions

Track judgment calls here as they are made.

---

## Q1 — Error signal for Global PID (OPEN)

**Question:** Is the error signal `e(k)` just `r_bar` (constant), or the projection of current residual stream onto `r_bar` (state-dependent)?

**Default:** Constant error (`e(k) = r_bar`). Matches Hank's framing of testing integral windup. Implement this first; add state-dependent variant as ablation if time allows.

**Status:** Default selected, not yet validated with Hank.

---

## Q2 — Injection point: pre- or post-layernorm? (OPEN)

**Question:** `angular_steering.py` hooks `model.model.layers[k].post_attention_layernorm` via `register_forward_pre_hook`, which captures the INPUT of the layernorm (= output of residual stream from attention block). That is a pre-layernorm injection. Does the paper intend pre- or post-layernorm injection?

**Evidence:** `angular_steering_causal.ipynb` Cell 86 comment says:
- `resid-mid` = input of `post_attention_layernorm` (Gemma: same)
- `resid-post` = input of next `input_layernorm`

So they treat the layernorm INPUT as the canonical residual stream. Injection should be at the layernorm INPUT (pre-layernorm), not output.

**Default:** Pre-layernorm injection (hook layernorm inputs via `register_forward_pre_hook`).

**Status:** Assumption; needs validation against paper §3 or §A.

---

## Q3 — Token position for DIM (OPEN)

**Question:** Which token positions are used for DIM extraction and steering?

**Evidence from `llama_many_layers.py`:**
- Extraction: LAST tokens of the "template suffix" (model-specific separator tokens at end of chat template, e.g., `<start_of_turn>model\n` for Gemma-2). Not last generation token.
- Steering injection: applied at ALL token positions during generation (hooks fire on every forward pass regardless of sequence position).

**Default:** Extract DIM at last token of template suffix; inject steering at all positions.

**Status:** Assumption; fine for initial implementation.

---

## Q4 — AdvBench split (OPEN)

**Default:** All 520 harmful behaviors for final numbers. 50 for iteration.

---

## Q5 — ASR scoring method (OPEN)

**Default:** String-match primary (JailbreakBench 12-phrase list). LLM-judge spot check on 50 samples to validate.

---

## Q6 — TransformerLens vs. HuggingFace (DECIDED)

**Situation:** The paper's notebooks use TransformerLens (`HookedTransformer`). CLAUDE.md says to stay on HuggingFace stack.

**Decision:** Use HuggingFace. `angular_steering.py` (the production module) already uses plain HF `register_forward_hook` / `register_forward_pre_hook` via the `add_hooks` context manager — no TransformerLens required. We adapt from this file, not the notebooks.

**Why:** Gemma-2-2B-it loads fine with HF AutoModelForCausalLM. No need for TL's overhead. Keeps us off the TransformerLens Gemma-2 support dependency.

---

## Q7 — PID gain values for Gemma-2-2B-it (ESCALATED)

**See `notes/gains_gemma2.md`.** No Gemma-2-2B-it specific gains in the repo. Recommendation is to use `Kp=0.9, Ki=0.01, Kd=0.01` from the primary pipeline. Awaiting Hank confirmation or paper appendix lookup.
