"""
Reference-model perplexity for coherence sanity-check.

Uses a small reference LM (default: EleutherAI/pythia-70m) to score how
"natural" the steered model's completions are. High PPL = surprising under
the reference = likely incoherent / OOD output (e.g. activation collapse
from integral windup pushing residual stream off-manifold).

Caveat: PPL is low on repetitive text too (e.g. "I cannot I cannot ...").
This is a sanity check, not a coherence ground truth.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REF_MODEL_ID = "EleutherAI/pythia-70m"
DEGENERATE_PPL_THRESHOLD = 1000.0


def load_reference_model(model_id: str = REF_MODEL_ID, device: str = "cpu"):
    """Load reference LM + tokenizer for perplexity scoring."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tok


@torch.no_grad()
def compute_perplexity(
    completions: list[str],
    ref_model,
    ref_tokenizer,
    device: str,
    batch_size: int = 16,
    max_length: int = 512,
) -> dict:
    """
    Per-completion perplexity under a reference LM.

    Empty completions are replaced with "[empty]" (so they score as
    high-PPL outliers, flagging a coherence failure) — they don't crash.

    Returns:
        {
            "per_completion": list[float],
            "mean":   float,
            "median": float,
            "p95":    float,
            "p99":    float,
            "n_degenerate": int,    # PPL > DEGENERATE_PPL_THRESHOLD
            "n_empty":      int,
            "n_total":      int,
            "ref_model":    str,
            "threshold":    float,
        }
    """
    n_empty = sum(1 for c in completions if not c.strip())
    completions_clean = [c if c.strip() else "[empty]" for c in completions]

    perplexities: list[float] = []
    for i in range(0, len(completions_clean), batch_size):
        batch = completions_clean[i : i + batch_size]
        enc = ref_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        input_ids = enc.input_ids
        attn = enc.attention_mask

        if input_ids.shape[1] < 2:
            # Single-token sequences have no next-token prediction → undefined PPL
            perplexities.extend([float("inf")] * len(batch))
            continue

        logits = ref_model(input_ids=input_ids, attention_mask=attn).logits  # (B, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn[:, 1:].contiguous().float()

        ce = F.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels, reduction="none"
        )  # (B, T-1)
        ce = ce * shift_mask
        token_counts = shift_mask.sum(dim=-1).clamp(min=1)
        seq_loss = ce.sum(dim=-1) / token_counts
        # Cap PPL at a finite very-large value to keep summary stats well-defined
        seq_loss = seq_loss.clamp(max=20.0)  # exp(20) ≈ 4.85e8
        perplexities.extend(seq_loss.exp().cpu().tolist())

    arr = np.array(perplexities, dtype=np.float64)
    return {
        "per_completion": perplexities,
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "n_degenerate": int((arr > DEGENERATE_PPL_THRESHOLD).sum()),
        "n_empty":      n_empty,
        "n_total":      len(arr),
        "ref_model":    REF_MODEL_ID,
        "threshold":    DEGENERATE_PPL_THRESHOLD,
    }
