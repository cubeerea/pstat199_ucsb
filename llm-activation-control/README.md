# LLM Activation Control: Jailbreaking Experiments

This folder contains the code for the **jailbreaking** experiments from our paper:

> **Activation Steering with a Feedback Controller** (ICLR 2026)
> Dung V. Nguyen, Nhi Y. Pham, Hieu M. Vu, Lei Zhang, Tan M. Nguyen

Our implementation builds on [Angular Steering](https://openreview.net/forum?id=uAfzFV7mv2) (Vu & Nguyen, NeurIPS 2025). PID Steering is integrated as a drop-in replacement for the steering direction computation step, controlled via `METHOD_PREFIX` in the scripts.

---

## Installation

### 1. Main environment

```bash
conda create -n angular_steering python=3.10
conda activate angular_steering
cd llm-activation-control/
pip install -r requirements.txt
conda deactivate
```

### 2. vLLM fork (required)

Angular Steering requires a custom fork of vLLM with control vector support. Install it from source:

```bash
cd ../vllm/
conda activate angular_steering
VLLM_USE_PRECOMPILED=1 pip install --editable .
conda deactivate
cd ..
```

> Follow the official vLLM GPU installation guide at https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html

### 3. lm-evaluation-harness (for TinyBenchmarks)

```bash
conda create -n lm_eval python=3.10
conda activate lm_eval
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
conda deactivate
cd ..
```

---

## Reproducing the Results

### Step 1 — Extract steering directions

Run `angular_steering.ipynb` to extract refusal directions and construct the steering plane. This generates the `output/` folder with the steering configs.

To switch between methods, set `METHOD_PREFIX` in the notebook:

```python
METHOD_PREFIX = "PID_"   # PID Steering (ours)
METHOD_PREFIX = ""       # DIM (baseline)
METHOD_PREFIX = "RePE_"  # RePE
METHOD_PREFIX = "ITI_"   # ITI
```

Precomputed steering directions for all methods are provided in `output/` so you can skip this step and proceed directly to generation.

### Step 2 — Generate steered responses

Edit `generate_responses.py` to select the model and method:

```python
model_ids = ["google/gemma-2-9b-it"]  # or other models
METHOD_PREFIX = "PID_"  # match the direction extracted in Step 1
```

Then run:

```bash
conda activate angular_steering
python generate_responses.py
```

### Step 3 — Evaluate jailbreaking

Edit `evaluate_jailbreak.py` to select models and method, then run evaluations. Some evaluators require serving LLMs first:

**LlamaGuard 3** (port 8898):
```bash
bash eval.sh
```
This serves `meta-llama/Llama-Guard-3-8B` via vLLM on port 8898. Uncomment the second block in `eval.sh` to also serve `Qwen/QVQ-72B-Preview` on port 8809 for LLM-as-a-judge evaluation.

Then run the evaluator:
```bash
conda activate angular_steering
python evaluate_jailbreak.py
```

Supported evaluation methods (set `methods` list in the script):
- `substring_matching` — no server needed
- `LlamaGuard 3` — requires serving on port 8898
- `HarmBench` — served natively (requires sufficient GPU memory)
- `LLM-as-a-judge` (QVQ-72B) — requires serving on port 8809

### Step 4 — Evaluate perplexity

```bash
conda activate angular_steering
python eval_perplexity.py
```

### Step 5 — Evaluate TinyBenchmarks

First, serve the endpoint server (one model at a time, set port per model in `configs.py`):

```bash
conda activate angular_steering
python endpoint.py <model_id>
# e.g. python endpoint.py Qwen/Qwen2.5-3B-Instruct
```

Then run `eval_tinybench.sh` in the `lm_eval` environment. Edit `MODELS`, `METHOD_PREFIX`, and `MODEL_PORTS` in the script to match your setup:

```bash
conda activate lm_eval
bash eval_tinybench.sh
```

Results are saved under `benchmarks/<dir_id>/<task>/<METHOD_PREFIX>/<model_name>/`.

**Evaluated models:** Qwen2.5-3B/7B/14B-Instruct, Llama3.2-3B-Instruct, Llama3.1-8B-Instruct, Gemma2-9B-IT

**Metrics:** ASR (attack success rate), tinyArc, tinyGSM8k, tinyMMLU, tinyTruthQA, tinyHellaSwag, tinyWinoGrande

### Step 6 — Visualize results

Use `visualization.ipynb` to reproduce the plots from the paper.

---

## Interactive Demo (Optional)

To play with Angular Steering via a chat UI:

```bash
conda activate angular_steering
python endpoint.py <model_id>   # serve the endpoint
python steering_demo.py         # launch Gradio UI
```

---

## File Overview

| File | Description |
|------|-------------|
| `angular_steering.ipynb` | Extract steering directions and construct the steering plane |
| `generate_responses.py` | Generate steered responses using the vLLM fork |
| `evaluate_jailbreak.py` | Evaluate ASR via substring matching, LlamaGuard 3, HarmBench, LLM-as-a-judge |
| `eval_perplexity.py` | Evaluate perplexity of steered generations |
| `eval.sh` | Serve LlamaGuard 3 / QVQ-72B for evaluation |
| `eval_tinybench.sh` | Run TinyBenchmarks evaluation via lm-evaluation-harness |
| `endpoint.py` | OpenAI-compatible endpoint server with Angular Steering |
| `steering_demo.py` | Gradio chat UI for interactive steering |
| `configs.py` | Model-specific direction IDs and method prefixes |
| `visualization.ipynb` | Visualize benchmark results |

---

## Citation

If you use this code, please cite both our work and the Angular Steering paper:

```bibtex
@inproceedings{nguyen2026pidsteering,
  title     = {Activation Steering with a Feedback Controller},
  author    = {Dung V. Nguyen and Nhi Y. Pham and Hieu M. Vu and Lei Zhang and Tan M. Nguyen},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}

@inproceedings{vu2025angular,
  title     = {Angular Steering: Behavior Control via Rotation in Activation Space},
  author    = {Hieu M. Vu and Tan Minh Nguyen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=uAfzFV7mv2}
}
```