
# Mean-AcT: Toxicity Mitigation & Diffusion Style Control

This folder contains the code for the **toxicity mitigation** and **image generation style control** experiments from our paper:

> **Activation Steering with a Feedback Controller** (ICLR 2026)
> Dung V. Nguyen, Nhi Y. Pham, Hieu M. Vu, Lei Zhang, Tan M. Nguyen

Our implementation builds on [Mean-AcT](https://github.com/apple/ml-act) (Rodriguez et al., ICLR 2025). PID Steering is integrated as the `mean_ot_pid` intervention, a drop-in replacement for `mean_ot` (Mean-AcT) and `linear_ot` (Linear-AcT).

---

## Setup

1. Clone the repository and navigate to this folder:

   ```bash
   git clone https://github.com/dungnvnus/pid-steering.git
   cd pid-steering/diffusion_steering
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:

   ```bash
   export DATA_DIR="path/to/data"
   export CACHE_DIR="path/to/cache"
   export HF_HUB_CACHE="path/to/hf/cache"   # optional
   export HF_TOKEN="your_token"              # required for Gemma-2
   ```

4. Download external datasets (RealToxicityPrompts, Jigsaw, COCO Captions):

   ```bash
   python -m act.scripts.download_external_data
   ```

---

## Toxicity Mitigation

We provide `run_mean_ot.sh` to run the toxicity mitigation pipeline. Fill in your paths and run:

```bash
bash run_mean_ot.sh <seed> [gpu_id]
```

Or run manually with `mean_ot_pid` (PID-AcT) as the intervention:

```bash
python -m act.scripts.pipeline \
    "task_params=toxicity" \
    "seed=38" \
    "results_dir=results_38" \
    "responses.batch_size=20" \
    "model=gemma-2-2b.yaml" \
    "intervention_params=mean_ot_pid" \
    "intervention_params.incremental=incr" \
    "wandb.mode=disabled" \
    "device=cuda:0"
```

To reproduce Mean-AcT (baseline), replace `mean_ot_pid` with `mean_ot`. To reproduce Linear-AcT, use `linear_ot`.

**Evaluated models:** Gemma2-2B, Llama3-8B

**Metrics:** CLS toxicity (%), zero-shot toxicity (%), perplexity (Wikipedia), perplexity (Mistral-7B), MMLU

---

## Diffusion Style Control

We provide `run_t2i.sh` to run the diffusion style control pipeline. Fill in your paths and run:

```bash
bash run_t2i.sh <seed> [gpu_id]
```

This will run the pipeline for the `cyberpunk` style on `FLUX.1-schnell` by default. To run other styles, edit the `for sty in cyberpunk` line in the script to include `steampunk` or other styles from `act/configs/task_params/coco_styles.yaml`.

Or run manually:

```bash
python -m act.scripts.pipeline \
    --config-name text_to_image_generation.yaml \
    "task_params=coco_styles" \
    "model=FLUX.1-schnell.yaml" \
    "seed=38" \
    "intervention_params=mean_ot_pid" \
    "task_params.src_subsets=['none']" \
    "task_params.dst_subsets=[cyberpunk]" \
    "results_dir=results_flux_pid_cyberpunk_38" \
    "task_params.prompt_subset=['none']" \
    "responses.batch_size=8" \
    "responses.max_batches=64" \
    "interventions.max_batches=null" \
    "intervention_params.incremental=incr" \
    "wandb.mode=disabled" \
    "evaluation=['text-to-image-generation','clip_score']" \
    "text_to_image_generation.batch_size=4" \
    "text_to_image_generation.max_batches=15" \
    "text_to_image_generation.create_gif=true" \
    "device=cuda:0"
```

**Evaluated models:** FLUX.1-Schnell, FLUX.1-Dev, SDXL-Lightning (configs in `act/configs/model/`)

**Metrics:** CLIP zero-shot style score, CLIPScore (content preservation)

Results are saved under `results_dir/generate_with_hooks_diffusion/`, organized by intervention strength.

---

## Available Interventions

| Config name     | Method          |
|-----------------|-----------------|
| `mean_ot_pid`   | PID-AcT (ours)  |
| `mean_ot`       | Mean-AcT        |
| `linear_ot`     | Linear-AcT      |
| `aura`          | AURA            |

Configs are in `act/configs/intervention_params/`.

---

## Visualizing Results

Use the provided notebooks to analyze and visualize results:

- `read_results.ipynb` — read and aggregate experiment outputs
- `visualize.ipynb` / `visualize.py` — generate plots

---

## Citation

If you use this code, please cite both our work and the original Mean-AcT paper:

```bibtex
@inproceedings{nguyen2026pidsteering,
  title     = {Activation Steering with a Feedback Controller},
  author    = {Dung V. Nguyen and Nhi Y. Pham and Hieu M. Vu and Lei Zhang and Tan M. Nguyen},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}

@inproceedings{rodriguez2025act,
  title     = {Controlling Language and Diffusion Models by Transporting Activations},
  author    = {Rodriguez, Pau and Blaas, Arno and Klein, Michal and Zappella, Luca and Apostoloff, Nicholas and Cuturi, Marco and Suau, Xavier},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```