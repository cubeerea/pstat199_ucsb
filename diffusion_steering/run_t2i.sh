#!/usr/bin/env bash

# Set paths (fill these in as needed)
export DATA_DIR=""
export CACHE_DIR=""
export HF_HUB_CACHE=""   # optionally, models will be saved here
export HF_TOKEN=""  # required for Gemma-2 or similar

# --- Argument parsing ---
if [ -z "$1" ]; then
  echo "Usage: $0 <seed> [gpu_id] [style]"
  exit 1
fi

seed=$1
GPU_ID=${2:-0}       # default: GPU 0
# style=${3:-sketch}   # default: sketch

model=FLUX.1-schnell.yaml
# model=FLUX.1-dev.yaml
# model=SDXL-Lightning.yaml

model_name="${model%.yaml}"   # strip the ".yaml" suffix
intervention_params=linear_ot

# --- Run pipeline for multiple styles ---
for sty in cyberpunk; do
  python -m act.scripts.pipeline \
      --config-name text_to_image_generation.yaml \
      "task_params=coco_styles" \
      "model=${model}" \
      "seed=$seed" \
      "intervention_params=${intervention_params}" \
      "task_params.src_subsets=['none']" \
      "task_params.dst_subsets=[$sty]" \
      "results_dir=results_${model_name}_${intervention_params}_${sty}_${seed}" \
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
      "device=cuda:$GPU_ID"
done
