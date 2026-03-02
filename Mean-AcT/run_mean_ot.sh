export DATA_DIR=""
export CACHE_DIR=""
export HF_HUB_CACHE=""  # optionally, models will be saved here
export HF_TOKEN="`"  # required for some specific models like Gemma-2

python -m act.scripts.download_external_data
# if [ -z "$1" ]; then
#   echo "Usage: $0 <seed> [gpu_id]"
#   exit 1
# fi

# seed=$1
# GPU_ID=${2:-0}


# # seed=38

# python -m act.scripts.pipeline \
# "task_params=toxicity" \
# "seed=$seed" \
# "results_dir=results_$seed" \
# "responses.batch_size=20" \
# "model=gemma-2-2b.yaml" \
# "responses.max_batches=1" \
# "wandb.mode=disabled" \
# "text_generation.num_sentences=10" \
# "text_generation.new_seq_len=48" \
# "text_generation.strength_sample_size=5" \
# "intervention_params.incremental=incr" \
# "device=cuda:$GPU_ID" \
# "model.dtype=float32"