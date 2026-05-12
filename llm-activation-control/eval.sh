# Usage: GPU=0 bash eval.sh
# Defaults to GPU=0 if not set.
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=$GPU \
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-Guard-3-8B \
  --port 8898 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --tensor-parallel-size 1

# Uncomment to also serve QVQ-72B for LLM-as-judge (requires 2 GPUs, e.g. GPU_JUDGE="0,1"):
# GPU_JUDGE=${GPU_JUDGE:-0,1}
# CUDA_VISIBLE_DEVICES=$GPU_JUDGE \
# python3 -m vllm.entrypoints.openai.api_server \
#   --model Qwen/QVQ-72B-Preview \
#   --trust-remote-code \
#   --port 8809 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.8 \
#   --max-model-len 4096 \
#   --tensor-parallel-size 2