CUDA_VISIBLE_DEVICES=6 \
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-Guard-3-8B \
  --port 8898 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --tensor-parallel-size 1

# CUDA_VISIBLE_DEVICES=0,1 \
# python3 -m vllm.entrypoints.openai.api_server \
#   --model Qwen/QVQ-72B-Preview \
#   --trust-remote-code \
#   --port 8809 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.8 \
#   --max-model-len 4096 \
#   --tensor-parallel-size 2