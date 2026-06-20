#!/usr/bin/env bash
# Serve the teacher judge behind an OpenAI-compatible endpoint via vLLM.
# The reward function talks to this through the `openai` client; later we point
# the same client at our distilled grader instead.
set -euo pipefail

GPU=${JUDGE_GPU:-0}
MODEL=${JUDGE_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
PORT=${JUDGE_PORT:-8000}

CUDA_VISIBLE_DEVICES="$GPU" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name judge \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
