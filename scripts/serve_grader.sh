#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
LORA_PATH=runs/grader-sft
SERVED_NAME=grader
GPU=1
PORT=8001

CUDA_VISIBLE_DEVICES="$GPU" \
uv run vllm serve "$BASE_MODEL" \
  --enable-lora \
  --lora-modules "${SERVED_NAME}=${LORA_PATH}" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
