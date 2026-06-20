#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

JUDGE_GPU=${JUDGE_GPU:-0}
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
JUDGE_PORT=${JUDGE_PORT:-8001}

# Start judge server in background
echo "Starting judge server on GPU $JUDGE_GPU, port $JUDGE_PORT..."
CUDA_VISIBLE_DEVICES="$JUDGE_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL_PATH" \
    --served-model-name judge \
    --port "$JUDGE_PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 &
JUDGE_PID=$!

# Start training in background — RL init takes longer than vLLM spin-up,
# so the judge will be ready well before the first reward call
echo "Starting training..."
OPENAI_BASE_URL="http://localhost:$JUDGE_PORT/v1" \
OPENAI_API_KEY=EMPTY \
JUDGE_MODEL=judge \
bash "$SCRIPT_DIR/train.sh" "$@" &
TRAIN_PID=$!

# Wait for training to finish, then clean up judge
wait "$TRAIN_PID"
TRAIN_EXIT=$?
kill "$JUDGE_PID" 2>/dev/null
exit "$TRAIN_EXIT"
