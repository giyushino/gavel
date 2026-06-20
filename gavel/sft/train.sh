#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

TRACE_LOG="${TRACE_LOG:-$SCRIPT_DIR/../grpo/traces.jsonl}"
SFT_OUT="${SFT_OUT:-$PROJECT_ROOT/runs/grader-sft}"
SFT_BASE="${SFT_BASE:-Qwen/Qwen2.5-3B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
AUDIT_FRAC="${AUDIT_FRAC:-0.2}"

echo "=== SFT grader distillation ==="
echo "  traces : $TRACE_LOG"
echo "  output : $SFT_OUT"
echo "  base   : $SFT_BASE"
echo "  gpu    : $CUDA_VISIBLE_DEVICES"

TRACE_LOG="$TRACE_LOG" \
SFT_OUT="$SFT_OUT" \
SFT_BASE="$SFT_BASE" \
AUDIT_FRAC="$AUDIT_FRAC" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python -m gavel.sft.train "$@"
