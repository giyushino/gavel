#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRACE_LOG="${TRACE_LOG:-runs/trl-grpo/traces.jsonl}"
SFT_BASE="${SFT_BASE:-Qwen/Qwen2.5-3B-Instruct}"
SFT_OUT="${SFT_OUT:-runs/grader-sft}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
AUDIT_FRAC="${AUDIT_FRAC:-0.2}"

echo "=== SFT grader audit ==="
echo "  traces  : $TRACE_LOG"
echo "  grader  : $SFT_OUT"
echo "  base    : $SFT_BASE"
echo "  gpu     : $CUDA_VISIBLE_DEVICES"

TRACE_LOG="$TRACE_LOG" \
SFT_BASE="$SFT_BASE" \
SFT_OUT="$SFT_OUT" \
AUDIT_FRAC="$AUDIT_FRAC" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
uv run python -m gavel.audit "$@"
