# Phase 0 — GRPO + LLM-judge reward

Train Qwen3-4B on DAPO-Math-17k with GRPO, where the reward comes from a second
LLM called over the **OpenAI API**. The judge is just an OpenAI-compatible
endpoint, so later we point the same client at our own distilled grader.

## Layout
- `gavel/data.py` — load DAPO-Math-17k, render chat prompt (Qwen3, thinking off), carry `ground_truth`.
- `gavel/reward.py` — `JudgeReward`: calls the judge via the `openai` client, parses `SCORE: <x>`, and logs `{question, completion, judge_trace, score, ground_truth}` to a JSONL (the free distillation set).
- `gavel/train_grpo.py` — TRL `GRPOTrainer`, Qwen3-4B + LoRA, HF generation.
- `scripts/serve_judge.sh` — serve the teacher judge with vLLM (OpenAI-compatible).

## Environment notes (this box)
- Train in conda env **`modelmerge`**. Its vLLM was ABI-broken against torch 2.8 and has been **uninstalled** there; TRL falls back to HF generation (`use_vllm=False`).
- The judge vLLM server runs from the **`verl`** env (working vLLM 0.11) — any env works, it's just an HTTP endpoint.
- Hardware here is **8×A40 (46 GB)**, not the 8×H100 in the design doc. LoRA keeps the 4B policy on one card.

## Run
```bash
# 1. Judge server on GPU 0 (any env with working vLLM)
CUDA_VISIBLE_DEVICES=0 conda run -n verl \
  vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name judge \
  --port 8000 --gpu-memory-utilization 0.85 --max-model-len 4096

# 2. Training on GPU 1
CUDA_VISIBLE_DEVICES=1 \
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY JUDGE_MODEL=judge \
TRACE_LOG=runs/phase0-qwen3-4b/traces.jsonl \
conda run --no-capture-output -n modelmerge python -m gavel.train_grpo
```

To use a real teacher instead of the local judge, point `OPENAI_BASE_URL` /
`OPENAI_API_KEY` / `JUDGE_MODEL` at OpenAI or OpenRouter — nothing else changes.

## Phase 2 — distill the rubric grader + the headline number

The canonical GRPO run (`gavel/grpo/`, verl + `grader.compute_score`) logs every
rubric grading call to `gavel/grpo/traces.jsonl` (score 0–9). That free, labeled
set is all Phase 2 needs — no judge server required.

```bash
# 3. Distill a small LoRA grader from the logged rubric traces (Qwen2.5-3B base).
#    Trains on the first 80% of traces; holds out the last 20% for audit.
CUDA_VISIBLE_DEVICES=0 \
TRACE_LOG=gavel/grpo/traces.jsonl \
SFT_OUT=runs/grader-sft \
python -m gavel.sft.train          # or: bash gavel/sft/train.sh

# 4. Audit — THE HEADLINE NUMBER. Re-grade the held-out 20% with the distilled
#    grader. Reports (a) FIDELITY to the teacher judge it replaces and
#    (b) GROUNDING vs INDEPENDENT mechanical correctness (gavel/verify.py runs
#    the symbolic answer-match, no LLM) with the teacher's grounding as ceiling.
CUDA_VISIBLE_DEVICES=0 \
TRACE_LOG=gavel/grpo/traces.jsonl \
SFT_OUT=runs/grader-sft \
python -m gavel.audit
```

`audit.py` prints the fidelity/grounding table and writes the full report to
`runs/grader-sft/audit.json`. This number does **not** depend on RL convergence —
it's the guaranteed deliverable.

### Phase 2 knobs (env vars)
- distill (`gavel.sft.train`): `SFT_BASE`, `SFT_OUT`, `EPOCHS`, `LR`, `MAX_LEN`,
  `LORA_R`, `LORA_ALPHA`, `BATCH_SIZE`, `GRAD_ACCUM`, `AUDIT_FRAC`, `USE_WANDB`.
- audit (`gavel.audit`): `SFT_BASE`, `SFT_OUT`, `AUDIT_FRAC`, `AUDIT_BATCH`,
  `AUDIT_REPORT`.

`AUDIT_FRAC` (default 0.2) must match between the two — it's the deterministic,
position-based train/held-out split, so the grader is never audited on traces it
was distilled from.

## Knobs (env vars)
`POLICY_MODEL`, `N_EXAMPLES`, `MAX_STEPS`, `BATCH_SIZE`, `GRAD_ACCUM`,
`NUM_GENERATIONS`, `MAX_PROMPT_LEN`, `MAX_COMPLETION_LEN`, `LR`, `KL_BETA`,
`LORA_R`, `USE_VLLM`, `USE_WANDB`.
