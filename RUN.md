# Gavel — run guide

Pipeline: **GRPO** (a strong LLM judges rollouts and its rubric trace + score is
logged for free) → **SFT** (distill those traces into a small cheap grader) →
**audit** (does the cheap grader reproduce the judge *and* track real truth?).

The judge is always an OpenAI-compatible endpoint, so the distilled grader is a
drop-in: later you point the same client at it instead of the teacher.

---

## Phase 0 — GRPO with the rubric judge (canonical, verl)

Train Qwen3-1.7B + LoRA on `poly_easy` with verl GRPO. The reward is a strong LLM
judging each rollout against a 0–9 rubric (`grader.compute_score`), and every
grading call's trace + score is appended to `gavel/grpo/traces.jsonl` — the free,
labeled distillation set.

### Layout
- `gavel/grpo/data.py` — build the verl parquet datasets (render prompt, carry `ground_truth`).
- `gavel/grpo/grader.py` — the rubric grader (0–9). `compute_score` is verl's `custom_reward_function`: builds the judge messages, parses `SCORE: <int>`, and logs `{data_source, problem, ground_truth, solution, judge_trace, score}` to `TRACE_LOG`.
- `gavel/grpo/train.sh` — verl `main_ppo` GRPO: Qwen3-1.7B + LoRA, reward `grader.compute_score`.
- `gavel/grpo/launch.sh` — starts the judge vLLM server **and** `train.sh` together (judge on `JUDGE_GPU`, training on the GPUs set in `train.sh`).

### Run
```bash
# Judge vLLM server (Qwen2.5-7B as "judge") + verl GRPO, one command.
# Judge on GPU $JUDGE_GPU (default 0); training on the GPUs in train.sh (4,5,6,7).
TRACE_LOG="$PWD/gavel/grpo/traces.jsonl" \
JUDGE_GPU=0 \
bash gavel/grpo/launch.sh
```

Teacher options (set in `gavel/grpo/grader.py` via env):
- **local** (default) — the vLLM judge from `launch.sh` via `OPENAI_BASE_URL`.
- **OpenRouter / API** — `JUDGE_BACKEND=openrouter OPENROUTER_API_KEY=...`; nothing else changes.

### Phase 0 (alternative) — TRL GRPO
`scripts/run_trl_grpo.sh` runs the same idea on **TRL** instead of verl: Qwen3-4B
policy via `accelerate`, the **binary correctness** judge in `gavel/reward.py`
(`JudgeReward`), a DeepSeek teacher, writing `runs/trl-grpo/traces.jsonl`.

- Layout: `gavel/data.py` (DAPO-Math loader), `gavel/reward.py` (`JudgeReward`, logs `{question, completion, ground_truth, judge_trace, judge_score}`), `gavel/trl_grpo/` (rollout + trainer), `gavel/train_grpo.py` (single-GPU TRL entry), `scripts/serve_judge.sh` (vLLM judge).
- Knobs (env vars): `POLICY_MODEL`, `N_EXAMPLES`, `MAX_STEPS`, `BATCH_SIZE`, `GRAD_ACCUM`, `NUM_GENERATIONS`, `MAX_PROMPT_LEN`, `MAX_COMPLETION_LEN`, `LR`, `KL_BETA`, `LORA_R`, `USE_VLLM`, `USE_WANDB`.
- **Note:** this path emits the *binary* judge format, not the rubric. Phase 2 below assumes the rubric traces (`gavel/grpo/traces.jsonl`); point `TRACE_LOG` at whichever trace log matches the grader you distilled.

---

## Phase 2 — distill the rubric grader + the headline number

The canonical GRPO run logs every rubric grading call to `gavel/grpo/traces.jsonl`
(score 0–9). That free, labeled set is all Phase 2 needs — no judge server required.

```bash
# 1. Distill a small LoRA grader from the logged rubric traces (Qwen2.5-3B base).
#    Trains on the first 80% of traces; holds out the last 20% for audit.
CUDA_VISIBLE_DEVICES=0 \
TRACE_LOG=gavel/grpo/traces.jsonl \
SFT_OUT=runs/grader-sft \
python -m gavel.sft.train          # or: bash gavel/sft/train.sh

# 2. Audit — THE HEADLINE NUMBER. Re-grade the held-out 20% with the distilled
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

---

## Environment notes
- verl path runs in the **`verl`** env (working vLLM); the TRL path and Phase 2
  run anywhere with `torch` + `trl` + `peft` (the box's `modelmerge` env, whose
  vLLM was ABI-broken vs torch 2.8, so TRL falls back to HF generation).
- The judge is just an HTTP endpoint — any env with working vLLM can serve it,
  or use an API teacher and skip the judge GPU entirely.
