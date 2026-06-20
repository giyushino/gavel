for the env, we're going to be using the conda env modelmerge for now

## Venv patches

Two files in `.venv` were patched to fix version incompatibilities. If the venv is rebuilt, reapply these (or just `uv pip install --upgrade prometheus-fastapi-instrumentator` for the second one).

1. **`verl/workers/actor/dp_actor.py` line 119** — TensorDict no longer supports `in` checks directly. Changed:
   ```python
   # before
   if "multi_modal_inputs" in micro_batch:
   # after
   if "multi_modal_inputs" in micro_batch.keys():
   ```

2. **`prometheus_fastapi_instrumentator/routing.py` line 55** — `_IncludedRouter` has no `path` attribute, crashing the vLLM health endpoint and all API requests. Changed:
   ```python
   # before
   if match == Match.FULL:
       route_name = route.path
   # after
   if match == Match.FULL:
       if not hasattr(route, "path"):
           continue
       route_name = route.path
   ```
   Alternatively: `uv pip install --upgrade prometheus-fastapi-instrumentator`

# DESIGN DOC
# Understudy — Design Doc

> Turn the throwaway cost of RL grading into a marketplace of cheap, verified autograders — judging the judges the same way they judge the models.

**Status:** hackathon build · **Deadline:** 5PM tomorrow · **Trainer:** TRL · **Compute:** 8×H100

---

## 1. The insight (read this first)

When you RL-train a model with an LLM-as-judge, you are already paying to grade thousands of rollouts. That grading is a sunk cost you cannot avoid. If you **log the judge's reasoning trace + score on every grading call**, that unavoidable expense becomes a free, labeled training set. SFT a small model on it and you get a grader that does the same job at a fraction of the cost and latency — and you can **reuse it** the next time anyone trains on that task.

Everything else in this doc is plumbing around that one idea. Protect it: it is the Novelty criterion, and it is the thing a judge can repeat to another judge in one sentence.

## 2. What we're building

A pipeline where a user specifies **a model + a dataset**, and we:

1. Convert the dataset into TRL-ready format (agent + cached script).
2. Generate a grading rubric, shown to the user to confirm.
3. Run GRPO with a strong LLM judging rollouts against that rubric.
4. Log every judge trace + score inside the reward function.
5. After training, SFT (LoRA) a small grader on those traces.
6. Cache two artifacts: the **conversion script** and the **distilled grader**.

Next time someone trains on the same dataset: known-good conversion replays instantly, and the cheap cached grader replaces the expensive judge.

### The marketplace framing (the trajectory story)

Distilled weights aren't the scarce asset — anyone can re-distill from traces, copies are free. What's scarce is **the human judgment of what "good" means**: the rubric plus a validated audit set proving the grader correlates with real ground truth. So this is a marketplace of *verified skill definitions*, each grader carrying a measured quality score with attribution to its author. We judge the judges the same way the judges judge the models — the track's "judge capability of a person or model," applied recursively.

## 3. Scope — build vs. stub

The single most important section for hitting the deadline. The rubric says *"we care about what works, not just what's described."* So make what *works* be the novel core, and *describe* the flaky orchestration.

**BUILD FOR REAL (the demo runs on these):**
- One GRPO run via TRL with an LLM-judge reward function.
- Trace + score logging inside that reward function.
- SFT (LoRA) of a small grader on the logged traces.
- The audit metric: distilled grader's correlation with ground truth on a held-out set. **This is the guaranteed deliverable.**

**STUB / DESCRIBE (show one cached success, never gamble live):**
- Agent dataset→parquet conversion (run it offline, cache the result, show the cached artifact).
- Cache replay / "one-click the second time" (show the cached script + grader exist; narrate the replay).
- Multi-grader / pooled-trace distillation (build single-run; pitch pooled as trajectory).
- Any node-provisioning / dependency-install automation (pre-baked, not live).

**Rule:** never let an unproven component gate the submission. The audit number must be on screen and recorded tonight.

## 4. Architecture

```
                    ┌─────────────────────────────────────────────┐
   user input  ──▶  │  model id + dataset id (+ optional rubric)   │
                    └───────────────────────┬─────────────────────┘
                                            │
                      ┌─────────────────────▼─────────────────────┐
                      │  Conversion agent (skill)                  │
                      │  dataset → TRL format · CACHE the script   │
                      └─────────────────────┬─────────────────────┘
                                            │
                      ┌─────────────────────▼─────────────────────┐
                      │  Rubric agent → user confirms              │
                      └─────────────────────┬─────────────────────┘
                                            │
       ┌────────────────────────────────────▼────────────────────────────────┐
       │  TRL GRPOTrainer                                                     │
       │    policy (Qwen) ── rollouts ──▶ reward_fn  ◀── vLLM judge server    │
       │                                     │                                │
       │                                     └──▶ LOG {prompt, completion,    │
       │                                          judge_trace, score} → JSONL │
       └────────────────────────────────────┬────────────────────────────────┘
                                            │  (after training)
                      ┌─────────────────────▼─────────────────────┐
                      │  TRL SFTTrainer (LoRA)                     │
                      │  traces JSONL → small grader · CACHE it    │
                      └─────────────────────┬─────────────────────┘
                                            │
                      ┌─────────────────────▼─────────────────────┐
                      │  Audit: grader vs ground truth on held-out │
                      │  → correlation number  (HEADLINE RESULT)   │
                      └────────────────────────────────────────────┘
```

### GPU layout (8×H100)
- **Policy training (GRPO):** 4–5 GPUs.
- **Rollout generation (vLLM, policy):** 2 GPUs.
- **Judge server (vLLM, distilled grader during/after distillation):** 1 GPU.
- **Teacher judge:** strong **API model** (Claude/GPT), not local — offloads it from the H100s and you're logging its traces regardless.

Exact split depends on model size; tune at build time. Don't over-optimize — get it running first.

## 5. Components

### 5.1 Policy model
- **Qwen2.5-7B-Instruct** (or Qwen3-8B). LoRA or full FT both fit comfortably on this hardware.
- Served for rollouts via vLLM. TRL's GRPO integrates with vLLM for generation.

### 5.2 The dataset & task choice — pick one with a VERIFIABLE CORE
This is the most important single decision. Choose a task where correctness is mechanically checkable, so you can prove the grader tracks *real* ground truth — not just that it agrees with the big judge (which is circular and unconvincing).

Recommended: **text-to-SQL** (execute the query, compare result set) or **code-with-unit-tests** (run the tests).
- Mechanical signal = ground truth (correctness).
- Rubric/LLM-judge signal = style, efficiency, reasoning quality.
- Audit = does the distilled grader's score correlate with the mechanical correctness?

Have ONE vetted dataset fully wired and cached before recording.

### 5.3 Conversion agent (skill)
- An agent given a "convert dataset to TRL format" skill.
- Output: a Python script that maps the dataset's columns → prompt/completion fields TRL expects.
- **Cache the script** keyed by dataset id. First run does the work; later runs replay.
- For the demo: run offline on the vetted dataset, confirm it trains, cache it. Show the cached artifact; do not run the agent live.

### 5.4 Rubric agent
- LLM generates a grading rubric for the task from the dataset + a few sample rows.
- Shown to the user for confirmation/edit (human-in-the-loop — this is also your "verified skill definition" provenance).
- The confirmed rubric is the prompt context for the judge.

### 5.5 Reward function (the critical seam)
TRL's `GRPOTrainer` takes a reward function: `(prompts, completions, **kwargs) -> list[float]`. This is where everything happens.

```python
def judge_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # 1. mechanical ground truth (verifiable core), if available
        gt = mechanical_score(prompt, completion)          # e.g. run SQL / tests

        # 2. LLM judge against the confirmed rubric
        judge_out = call_judge(rubric, prompt, completion) # returns {trace, score}

        # 3. LOG — this is the distillation dataset, accumulated for free
        log_trace({
            "prompt": prompt,
            "completion": completion,
            "judge_trace": judge_out["trace"],
            "judge_score": judge_out["score"],
            "ground_truth": gt,
        })

        rewards.append(judge_out["score"])  # continuous rubric score is fine
    return rewards
```

Notes:
- **Log inside the reward fn** — it's the one place every grading call passes through.
- Continuous rubric scores work in GRPO unchanged; group standardization normalizes any scale.
- With a noisy LLM judge, consider mean-only (leave-one-out) baseline instead of dividing by std (Dr.GRPO) — one-line change if training is unstable.
- Keep the KL term (β) and clipping on — soft judge rewards are more hackable than verifiable ones; the reference anchor is your main defense against reward hacking.
- Batch judge calls / cache identical completions to cut latency — the judge is the throughput bottleneck.

### 5.6 Teacher judge
- Strong API model (Claude/GPT). Prompt = confirmed rubric + prompt + completion → returns a **reasoning trace + numeric score**. Ask for the trace explicitly (it's the SFT target).
- Spot-check a fraction on the mechanical ground truth to catch judge drift/hacking — nice robustness slide.

### 5.7 Grader distillation (SFT)
- After (or during) training, take the logged JSONL: input = (rubric, prompt, completion), target = (teacher trace + score).
- `SFTTrainer` (LoRA) on a small base (e.g. Qwen2.5-3B).
- Output: the cached grader. **Cache it keyed by dataset id.**

### 5.8 Audit / quality metric — THE HEADLINE
- Held-out set of (prompt, completion) with known mechanical ground truth.
- Score each with the distilled grader.
- Compute **correlation (Spearman/Pearson) between grader score and ground truth**, and agreement with the teacher judge.
- This number is the proof the idea works, it's quantitative, and it does **not depend on RL convergence**. Generate and record it first.

### 5.9 Cache
- Simple keyed store (even a dict + filesystem is fine for the hackathon).
- Key: dataset id. Values: conversion script path, distilled grader (LoRA) path, rubric, audit score.
- "Train on this dataset again" → load both, skip the expensive path.

## 6. Data flow summary
1. Dataset → conversion agent → TRL-format parquet (cached script).
2. Sample rows → rubric agent → user confirms.
3. GRPO loop: policy generates → reward_fn (mechanical + judge) → log traces → update.
4. Traces JSONL → SFTTrainer (LoRA) → distilled grader (cached).
5. Held-out set → grader → correlation vs ground truth → headline number.

## 7. Build order (deadline-driven)

Front-load the guaranteed result; everything risky is pre-baked.

**Phase 0 — prove the boring core (do FIRST):**
- TRL `GRPOTrainer` training Qwen on the vetted dataset, end to end, with a trivial reward. If this doesn't run, nothing else matters. Fix this before anything fancy.

**Phase 1 — the reward seam:**
- Wire the LLM-judge reward fn + mechanical ground truth. Confirm reward varies sensibly. Turn on trace logging.

**Phase 2 — distillation + the headline number:**
- SFT the small grader on logged traces. Compute correlation-with-ground-truth on held-out. **Get this number on screen.**

**Phase 3 — RECORD the guaranteed demo.** As soon as the number is visible. Now you have a submittable video no matter what.

**Phase 4 — caching + agent conversion (polish):**
- Cache conversion script + grader. Wire the "second run is one-click" replay. Pre-run the agent conversion offline and cache it.

**Phase 5 — UI + re-record.** Thin wizard (model + dataset → rubric confirm → train → grader). Re-record if the RL curve also looks good / a second grader finishes.

**If you're behind:** ship after Phase 3. The audit number + the insight + the trajectory pitch is a complete, placing submission on its own.

## 8. Demo / video plan (5 min)
- **0:00–0:45** — The insight, stated plainly. The sunk-cost → free-grader line.
- **0:45–1:30** — Problem/fit: RL & eval are bottlenecked on expensive, slow, hackable judges.
- **1:30–3:30** — Live: graded GRPO run logging traces → distill grader → **correlation number on screen.** The "it works" proof. Show the number, not "trust me."
- **3:30–4:15** — Marketplace recursion (judge the judges) + cached artifacts (grader + conversion script) → "second run is cheap and one-click." Mention other graders training on the same pipeline = trajectory.
- **4:15–5:00** — Impact at scale: a shared library of verified cheap graders that makes RL/eval affordable for everyone; graders compound as traces pool.

Be honest about proven vs. in-progress: "verified result on task A; B and C training now on the identical pipeline" is stronger (trajectory) and truthful.

## 9. Risks & mitigations
| Risk | Mitigation |
|---|---|
| RL doesn't converge by 5PM | Headline = audit number, which is convergence-independent. Record Phase 3 early. |
| Agent dataset conversion breaks live | Pre-run offline, cache, show cached artifact. Never live. |
| verl/TRL data-format jank | TRL eats HF formats more directly; vet the one demo dataset fully. |
| Judge is slow → low throughput | Distilled local grader on vLLM; batch/cache judge calls. |
| Reward hacking (soft rewards) | Keep KL + clipping on; spot-check judge vs mechanical ground truth. |
| Circular "grader agrees with judge" claim | Verifiable-core task → correlate grader with REAL ground truth. |
| Single-run grader is brittle | Build single-run; pitch pooled-trace robustness as trajectory. |

## 10. Stack
- **Trainer:** TRL (`GRPOTrainer` + `SFTTrainer`, one library, one format).
- **Policy:** Qwen2.5-7B-Instruct. **Grader base:** Qwen2.5-3B. LoRA via PEFT.
- **Serving:** vLLM (rollouts + grader).
- **Teacher judge:** strong API model (Claude/GPT).
- **Glue:** Python; simple filesystem cache; thin UI (whatever's fastest — Streamlit/Gradio).


---

**The one thing that wins or loses this:** the distilled grader's correlation-with-ground-truth number, on screen, recorded early. Build toward that first; everything else is polish on top of an already-submittable result.

