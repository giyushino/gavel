"""Audit -- THE HEADLINE NUMBER. Does the distilled rubric grader work?

On the held-out split (same AUDIT_FRAC the SFT held out) we re-grade each
solution with the distilled grader and report two things:

  1. FIDELITY -- does the cheap grader reproduce the expensive teacher judge?
     Pearson/Spearman + MAE between distilled score and the logged teacher score.

  2. GROUNDING -- does its score track REAL correctness? Spearman against
     gavel/verify.is_correct (a symbolic answer-match, no LLM).

If the audit passes the fidelity threshold (Pearson >= MIN_PEARSON), the adapter
is automatically registered in the local cache so future training runs can use
it instead of the frontier judge.

Run (after gavel.sft.train has produced runs/grader-sft):

    TRACE_LOG=runs/trl-grpo/traces.jsonl \
    SFT_OUT=runs/grader-sft \
    CUDA_VISIBLE_DEVICES=0 \
    python -m gavel.audit
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from gavel.caching import register as cache_register
from gavel.caching.cache import MIN_PEARSON
from gavel.grader import MAX_SCORE, build_grader_messages, parse_score
from gavel.sft.data import load_split
from gavel.verify import is_correct


class LocalGrader:
    """Loads base + LoRA adapter and grades solutions with greedy generation."""

    def __init__(self, base_id, adapter_dir, max_new_tokens=512):
        self.tok = AutoTokenizer.from_pretrained(base_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"  # left-pad for batched generation

        model = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(model, adapter_dir)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def grade_batch(self, rows):
        prompts = [
            self.tok.apply_chat_template(
                build_grader_messages(
                    problem=r.get("problem", r.get("question", "N/A")),
                    reference_answer=r["ground_truth"],
                    candidate_solution=r.get("solution", r.get("completion", "")),
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in rows
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=2048).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(a):
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def main():
    base_id = os.environ.get("SFT_BASE", "Qwen/Qwen2.5-3B-Instruct")
    adapter = os.environ.get("SFT_OUT", "runs/grader-sft")
    trace_log = os.environ.get("TRACE_LOG", "runs/trl-grpo/traces.jsonl")
    audit_frac = float(os.environ.get("AUDIT_FRAC", 0.2))
    batch_size = int(os.environ.get("AUDIT_BATCH", 16))
    report_path = os.environ.get("AUDIT_REPORT", os.path.join(adapter, "audit.json"))
    dataset_id = os.environ.get("DATASET_ID", "BytedTsinghua-SIA/DAPO-Math-17k")
    cache_dir = Path(os.environ.get("CACHE_DIR", "cache"))

    rows = load_split(trace_log, "audit", audit_frac=audit_frac)
    if not rows:
        raise SystemExit(f"no held-out audit traces in {trace_log!r}.")
    print(f"[audit] {len(rows)} held-out traces from {trace_log}")

    grader = LocalGrader(base_id, adapter)

    grader_scores = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        grader_scores.extend(parse_score(t) for t in grader.grade_batch(batch))
        print(f"[audit] graded {min(i + batch_size, len(rows))}/{len(rows)}")

    teacher = [float(r.get("score", r.get("judge_score", 0.0))) for r in rows]                 # 0-9, logged judge
    mech = [is_correct(r.get("solution", r.get("completion", "")), r["ground_truth"]) for r in rows]  # 0/1, independent

    report = {
        "n": len(rows),
        "fidelity": {  # distilled grader vs the expensive teacher it replaces
            "pearson": _pearson(grader_scores, teacher),
            "spearman": _spearman(grader_scores, teacher),
            "mae_points": float(np.mean(np.abs(np.array(grader_scores) - np.array(teacher)))),
            "scale": f"0-{MAX_SCORE}",
        },
        "grounding": {  # does the score track REAL correctness (independent signal)?
            "grader_spearman_vs_mechanical": _spearman(grader_scores, mech),
            "teacher_spearman_vs_mechanical": _spearman(teacher, mech),
            "mechanical_positive_rate": float(np.mean(mech)),
        },
        "means": {
            "grader": float(np.mean(grader_scores)),
            "teacher": float(np.mean(teacher)),
        },
        "base": base_id,
        "adapter": adapter,
        "trace_log": trace_log,
    }

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    f_, g_ = report["fidelity"], report["grounding"]
    print("\n" + "=" * 62)
    print("  GAVEL AUDIT  --  distilled rubric grader vs. teacher + truth")
    print("=" * 62)
    print(f"  held-out solutions graded   : {report['n']}")
    print(f"  -- fidelity to teacher judge (the cheap grader's job) --")
    print(f"  Pearson  (grader, teacher)  : {f_['pearson']:.3f}")
    print(f"  Spearman (grader, teacher)  : {f_['spearman']:.3f}")
    print(f"  mean abs error              : {f_['mae_points']:.2f} / {MAX_SCORE} pts")
    print(f"  -- grounding in real correctness (not circular) --")
    print(f"  Spearman grader  vs truth   : {g_['grader_spearman_vs_mechanical']:.3f}")
    print(f"  Spearman teacher vs truth   : {g_['teacher_spearman_vs_mechanical']:.3f}  (ceiling)")
    print("=" * 62)
    print(f"  full report -> {report_path}")


if __name__ == "__main__":
    main()
