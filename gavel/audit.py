"""Audit -- THE HEADLINE NUMBER. Does the distilled grader track real truth?

This number is what wins or loses the project, and it does NOT depend on RL
convergence. On a held-out split of the trace log (completions the policy
produced + the teacher's scores), we:

  1. re-grade each completion with the distilled LoRA grader (local generation),
  2. compute an INDEPENDENT mechanical correctness for each (gavel/verify.py),
  3. report correlation(grader_score, mechanical_truth) + agreement with teacher.

Because step 2 shares no machinery with the LLM judge, a high correlation means
the cheap grader tracks *real* ground truth, not just that it parrots the judge.

Run (after distill.py has produced a grader):

    TRACE_LOG=runs/phase0-qwen3-4b/traces.jsonl \
    GRADER_OUT=runs/phase0-qwen3-4b/grader \
    CUDA_VISIBLE_DEVICES=1 \
    conda run --no-capture-output -n modelmerge python -m gavel.audit
"""

import json
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from gavel.reward import build_judge_messages, parse_score
from gavel.traces import load_split
from gavel.verify import is_correct


class LocalGrader:
    """Loads base + LoRA adapter and grades completions with greedy generation."""

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
                build_judge_messages(r["question"], r["ground_truth"], r["completion"]),
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
        traces = self.tok.batch_decode(gen, skip_special_tokens=True)
        return traces


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
    # average ranks for ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def main():
    base_id = os.environ.get("GRADER_BASE", "Qwen/Qwen2.5-3B-Instruct")
    adapter = os.environ.get("GRADER_OUT", "runs/phase0-qwen3-4b/grader")
    trace_log = os.environ.get("TRACE_LOG", "runs/phase0-qwen3-4b/traces.jsonl")
    audit_frac = float(os.environ.get("AUDIT_FRAC", 0.2))
    batch_size = int(os.environ.get("AUDIT_BATCH", 16))
    report_path = os.environ.get("AUDIT_REPORT", os.path.join(adapter, "audit.json"))

    rows = load_split(trace_log, "audit", audit_frac=audit_frac)
    if not rows:
        raise SystemExit(f"no held-out audit traces in {trace_log!r}.")
    print(f"[audit] {len(rows)} held-out traces from {trace_log}")

    grader = LocalGrader(base_id, adapter)

    grader_scores, traces = [], []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        bt = grader.grade_batch(batch)
        traces.extend(bt)
        grader_scores.extend(parse_score(t) for t in bt)
        print(f"[audit] graded {min(i + batch_size, len(rows))}/{len(rows)}")

    # Independent mechanical truth, and the teacher's own scores from the log.
    mech = [is_correct(r["completion"], r["ground_truth"]) for r in rows]
    teacher = [float(r.get("judge_score", 0.0)) for r in rows]

    g_bin = [1.0 if s >= 0.5 else 0.0 for s in grader_scores]
    t_bin = [1.0 if s >= 0.5 else 0.0 for s in teacher]

    def acc(pred, truth):
        return float(np.mean([p == t for p, t in zip(pred, truth)]))

    report = {
        "n": len(rows),
        "headline": {
            "grader_vs_mechanical_pearson": _pearson(grader_scores, mech),
            "grader_vs_mechanical_spearman": _spearman(grader_scores, mech),
            "grader_accuracy_vs_mechanical": acc(g_bin, mech),
        },
        "context": {
            "teacher_accuracy_vs_mechanical": acc(t_bin, mech),
            "grader_agreement_with_teacher": acc(g_bin, t_bin),
            "grader_vs_teacher_pearson": _pearson(grader_scores, teacher),
            "mechanical_positive_rate": float(np.mean(mech)),
        },
        "base": base_id,
        "adapter": adapter,
        "trace_log": trace_log,
    }

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    h, c = report["headline"], report["context"]
    print("\n" + "=" * 60)
    print("  GAVEL AUDIT  --  distilled grader vs. real ground truth")
    print("=" * 60)
    print(f"  held-out completions graded : {report['n']}")
    print(f"  Pearson  (grader, truth)    : {h['grader_vs_mechanical_pearson']:.3f}")
    print(f"  Spearman (grader, truth)    : {h['grader_vs_mechanical_spearman']:.3f}")
    print(f"  grader  accuracy vs truth   : {h['grader_accuracy_vs_mechanical']:.1%}")
    print(f"  teacher accuracy vs truth   : {c['teacher_accuracy_vs_mechanical']:.1%}  (ceiling)")
    print(f"  grader agreement w/ teacher : {c['grader_agreement_with_teacher']:.1%}")
    print("=" * 60)
    print(f"  full report -> {report_path}")


if __name__ == "__main__":
    main()
