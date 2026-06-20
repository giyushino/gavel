"""Load and format rubric-grader traces for SFT.

Each row written by gavel/grpo/grader.py looks like:
    {data_source, problem, ground_truth, solution, judge_trace, score}

We reconstruct the exact messages the rubric grader received (via the shared
build_grader_messages), so the distilled grader sees identical inputs at
inference time. Rows logged before `problem` was added fall back to "N/A",
matching grader.py's own default when extra_info is absent.
"""

import json
import os

from datasets import Dataset

from gavel.grader import build_grader_messages


def load_traces(path: str, drop_errors: bool = True, dedup: bool = True):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"trace log not found: {path!r}. Run GRPO first so grader.py "
            f"accumulates traces, or set TRACE_LOG to an existing file."
        )

    rows, seen = [], set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            trace = r.get("judge_trace", "")
            if drop_errors and (not trace or trace.startswith("[judge error]")):
                continue

            if dedup:
                key = (r.get("ground_truth", ""), r.get("solution", r.get("completion", "")))
                if key in seen:
                    continue
                seen.add(key)

            rows.append(r)
    return rows


def split_traces(rows, audit_frac: float = 0.2):
    if not rows:
        return [], []
    n_audit = max(1, int(round(len(rows) * audit_frac))) if len(rows) > 1 else 0
    cut = len(rows) - n_audit
    return rows[:cut], rows[cut:]


def load_split(path: str, which: str, audit_frac: float = 0.2, **kw):
    train, audit = split_traces(load_traces(path, **kw), audit_frac=audit_frac)
    if which == "train":
        return train
    if which == "audit":
        return audit
    raise ValueError(f"which must be 'train' or 'audit', got {which!r}")


def to_example(r: dict) -> dict:
    """Convert one trace row into the prompt/completion format TRL SFTTrainer expects."""
    return {
        "prompt": build_grader_messages(
            problem=r.get("problem", r.get("question", "N/A")),
            reference_answer=r["ground_truth"],
            candidate_solution=r.get("solution", r.get("completion", "")),
        ),
        "completion": [{"role": "assistant", "content": r["judge_trace"]}],
    }


def build_dataset(
    path: str,
    which: str = "train",
    audit_frac: float = 0.2,
) -> tuple[Dataset, list[dict]]:
    rows = load_split(path, which, audit_frac=audit_frac)
    return Dataset.from_list([to_example(r) for r in rows]), rows
