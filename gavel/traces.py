"""Load and split the trace log that the GRPO reward function writes for free.

Each row of `traces.jsonl` (see gavel/reward.py) is one grading call:
    {question, completion, ground_truth, judge_trace, judge_score}

`distill.py` trains on the TRAIN split; `audit.py` evaluates on the held-out
AUDIT split. Both call `load_split(...)` with the same args so the grader is
never audited on completions it was distilled from.
"""

import json
import os


def load_traces(path: str, drop_errors: bool = True, dedup: bool = True):
    """Read traces.jsonl into a list of dicts (order preserved)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"trace log not found: {path!r}. Run GRPO first (see RUN.md) so the "
            f"reward function logs traces, or set TRACE_LOG to an existing file."
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
                continue  # tolerate a torn final line from a killed run

            trace = r.get("judge_trace", "")
            if drop_errors and (not trace or trace.startswith("[judge error]")):
                continue

            if dedup:
                key = (r.get("question", ""), r.get("completion", ""))
                if key in seen:
                    continue
                seen.add(key)

            rows.append(r)
    return rows


def split_traces(rows, audit_frac: float = 0.2):
    """Deterministic split by position: last `audit_frac` is held out for audit.

    Position-based (not random) so distill and audit agree without sharing a
    seed, and so re-running either script is reproducible.
    """
    if not rows:
        return [], []
    n_audit = max(1, int(round(len(rows) * audit_frac))) if len(rows) > 1 else 0
    cut = len(rows) - n_audit
    return rows[:cut], rows[cut:]


def load_split(path: str, which: str, audit_frac: float = 0.2, **kw):
    """Convenience: return either the 'train' or 'audit' split of a trace log."""
    train, audit = split_traces(load_traces(path, **kw), audit_frac=audit_frac)
    if which == "train":
        return train
    if which == "audit":
        return audit
    raise ValueError(f"which must be 'train' or 'audit', got {which!r}")
