"""Independent mechanical ground-truth check for DAPO-Math completions.

This deliberately calls NO model. It extracts the final answer from a completion
(last ``\\boxed{...}``, else last number) and compares it to the reference answer
with light normalization. Because it shares no machinery with the LLM judge, it
is a genuinely independent signal -- so "the grader just agrees with the judge"
circularity does not apply to the audit's headline correlation.

DAPO-Math `ground_truth` is typically a bare value, e.g. "34" or "1/2".
"""

import re
from fractions import Fraction

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_TOL = 1e-6


def extract_answer(text: str):
    """Best-effort final-answer extraction: prefer \\boxed{}, else last number."""
    if not text:
        return None
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = _NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def _normalize(s):
    """Coerce an answer string to a float when possible, else a cleaned string."""
    if s is None:
        return None
    s = s.strip().strip("$").replace(" ", "").replace(",", "").rstrip(".")
    if s == "":
        return None
    try:
        return float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        return s.lower()


def is_correct(completion: str, ground_truth: str) -> float:
    """1.0 if the completion's final answer matches the reference, else 0.0.

    `ground_truth` may itself be wrapped in \\boxed{}; handle both.
    """
    a = _normalize(extract_answer(completion))
    gt_raw = extract_answer(ground_truth) if "\\boxed" in (ground_truth or "") else ground_truth
    b = _normalize(gt_raw)
    if a is None or b is None:
        return 0.0
    if isinstance(a, float) and isinstance(b, float):
        return 1.0 if abs(a - b) < _TOL else 0.0
    return 1.0 if str(a) == str(b) else 0.0
