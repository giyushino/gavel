"""LLM-judge reward function for GRPO.

The reward function calls *another* LLM through the OpenAI client. Right now that
endpoint is a locally-served teacher (vLLM's OpenAI-compatible server), but
because we only ever speak the OpenAI API, swapping in our own distilled grader
later is a one-line base_url change -- which is the whole point of the project.

It also logs {question, completion, judge_trace, score, ground_truth} to a JSONL
file inside the reward call. That log is the free, labeled distillation set: the
grading is a sunk cost we are paying anyway, so we capture it.
"""

import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

_DEFAULT_JUDGE_SYSTEM = (
    "You are a strict grader for math problems. You are given a problem, the "
    "reference final answer, and a student's full solution. Decide whether the "
    "student's FINAL answer matches the reference answer (allow equivalent forms, "
    "e.g. 1/2 == 0.5). Reason briefly in one or two sentences, then output, on its "
    "own final line, exactly:\nSCORE: <x>\nwhere <x> is 1.0 if the final answer is "
    "correct and 0.0 if it is wrong."
)

# Set JUDGE_SYSTEM_PROMPT at launch time to use a custom generated rubric.
JUDGE_SYSTEM = os.environ.get("JUDGE_SYSTEM_PROMPT", _DEFAULT_JUDGE_SYSTEM)

# When using a multi-dimension rubric (e.g. max 9), set this to normalize to [0, 1].
_JUDGE_MAX_SCORE = float(os.environ.get("JUDGE_MAX_SCORE", "1"))

JUDGE_USER_TMPL = (
    "Problem:\n{question}\n\n"
    "Reference answer:\n{ground_truth}\n\n"
    "Student solution:\n{completion}\n\n"
    "Grade it."
)

_XML_TOTAL_RE = re.compile(r"<total>\s*(\d+)\s*</total>")
_SCORE_RE = re.compile(r"SCORE:\s*([0-9]*\.?[0-9]+)")


def parse_score(text: str) -> float:
    """Extract score from XML <total> or fallback SCORE: format. Returns raw value."""
    m = _XML_TOTAL_RE.search(text or "")
    if m:
        return float(m.group(1))
    matches = _SCORE_RE.findall(text or "")
    if not matches:
        return 0.0
    try:
        return float(matches[-1])
    except ValueError:
        return 0.0


# Backwards-compatible private alias.
_parse_score = parse_score


def build_judge_messages(question: str, ground_truth: str, completion: str):
    """The exact chat messages the teacher judge sees for one grading call.

    Shared by the live reward function, the distillation SFT target, and the
    audit so the distilled grader is a true drop-in: same prompt in, same
    `SCORE:` format out.
    """
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER_TMPL.format(
                question=question,
                ground_truth=ground_truth,
                completion=completion,
            ),
        },
    ]


class JudgeReward:
    """Callable reward function compatible with TRL's GRPOTrainer.

    Trainer calls it as reward(prompts=..., completions=..., **columns), passing
    each extra dataset column (here: `question`, `ground_truth`) as a kwarg list.
    """

    __name__ = "judge_reward"  # TRL uses this for logging

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        log_path: str | None = None,
        max_workers: int = 32,
        max_tokens: int = 2048,
        log_samples: int | None = None,
    ):
        self.model = model or os.environ.get("JUDGE_MODEL", "judge")
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
        self.log_path = log_path or os.environ.get("TRACE_LOG", "traces.jsonl")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._step = 0
        self.log_samples = log_samples if log_samples is not None else int(os.environ.get("LOG_SAMPLES", 2))

    def _grade_one(self, question: str, completion: str, ground_truth: str):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=build_judge_messages(question, ground_truth, completion),
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            trace = resp.choices[0].message.content or ""
        except Exception as e:  # never let a judge hiccup kill a training step
            trace = f"[judge error] {e}"
        return trace, parse_score(trace)

    def _log(self, rows):
        with self._lock, open(self.log_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def __call__(self, completions, prompts=None, question=None, ground_truth=None, **kwargs):
        n = len(completions)
        questions = question if question is not None else [""] * n
        gts = ground_truth if ground_truth is not None else [""] * n

        results = list(
            self.executor.map(
                self._grade_one, questions, completions, gts
            )
        )
        scores = [s for _, s in results]

        self._log(
            {
                "question": questions[i],
                "completion": completions[i],
                "ground_truth": gts[i],
                "judge_trace": results[i][0],
                "judge_score": results[i][1],
            }
            for i in range(n)
        )

        self._step += 1
        if self.log_samples > 0:
            indices = random.sample(range(n), min(self.log_samples, n))
            sep = "─" * 60
            print(f"\n{sep}  step {self._step}  {sep}")
            for idx in indices:
                trace, score = results[idx]
                print(f"Q:\n{questions[idx]}")
                print(f"\nANSWER:\n{completions[idx]}")
                print(f"\nGRADE:\n{trace}")
                print(f"\nSCORE: {score:.3f}")
                print()

        return scores
