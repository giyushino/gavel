"""Dataset loading + formatting for GRPO.

DAPO-Math-17k rows look like:
    data_source:  "math_dapo"
    prompt:       [{"role": "user", "content": "<problem ... Answer: $Answer>"}]
    ability:      "MATH"
    reward_model: {"ground_truth": "34", "style": "..."}
    extra_info:   {"index": "..."}

GRPO wants a `prompt` column. We pre-render the chat template to a plain
string here (rather than handing GRPO the conversational list) so we can pin
`enable_thinking=False` on Qwen3 and keep completions short and parseable for
Phase 0. The `question` and `ground_truth` columns ride along and are handed to
the reward function as keyword args by the trainer.
"""

import os

from datasets import load_dataset

DEFAULT_DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"


def _extract(ex, dataset_id: str) -> dict:
    """Normalize a raw dataset row to {question, ground_truth}."""
    did = dataset_id.lower()
    if "gsm8k" in did:
        question = ex["question"]
        # GSM8K answers look like "... #### 42"
        answer = ex["answer"].split("####")[-1].strip()
        return {"question": question, "ground_truth": answer}
    else:
        # DAPO-Math-17k and similar
        return {
            "question": ex["prompt"][0]["content"],
            "ground_truth": ex["reward_model"]["ground_truth"],
        }


def build_dataset(tokenizer, n: int | None = None, enable_thinking: bool = False):
    dataset_id = os.environ.get("DATASET_ID", DEFAULT_DATASET_ID)
    is_gsm8k = "gsm8k" in dataset_id.lower()
    ds = load_dataset(dataset_id, "main" if is_gsm8k else None, split="train")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    def render(ex):
        norm = _extract(ex, dataset_id)
        messages = [{"role": "user", "content": norm["question"] + "\n\nThink step by step and put your final answer in \\boxed{}."}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return {"prompt": prompt, "question": norm["question"], "ground_truth": norm["ground_truth"]}

    ds = ds.map(render, remove_columns=ds.column_names)
    return ds
