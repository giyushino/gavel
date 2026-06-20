"""Distill the logged judge traces into a small LoRA grader -- the whole point.

The GRPO reward function already paid to grade every rollout and logged each
call to `traces.jsonl`. Here we turn that free, labeled set into a cheap grader:
SFT a small base model to reproduce the teacher judge's reasoning trace + score.

    input  = build_judge_messages(question, ground_truth, completion)  # what the teacher saw
    target = judge_trace                                               # ...ending in `SCORE: <x>`

We train in prompt->completion form, so only the trace tokens carry loss. The
result is a drop-in for the teacher: same prompt, same `SCORE:` output, a
fraction of the cost/latency.

Run (after a GRPO run has produced traces -- see RUN.md):

    TRACE_LOG=runs/phase0-qwen3-4b/traces.jsonl \
    GRADER_OUT=runs/phase0-qwen3-4b/grader \
    CUDA_VISIBLE_DEVICES=1 \
    conda run --no-capture-output -n modelmerge python -m gavel.distill
"""

import os

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from gavel.reward import build_judge_messages
from gavel.traces import load_split


def _int(name, default):
    return int(os.environ.get(name, default))


def _to_example(r):
    return {
        "prompt": build_judge_messages(r["question"], r["ground_truth"], r["completion"]),
        "completion": [{"role": "assistant", "content": r["judge_trace"]}],
    }


def main():
    base_id = os.environ.get("GRADER_BASE", "Qwen/Qwen2.5-3B-Instruct")
    out_dir = os.environ.get("GRADER_OUT", "runs/phase0-qwen3-4b/grader")
    trace_log = os.environ.get("TRACE_LOG", "runs/phase0-qwen3-4b/traces.jsonl")
    audit_frac = float(os.environ.get("AUDIT_FRAC", 0.2))

    rows = load_split(trace_log, "train", audit_frac=audit_frac)
    if not rows:
        raise SystemExit(
            f"no usable training traces in {trace_log!r} "
            f"(after dropping judge errors and holding out {audit_frac:.0%} for audit)."
        )
    print(f"[distill] {len(rows)} training traces from {trace_log}")

    dataset = Dataset.from_list([_to_example(r) for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=_int("LORA_R", 16),
        lora_alpha=_int("LORA_ALPHA", 32),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    config = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=_int("BATCH_SIZE", 4),
        gradient_accumulation_steps=_int("GRAD_ACCUM", 4),
        num_train_epochs=float(os.environ.get("EPOCHS", 3)),
        learning_rate=float(os.environ.get("LR", 2e-4)),
        max_length=_int("MAX_LEN", 2048),
        bf16=True,
        gradient_checkpointing=True,
        packing=False,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=base_id,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    print(f"[distill] saved LoRA grader -> {out_dir}")


if __name__ == "__main__":
    main()
