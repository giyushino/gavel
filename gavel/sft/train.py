"""SFT distillation of the rubric grader.

Loads traces logged by gavel/grpo/grader.py during GRPO training and SFT-fines
a small model to reproduce the rubric grader's reasoning trace + aspect scores.
The result is a drop-in replacement for the expensive API judge: same prompt in,
same structured output, at a fraction of the cost and latency.

Run (after a GRPO run has populated TRACE_LOG):

    TRACE_LOG=gavel/grpo/traces.jsonl \
    SFT_OUT=runs/grader-sft \
    CUDA_VISIBLE_DEVICES=0 \
    python -m gavel.sft.train

Env vars (all optional — defaults shown):
    TRACE_LOG       path to the rubric-grader trace JSONL
    SFT_OUT         where to save the LoRA adapter
    SFT_BASE        base model to fine-tune
    AUDIT_FRAC      fraction of traces held out for audit  (default 0.2)
    BATCH_SIZE      per-device train batch size             (default 4)
    GRAD_ACCUM      gradient accumulation steps             (default 4)
    EPOCHS          number of training epochs               (default 3)
    LR              learning rate                           (default 2e-4)
    MAX_LEN         max sequence length                     (default 2048)
    LORA_R          LoRA rank                               (default 16)
    LORA_ALPHA      LoRA alpha                              (default 32)
    USE_WANDB       set to 1 to enable wandb logging        (default 0)
"""

import os

from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from gavel.sft.data import build_dataset


def _int(name, default):
    return int(os.environ.get(name, default))


def _float(name, default):
    return float(os.environ.get(name, default))


def _flag(name, default="0"):
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def main():
    trace_log = os.environ.get("TRACE_LOG", "gavel/grpo/traces.jsonl")
    out_dir = os.environ.get("SFT_OUT", "runs/grader-sft")
    base_id = os.environ.get("SFT_BASE", "Qwen/Qwen2.5-3B-Instruct")
    audit_frac = _float("AUDIT_FRAC", 0.2)

    dataset, rows = build_dataset(trace_log, which="train", audit_frac=audit_frac)
    if len(dataset) == 0:
        raise SystemExit(
            f"no usable training traces in {trace_log!r} "
            f"(after filters and holding out {audit_frac:.0%} for audit)."
        )
    print(f"[sft] {len(rows)} training traces from {trace_log}")

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
        num_train_epochs=_float("EPOCHS", 3),
        learning_rate=_float("LR", 2e-4),
        max_length=_int("MAX_LEN", 2048),
        bf16=True,
        gradient_checkpointing=True,
        packing=False,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["wandb"] if _flag("USE_WANDB") else [],
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
    print(f"[sft] saved LoRA grader -> {out_dir}")


if __name__ == "__main__":
    main()
