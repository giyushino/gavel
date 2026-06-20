"""Phase 0: GRPO training of Qwen3-4B on DAPO-Math-17k with an LLM-judge reward.

Run (after the judge server is up, see scripts/serve_judge.sh):

    OPENAI_BASE_URL=http://localhost:8000/v1 \
    OPENAI_API_KEY=EMPTY \
    JUDGE_MODEL=judge \
    CUDA_VISIBLE_DEVICES=1 \
    python -m gavel.train_grpo

Knobs are env vars (see DEFAULTS below). LoRA keeps the 4B policy on a single
A40; generation is plain HF for Phase 0 robustness (flip USE_VLLM=1 later).
"""

import os

import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from gavel.data import build_dataset
from gavel.reward import JudgeReward


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_flag(name, default="0"):
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def main():
    model_id = os.environ.get("POLICY_MODEL", "Qwen/Qwen3-4B")
    output_dir = os.environ.get("OUTPUT_DIR", "runs/phase0-qwen3-4b")
    n_examples = os.environ.get("N_EXAMPLES")
    n_examples = int(n_examples) if n_examples else None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(tokenizer, n=n_examples, enable_thinking=False)

    reward = JudgeReward()

    peft_config = LoraConfig(
        r=env_int("LORA_R", 16),
        lora_alpha=env_int("LORA_ALPHA", 32),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=env_int("BATCH_SIZE", 8),
        gradient_accumulation_steps=env_int("GRAD_ACCUM", 4),
        num_generations=env_int("NUM_GENERATIONS", 8),
        max_prompt_length=env_int("MAX_PROMPT_LEN", 1024),
        max_completion_length=env_int("MAX_COMPLETION_LEN", 1024),
        learning_rate=float(os.environ.get("LR", 1e-5)),
        beta=float(os.environ.get("KL_BETA", 0.04)),
        temperature=float(os.environ.get("TEMPERATURE", 1.0)),
        max_steps=env_int("MAX_STEPS", 200),
        logging_steps=1,
        save_steps=env_int("SAVE_STEPS", 50),
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=env_flag("USE_VLLM", "0"),
        log_completions=True,
        report_to=["wandb"] if env_flag("USE_WANDB", "0") else [],
    )

    trainer = GRPOTrainer(
        model=model_id,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
