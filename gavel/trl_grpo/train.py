"""TRL GRPO + PEFT training with explicit rollout collection.

Rough parity with gavel/train_grpo.py but structured as a package module and
with a COLLECT_ONLY mode that just harvests judge-scored rollouts without doing
any gradient updates.

Run (judge server must be up — see scripts/serve_judge.sh):

    # Full GRPO training:
    OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY JUDGE_MODEL=judge \\
    CUDA_VISIBLE_DEVICES=1 python -m gavel.trl_grpo.train

    # Rollout collection only (no gradient updates):
    COLLECT_ONLY=1 \\
    OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY JUDGE_MODEL=judge \\
    CUDA_VISIBLE_DEVICES=1 python -m gavel.trl_grpo.train

Knobs are env vars (see DEFAULTS below).
"""

import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from gavel.data import build_dataset
from gavel.reward import JudgeReward
from gavel.trl_grpo.rollout import RolloutCollector
from gavel.trl_grpo.trainer import GRPOTrainerGCFixed


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_flag(name, default="0"):
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=env_int("LORA_R", 16),
        lora_alpha=env_int("LORA_ALPHA", 32),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def collect_only(model_id: str, tokenizer, dataset, judge: JudgeReward) -> None:
    """Load model, collect rollouts for the full dataset, and exit."""
    rollout_log = os.environ.get("ROLLOUT_LOG", "rollouts.jsonl")
    num_generations = env_int("NUM_GENERATIONS", 8)
    max_new_tokens = env_int("MAX_COMPLETION_LEN", 1024)
    temperature = float(os.environ.get("TEMPERATURE", 1.0))
    batch_size = env_int("COLLECT_BATCH_SIZE", 4)

    print(f"[collect] loading {model_id} for rollout collection…")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    peft_config = build_peft_config()
    model = get_peft_model(model, peft_config)
    model.eval()

    collector = RolloutCollector(
        model=model,
        tokenizer=tokenizer,
        judge=judge,
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        log_path=rollout_log,
    )
    print(f"[collect] writing rollouts to {rollout_log}")
    collector.run_dataset(dataset, batch_size=batch_size)
    print("[collect] done")


def train(model_id: str, tokenizer, dataset, judge: JudgeReward) -> None:
    """Run GRPO training with TRL + PEFT LoRA."""
    output_dir = os.environ.get("OUTPUT_DIR", "runs/trl-grpo")

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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=env_flag("USE_VLLM", "0"),
        log_completions=True,
        report_to=["wandb"] if env_flag("USE_WANDB", "0") else [],
    )

    trainer = GRPOTrainerGCFixed(
        model=model_id,
        args=config,
        train_dataset=dataset,
        reward_funcs=judge,
        peft_config=build_peft_config(),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)


def main():
    model_id = os.environ.get("POLICY_MODEL", "Qwen/Qwen3-4B")
    n_examples = os.environ.get("N_EXAMPLES")
    n_examples = int(n_examples) if n_examples else None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = build_dataset(tokenizer, n=n_examples, enable_thinking=False)

    first = dataset[0]
    toks = tokenizer(first["prompt"], return_tensors="pt")
    print(f"\n[debug] tokenizer pad={tokenizer.pad_token!r}({tokenizer.pad_token_id}) "
          f"eos={tokenizer.eos_token!r}({tokenizer.eos_token_id})")
    print(f"[debug] first prompt token count: {toks['input_ids'].shape[1]}")
    print(f"[debug] prompt tail (last 120 chars): {repr(first['prompt'][-120:])}")

    # Quick direct-generation smoke-test: if this produces garbage, the issue is
    # in the model/tokenizer, not in TRL. Runs on CPU-offloaded model, cheap.
    print("[debug] running direct greedy generation (20 tokens)…")
    _m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    _in = tokenizer(first["prompt"], return_tensors="pt").to(_m.device)
    with torch.no_grad():
        _out = _m.generate(**_in, max_new_tokens=20, do_sample=False)
    _completion = tokenizer.decode(_out[0][_in["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[debug] direct generation: {repr(_completion)}")
    del _m, _in, _out
    torch.cuda.empty_cache()
    print()

    # JudgeReward logs traces to TRACE_LOG; in collect-only mode we also write
    # structured rollouts (with prompt) to ROLLOUT_LOG via RolloutCollector.
    judge = JudgeReward()

    if env_flag("COLLECT_ONLY"):
        collect_only(model_id, tokenizer, dataset, judge)
    else:
        train(model_id, tokenizer, dataset, judge)


if __name__ == "__main__":
    main()
