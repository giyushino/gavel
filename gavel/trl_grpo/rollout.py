"""Explicit rollout collection: generate completions and score with the LLM judge.

Used standalone (COLLECT_ONLY=1) to harvest a scored rollout log without doing
any gradient updates, or called from train.py as a side-channel that saves
rollouts independently of TRL's internal generation loop.
"""

import json
import os
import threading

import torch

from gavel.reward import JudgeReward


class RolloutCollector:
    """Generate completions from a policy and score them with the LLM judge.

    Works with any HF CausalLM (plain or PEFT-wrapped).  Results are appended
    to `log_path` as JSONL so downstream distillation can consume them.
    """

    def __init__(
        self,
        model,
        tokenizer,
        judge: JudgeReward,
        num_generations: int = 8,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        log_path: str = "rollouts.jsonl",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.judge = judge
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.log_path = log_path
        self._lock = threading.Lock()

    @torch.no_grad()
    def _generate(self, prompt: str) -> list[str]:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=self.num_generations,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return [
            self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            for out in outputs
        ]

    def collect_batch(
        self,
        prompts: list[str],
        questions: list[str],
        ground_truths: list[str],
    ) -> tuple[list[str], list[float]]:
        """Generate + score + log rollouts for a batch of prompts.

        Returns flat (completions, scores) over all prompts × num_generations.
        """
        all_completions: list[str] = []
        all_scores: list[float] = []
        rows: list[dict] = []

        for prompt, question, gt in zip(prompts, questions, ground_truths):
            completions = self._generate(prompt)
            scores = self.judge(
                completions=completions,
                question=[question] * len(completions),
                ground_truth=[gt] * len(completions),
            )
            for c, s in zip(completions, scores):
                rows.append(
                    {
                        "prompt": prompt,
                        "question": question,
                        "ground_truth": gt,
                        "completion": c,
                        "score": s,
                    }
                )
            all_completions.extend(completions)
            all_scores.extend(scores)

        if self.log_path:
            with self._lock, open(self.log_path, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

        return all_completions, all_scores

    def run_dataset(self, dataset, batch_size: int = 4) -> None:
        """Collect rollouts for an entire HF dataset (collect-only mode)."""
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total = len(dataset)
        processed = 0
        for batch in loader:
            self.collect_batch(
                prompts=batch["prompt"],
                questions=batch["question"],
                ground_truths=batch["ground_truth"],
            )
            processed += len(batch["prompt"])
            print(f"[rollout] {processed}/{total}", flush=True)
