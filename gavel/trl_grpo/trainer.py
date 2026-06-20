"""GRPOTrainer subclass that fixes gradient-checkpointing + KV-cache conflict.

Problem
-------
Qwen3 (and other models built on transformers.modeling_layers.GradientCheckpointingLayer)
checks `self.gradient_checkpointing and self.training` in every decoder layer's
__call__. When both are True it nullifies `past_key_values`, disabling the KV
cache. TRL never calls model.eval() before generation, so the model stays in
training mode and every generate() step after the first token sees no context →
garbage output.

Fix
---
Override _generate_and_score_completions to flip gradient_checkpointing=False on
all affected modules before the generation+reward pass, then restore it after.
_compute_loss (the actual backward) runs with gradient checkpointing on as normal,
so activation memory stays low.
"""

from trl import GRPOTrainer


class GRPOTrainerGCFixed(GRPOTrainer):
    """GRPOTrainer that disables gradient checkpointing on decoder layers during
    rollout generation to prevent Qwen3 from nullifying the KV cache."""

    def _generate_and_score_completions(self, inputs):
        gc_modules = [
            m for m in self.model.modules()
            if getattr(m, "gradient_checkpointing", False)
        ]
        for m in gc_modules:
            m.gradient_checkpointing = False
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            for m in gc_modules:
                m.gradient_checkpointing = True
