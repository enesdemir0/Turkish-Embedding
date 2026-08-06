"""No-op distillation objective — returns the cloned model unchanged, with zero
training steps.

Exists to answer one specific question: how much of the final MTEB score comes
from the cloned teacher backbone + embedding_init alone, versus what distillation
training adds on top. Since distil-trainer trains the *entire* student model (no
frozen layers — confirmed by reading its installed source, embedding_trainer.py:
`AdamW(self.student_model.parameters())`), the backbone is only guaranteed
identical to the teacher's at this exact point, before any training runs. Comparing
this checkpoint's eval numbers against a real distillation run's is the only way to
measure that gap directly rather than assume it.

Not meant for production models — pair with configs/experiments/exp003_zero_distillation.yaml.
"""

from __future__ import annotations

from typing import Any

from src.distillation.base import DistillationObjective
from src.distillation.registry import DISTILLATION_REGISTRY


@DISTILLATION_REGISTRY.register("skip")
class SkipDistillationObjective(DistillationObjective):
    """params: none. Returns `model` as-is, with no training performed."""

    def train(self, model: Any, tracker: Any | None = None) -> Any:
        if tracker is not None:
            tracker.log_params({"distillation_skipped": True})
        return model
