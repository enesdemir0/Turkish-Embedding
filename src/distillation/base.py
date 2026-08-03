"""Interface every distillation/training objective must implement.

`cosine_distillation.py` (Step 4) implements the paper's exact L_cos objective via
distil-trainer. Future objectives (contrastive fine-tune with hard negatives,
Matryoshka loss) register under new names — the pipeline calls `.train()` without
knowing which objective is behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DistillationObjective(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def train(self, model: Any) -> Any:
        """Train `model` in place (or return a new trained model) and return it."""
        raise NotImplementedError
