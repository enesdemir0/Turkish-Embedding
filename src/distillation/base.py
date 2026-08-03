"""Interface every distillation/training objective must implement.

`cosine.py` (Step 4) implements the paper's exact L_cos objective via
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
    def train(self, model: Any, tracker: Any | None = None) -> Any:
        """Train `model` in place (or return a new trained model) and return it.

        `tracker` is passed in by the pipeline (not read from `self.params`) since a
        live ExperimentTracker object can't be expressed in a YAML config — any
        strategy that wants to log live metrics (e.g. a per-step loss curve) takes
        it as a call argument instead."""
        raise NotImplementedError
