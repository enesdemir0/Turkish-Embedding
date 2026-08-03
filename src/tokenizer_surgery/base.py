"""Interface every tokenizer-surgery strategy must implement.

Concrete strategies (e.g. `frequency_pruning.py`, added in Step 2; a future
`morphology_aware.py` for Zemberek-segmented pruning) register themselves with
TOKENIZER_SURGERY_REGISTRY and are selected purely by name from the experiment config
— the pipeline never imports a concrete strategy directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TokenizerSurgeryStrategy(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def build(self) -> Any:
        """Construct and return the target (student) tokenizer."""
        raise NotImplementedError
