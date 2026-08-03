"""Interface every tokenizer-surgery strategy must implement.

Concrete strategies (`reuse_pretrained.py`; a future `frequency_pruning.py` for
from-scratch construction, `morphology_aware.py` for Zemberek-segmented pruning)
register themselves with TOKENIZER_SURGERY_REGISTRY and are selected purely by name
from the experiment config — the pipeline never imports a concrete strategy directly.

`build()` returns a *string* (an HF repo id or a local directory path), not a loaded
tokenizer object — confirmed against transformer_cloner's real API, whose cloners
take `target_tokenizer_id: str` and load it themselves via `AutoTokenizer.
from_pretrained` internally. A from-scratch strategy trains its tokenizer and saves
it to a local directory, then returns that directory's path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TokenizerSurgeryStrategy(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def build(self) -> str:
        """Construct the target (student) tokenizer; return its HF repo id or local path."""
        raise NotImplementedError
