"""Interface every embedding-initialization strategy must implement.

This is the `Compose()` slot from the paper (§3.3 of EMBEDD~3.MD): given the teacher's
embedding table and a mapping from target-vocab tokens to teacher-vocab token
sequences, produce the target vocabulary's new embedding matrix. `mean_composition.py`
(Step 3) implements the paper's exact strategy; `focus_init.py` / `wechsel_init.py`
(future work) are drop-in replacements — model_cloning/clone.py depends only on this
interface, never on a specific strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingInitStrategy(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def compose(self, teacher_model: Any, target_tokenizer: Any, teacher_tokenizer: Any) -> Any:
        """Return the new (vocab_size x hidden_dim) embedding matrix for target_tokenizer."""
        raise NotImplementedError
