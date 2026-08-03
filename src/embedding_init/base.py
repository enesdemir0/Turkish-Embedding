"""Interface every embedding-initialization strategy must implement.

This is the `Compose()` slot from the paper (§3.3 of EMBEDD~3.MD): given the teacher's
embedding table and a mapping from target-vocab tokens to teacher-vocab token
sequences, produce the target vocabulary's new embedding matrix.

`compose()` returns one of two things:
  - `None` — the strategy is already what `transformer_cloner`'s built-in cloning
    does natively (true today only for `mean_composition`, since the installed
    transformer-cloner always mean-composes internally). model_cloning/clone.py
    takes the native clone's embedding table as-is.
  - a `(vocab_size x hidden_dim)` tensor — a strategy with no transformer-cloner
    equivalent (FOCUS, WECHSEL, ...) computed its own matrix; clone.py overwrites
    the natively-cloned embedding table with it. Nothing else in the pipeline needs
    to know which case it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingInitStrategy(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def compose(self, teacher_model: Any, target_tokenizer: Any, teacher_tokenizer: Any) -> Any | None:
        """Return a (vocab_size x hidden_dim) embedding matrix, or None (see module docstring)."""
        raise NotImplementedError
