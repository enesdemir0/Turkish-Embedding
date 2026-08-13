"""Interface every model-cloning strategy must implement.

Unlike the other stages, a model-cloning strategy may not need a target tokenizer
or an embedding-init composition at all (e.g. `native_pretrained`, which loads an
existing model that already has its own co-trained tokenizer). The
`requires_tokenizer_surgery`/`requires_embedding_init` class flags let
`src/pipeline.py` decide whether to run those upstream stages *before*
instantiating this one, without every strategy needing its own special-cased
orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelCloningStrategy(ABC):
    requires_tokenizer_surgery: bool = True
    requires_embedding_init: bool = True

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def build(self, *, target_tokenizer: str | None, embedding_init_strategy: Any | None) -> Any:
        """Return a sentence_transformers.SentenceTransformer ready for distillation.

        target_tokenizer/embedding_init_strategy are None when this strategy's
        requires_* flags are False — the pipeline skips those stages entirely.
        """
        raise NotImplementedError
