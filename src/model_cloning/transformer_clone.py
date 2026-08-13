"""Registers the existing weight-preserving clone (`clone.py::clone_model`) as a
named model_cloning strategy, so it fits the same registry pattern as every other
stage. `clone.py` itself is unmodified — this is a thin dispatch wrapper.
"""

from __future__ import annotations

from typing import Any

from src.model_cloning.base import ModelCloningStrategy
from src.model_cloning.registry import MODEL_CLONING_REGISTRY


@MODEL_CLONING_REGISTRY.register("transformer_clone")
class TransformerCloneStrategy(ModelCloningStrategy):
    """Weight-preserving clone (transformer-cloner) of `teacher_model` with
    `target_tokenizer` transplanted in, per `embedding_init_strategy`.

    params: teacher_model (str), max_seq_length (int).
    """

    def build(self, *, target_tokenizer: str | None, embedding_init_strategy: Any | None):
        from src.model_cloning.clone import clone_model  # deferred: heavy deps

        return clone_model(
            target_tokenizer=target_tokenizer,
            embedding_init_strategy=embedding_init_strategy,
            **self.params,
        )
