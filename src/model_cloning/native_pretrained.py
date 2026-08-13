"""Loads an existing pretrained encoder that already has its own co-trained
tokenizer (e.g. BERTurk) and wraps it with mean pooling into a
SentenceTransformer, ready for distillation.

No tokenizer_surgery, no embedding_init composition — there's no cross-tokenizer
graft happening, so there's nothing for the backbone to be unfamiliar with. This
sidesteps the exact failure mode that sank exp006 (frequency_pruning) and exp007
(BERTurk used as a tokenizer transplant onto the Gemma backbone): a backbone whose
attention layers never learned the new segmentation scheme, which a small training
budget can't fully re-align.

`model_cloning.teacher_model` in the experiment config, if present, is NOT read
here — it identifies the *distillation* teacher (google/embeddinggemma-300m) for
naming/tagging only. This strategy's actual source model is `pretrained_model`.
"""

from __future__ import annotations

import logging
from typing import Any

from src.model_cloning.base import ModelCloningStrategy
from src.model_cloning.registry import MODEL_CLONING_REGISTRY

logger = logging.getLogger(__name__)


@MODEL_CLONING_REGISTRY.register("native_pretrained")
class NativePretrainedStrategy(ModelCloningStrategy):
    """params:
        pretrained_model: HF repo id to load (required), e.g.
            "dbmdz/bert-base-turkish-128k-cased".
        max_seq_length: student's max sequence length. Should not exceed the
            model's native max_position_embeddings (512 for BERTurk) — a stock
            BERT-family model's learned absolute position embeddings don't
            extrapolate past what it was pretrained with, unlike the paper's
            Gemma-derived clone, which extends context via a structural choice
            specific to that architecture.
        pooling_mode: passed to sentence_transformers.models.Pooling (default "mean").
    """

    requires_tokenizer_surgery = False
    requires_embedding_init = False

    def build(self, *, target_tokenizer: str | None, embedding_init_strategy: Any | None):
        from sentence_transformers import SentenceTransformer, models

        pretrained_model = self.params.get("pretrained_model")
        if not pretrained_model:
            raise ValueError(
                "native_pretrained model_cloning strategy requires a 'pretrained_model' "
                "param (HF repo id)"
            )
        max_seq_length = self.params.get("max_seq_length")
        pooling_mode = self.params.get("pooling_mode", "mean")

        transformer = models.Transformer(pretrained_model, max_seq_length=max_seq_length)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling_mode)
        model = SentenceTransformer(modules=[transformer, pooling])
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length

        logger.info("Loaded native pretrained backbone %s (%s pooling)", pretrained_model, pooling_mode)
        return model
