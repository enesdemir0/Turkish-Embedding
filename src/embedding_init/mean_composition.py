"""The paper's exact embedding-init strategy: E'_j = mean(E_i1, ..., E_ik) — see
EMBEDD~3.MD §3.3.

`transformer_cloner.SentenceTransformerCloner` (the class model_cloning/clone.py
uses) always performs this exact mean-composition internally when cloning the
Transformer module — confirmed by reading its installed source (v0.2.10), which
calls `source_embeddings.mean(dim=0)` and has no parameter to select a different
strategy. So this strategy is a no-op: returning None tells clone.py "the native
clone already is this — nothing to override."

Strategies with no transformer-cloner equivalent (FOCUS, WECHSEL, ...) instead
return an actual (vocab_size x hidden_dim) tensor from `compose()`, which clone.py
injects by overwriting the cloned model's embedding weights after the fact.
"""

from __future__ import annotations

from src.embedding_init.base import EmbeddingInitStrategy
from src.embedding_init.registry import EMBEDDING_INIT_REGISTRY


@EMBEDDING_INIT_REGISTRY.register("mean_composition")
class MeanCompositionEmbeddingInit(EmbeddingInitStrategy):
    def compose(self, teacher_model, target_tokenizer, teacher_tokenizer):
        return None
