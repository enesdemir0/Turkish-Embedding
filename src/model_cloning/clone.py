"""Backbone-preserving clone (transformer-cloner) + the paper's structural override
(extended context window — EMBEDD~3.MD §4.1).

Pooling/Dense/Normalize modules are inherited as-is from the teacher via
`SentenceTransformerCloner`, since the teacher (EmbeddingGemma-300M) is itself
already a packaged SentenceTransformer with those modules — cloning copies their
config/weights verbatim, matching the paper's "backbone preserved exactly"
principle (EMBEDD~3.MD §3.1). This is *not* re-derived from the paper's stated
values (mean pooling w/ include_prompt=True; Dense 768->3072->768, no bias, Identity
activation; L2 normalize) — `_log_structural_config` surfaces the actually-cloned
config so a mismatch against those stated values is visible rather than silently
assumed or force-overwritten (which would discard real cloned weights).

The embedding table's content is decided by whichever EmbeddingInitStrategy the
config selected — see embedding_init/base.py for how a strategy can either accept
the native transformer-cloner clone (mean_composition, today's only option) as-is,
or supply its own matrix to inject (FOCUS/WECHSEL, future work).
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def clone_model(
    *,
    teacher_model: str,
    target_tokenizer: str,
    embedding_init_strategy: Any,
    max_seq_length: int,
    **extra_params: Any,
):
    """Returns a sentence_transformers.SentenceTransformer.

    teacher_model: HF repo id of the teacher (e.g. "google/embeddinggemma-300m").
    target_tokenizer: HF repo id or local path for the student's tokenizer — whatever
        the tokenizer_surgery stage's build() returned.
    embedding_init_strategy: an EmbeddingInitStrategy instance.
    max_seq_length: student's max sequence length (paper: 8192, vs teacher's 2048).
    """
    if extra_params:
        logger.warning("clone_model() received unrecognized model_cloning params: %s", list(extra_params))

    from transformer_cloner import SentenceTransformerCloner

    cloner = SentenceTransformerCloner(model_path=teacher_model, target_tokenizer_id=target_tokenizer)
    cloner.clone(verbose=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cloner.save(tmp_dir)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(tmp_dir)

    _maybe_override_embeddings(model, teacher_model, target_tokenizer, embedding_init_strategy)

    model.max_seq_length = max_seq_length
    _log_structural_config(model)

    return model


def _maybe_override_embeddings(model, teacher_model: str, target_tokenizer: str, embedding_init_strategy: Any) -> None:
    from transformers import AutoTokenizer

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    composed_matrix = embedding_init_strategy.compose(
        teacher_model=teacher_model,
        target_tokenizer=target_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )
    if composed_matrix is None:
        return  # native transformer-cloner clone (mean-composition) already applies

    import torch

    embedding_layer = model[0].auto_model.get_input_embeddings()
    with torch.no_grad():
        embedding_layer.weight.copy_(torch.as_tensor(composed_matrix, dtype=embedding_layer.weight.dtype))


def _log_structural_config(model) -> None:
    for module in model:
        config = getattr(module, "get_config_dict", lambda: {})()
        logger.info("Cloned module %s: %s", type(module).__name__, config)
