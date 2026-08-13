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
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


def _wordpiece_continuing_subword_prefix(tokenizer) -> str | None:
    """Returns the WordPiece continuation-piece prefix (e.g. "##" for BERT-family
    tokenizers like BERTurk) if the tokenizer's backend uses WordPiece, else None.
    Read from the tokenizer's own backend model config rather than hardcoding "##",
    so this stays correct for any WordPiece tokenizer regardless of chosen prefix.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    return getattr(backend.model, "continuing_subword_prefix", None) or None


def _fast_build_token_id_map(self, batch_size: int = 5000, verbose: bool = True) -> dict[int, list[int]]:
    """Drop-in replacement for TransformerCloner.build_token_id_map (confirmed by
    reading the installed source, v0.2.10) that fixes two real bugs:

    1. O(n^2) performance bug: the original does `self.target_tokenizer.vocab[vocab]`
       *inside* the per-token loop. For a fast HF tokenizer, `.vocab` rebuilds its
       entire vocabulary dict from the Rust backend on every access — so for a
       131K-token vocabulary, that's a 131K-entry dict rebuilt 131,073 times (once
       per token) instead of once. That's what turns a step that should take seconds
       into multiple hours. Fixed by caching target_vocab once, outside the loop.
    2. WordPiece continuation-prefix bug (found via exp007's BERTurk result, a real
       -11.5pt regression vs. the paper's own tokenizer): the original re-encodes
       each target vocab *key string* verbatim through the teacher's tokenizer to
       decide how to compose its embedding. For WordPiece vocabularies (BERTurk),
       continuation-piece keys are literally stored with a "##" prefix (e.g.
       "##lar") — re-encoding that string verbatim tokenizes the literal "#"
       characters instead of the actual subword "lar", producing a garbage
       composition for roughly half the vocabulary (WordPiece's continuation
       pieces). Since Turkish is heavily suffix-based, this poisoned a large,
       pervasive fraction of the embedding table — consistent with the uniform,
       across-the-board task degradation observed, not a narrow retrieval-specific
       tokenization-mismatch pattern. Fixed by stripping the continuation prefix
       (read from the tokenizer's own backend config, not hardcoded) before
       re-encoding, so continuation pieces compose from their actual subword text.
       No-op for non-WordPiece tokenizers (prefix is None).
    """
    target_vocab = self.target_tokenizer.vocab
    target_vocab_keys = list(target_vocab.keys())
    total_tokens = len(target_vocab_keys)
    continuing_prefix = _wordpiece_continuing_subword_prefix(self.target_tokenizer)

    if verbose:
        print(f"Building token ID map for {total_tokens} tokens...")
        if continuing_prefix:
            print(f"Detected WordPiece continuation prefix {continuing_prefix!r} — stripping before re-encoding")

    self.token_id_map = {}

    for batch_start in range(0, total_tokens, batch_size):
        batch_end = min(batch_start + batch_size, total_tokens)
        batch_tokens = target_vocab_keys[batch_start:batch_end]

        if continuing_prefix:
            encode_texts = [
                token[len(continuing_prefix):] if token.startswith(continuing_prefix) else token
                for token in batch_tokens
            ]
        else:
            encode_texts = batch_tokens

        batch_encoded = self.org_tokenizer(
            encode_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        for i, vocab_token in enumerate(batch_tokens):
            target_id = target_vocab[vocab_token]
            source_ids = batch_encoded["input_ids"][i][1:]
            self.token_id_map[target_id] = source_ids

        if verbose:
            print(f"Processed {batch_end}/{total_tokens} tokens")

    if verbose:
        print(f"Token ID map built with {len(self.token_id_map)} entries")

    return self.token_id_map


@contextmanager
def _patch_slow_vocab_lookup():
    from transformer_cloner.cloner import TransformerCloner

    original = TransformerCloner.build_token_id_map
    TransformerCloner.build_token_id_map = _fast_build_token_id_map
    try:
        yield
    finally:
        TransformerCloner.build_token_id_map = original


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
    with _patch_slow_vocab_lookup():
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
