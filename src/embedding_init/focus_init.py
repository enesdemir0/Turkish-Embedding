"""FOCUS embedding-init strategy — Dobler & de Melo, 2023 (arXiv:2305.14481),
`pip install deepfocus` (GitHub `konstantinjdobler/focus`). See TURKIS~1.MD §1.2/§5.

FOCUS composes each target-vocab token's embedding as a sparsemax-weighted
combination of the teacher's overlapping-subword embeddings, where the weights come
from an auxiliary embedding space (fasttext) trained on a target-language corpus —
unlike WECHSEL, no bilingual dictionary is required, which is why TURKIS~1.MD flags
it as the most practical drop-in replacement for the paper's mean-composition
`Compose()` (`mean_composition.py`).

CONFIRMED against a real Colab traceback (not just guessed): `deepfocus.focus.FOCUS`'s
real signature is `FOCUS(target_tokenizer, source_tokenizer, source_embeddings,
auxiliary_embedding_mode, target_training_data_path, fasttext_model_path,
language_identifier, fasttext_model_epochs, fasttext_model_dim,
fasttext_model_min_count, exact_match_all, match_symbols, fuzzy_match_all,
extend_tokenizer, processes, seed, device, verbosity)` — our guessed kwarg names
below (`source_embeddings`, `source_tokenizer`, `target_tokenizer`,
`target_training_data_path`, `language_identifier`) all match it.

Unlike `mean_composition` (which returns None because transformer-cloner's native
clone already *is* mean-composition), FOCUS has no transformer-cloner equivalent, so
this is the first strategy that actually returns a real matrix for
`model_cloning/clone.py`'s `_maybe_override_embeddings` to inject.

Also observed on a real Colab run (not guessed): deepfocus caches its tokenized
corpus under ~/.cache/deepfocus/data/, keyed by filename rather than content, which
reused a stale/empty cache entry across retries and produced a `SchemaInferenceError`
even after the underlying corpus file was fixed. `_clear_stale_deepfocus_tokenization_cache`
below clears that cache for the current corpus filename before every compose() call.

A third real bug found this way: the teacher tokenizer (google/embeddinggemma-300m)
reports `len(teacher_tokenizer) == 262145`, but the teacher model's embedding matrix
(`get_input_embeddings().weight`) only has 262144 rows — one token id (262144) has no
corresponding embedding row. `transformer_cloner` already tolerates this silently
(confirmed by the earlier "Error mapping token ...: index 262144 is out of bounds for
dimension 0 with size 262144 / Warning: 1 tokens could not be mapped" log line from
model_cloning/clone.py's cloning step), but deepfocus's FOCUS() has no such tolerance
and crashes with an unhandled IndexError at that same index/size pair. `_pad_source_embeddings_to_tokenizer_size`
below pads the teacher embedding matrix with one extra (mean-vector) row so that id
becomes valid, before it's ever handed to FOCUS.
"""

from __future__ import annotations

from pathlib import Path

from src.embedding_init.base import EmbeddingInitStrategy
from src.embedding_init.registry import EMBEDDING_INIT_REGISTRY


def _clear_stale_deepfocus_tokenization_cache(corpus_path: Path) -> None:
    """deepfocus caches its tokenized version of a corpus file under
    ~/.cache/deepfocus/data/<filename>_tokenized_<hash>.txt, keyed off the file
    PATH rather than its content (observed empirically: re-running against a
    freshly-regenerated corpus file at the same path reused an empty cached
    tokenization left over from an earlier failed attempt, producing a confusing
    downstream `SchemaInferenceError` instead of re-tokenizing). Clearing any
    cache entries for this exact filename before every compose() call avoids that
    trap — a slightly wasted re-tokenization is a much smaller cost than a
    silently-stale cache poisoning the run.
    """
    cache_dir = Path.home() / ".cache" / "deepfocus" / "data"
    if not cache_dir.exists():
        return
    for stale_file in cache_dir.glob(f"{corpus_path.name}_tokenized_*"):
        stale_file.unlink()


def _pad_source_embeddings_to_tokenizer_size(source_embeddings, tokenizer_length: int):
    """See module docstring, 3rd bug: some teacher tokenizers report more tokens
    than the embedding matrix has rows. FOCUS indexes source_embeddings by raw
    token id with no bounds check, so any id in that gap crashes it. Pad with the
    mean embedding row (the conventional default for a newly-added token) rather
    than zeros, so the fabricated row isn't an out-of-distribution outlier in the
    embedding space it's about to be composed from."""
    import torch

    current_size = source_embeddings.shape[0]
    if current_size >= tokenizer_length:
        return source_embeddings

    missing_rows = tokenizer_length - current_size
    pad = source_embeddings.mean(dim=0, keepdim=True).repeat(missing_rows, 1)
    return torch.cat([source_embeddings, pad], dim=0)


@EMBEDDING_INIT_REGISTRY.register("focus_init")
class FocusEmbeddingInit(EmbeddingInitStrategy):
    """params:
        target_corpus_path: path to a plain-text corpus file (one document per
            line) used to fit FOCUS's auxiliary target-language fasttext
            embeddings. If this file doesn't exist yet, compose() builds it
            automatically (see src/data/focus_corpus.py) from the existing
            distillation corpus (alibayram/wikipedia-40-langs-with-embeddings) —
            no separate manual step required before running the pipeline.
        language: target language code passed to FOCUS itself, e.g. "tr".
        corpus_language: language column value used to filter the corpus when
            auto-building target_corpus_path (default "tur", matching the
            dataset's `lang` column — see teacher_embeddings.py). Only used if
            target_corpus_path doesn't already exist.
        fasttext_model_path: optional path to a pre-trained fasttext model, if
            FOCUS expects one instead of training from target_corpus_path.
    Param names are a best guess pending confirmation against the real deepfocus
    API on Colab — see module docstring.
    """

    def compose(self, teacher_model, target_tokenizer, teacher_tokenizer):
        # Lazy imports (matches model_cloning/clone.py's convention) so this module
        # stays importable/py_compile-able without deepfocus/torch/transformers
        # installed.
        from pathlib import Path

        from transformers import AutoModel, AutoTokenizer

        from deepfocus import FOCUS  # import path UNCONFIRMED — see module docstring

        corpus_path = Path(self.params["target_corpus_path"])
        # Treat a missing OR empty file as "needs (re)building" — an empty file
        # left over from an earlier interrupted/failed attempt would otherwise be
        # silently accepted here and only surface as a confusing error deep inside
        # deepfocus's own tokenization step.
        if not corpus_path.exists() or corpus_path.stat().st_size == 0:
            # Build it on the fly rather than requiring a separate manual step
            # before the pipeline runs — reuses the same distillation corpus
            # (alibayram/wikipedia-40-langs-with-embeddings) filtered to the
            # target language.
            from src.data.focus_corpus import extract_turkish_corpus_file

            extract_turkish_corpus_file(corpus_path, language=self.params.get("corpus_language", "tur"))

        _clear_stale_deepfocus_tokenization_cache(corpus_path)

        source_model = AutoModel.from_pretrained(teacher_model)
        source_embeddings = source_model.get_input_embeddings().weight
        source_embeddings = _pad_source_embeddings_to_tokenizer_size(source_embeddings, len(teacher_tokenizer))

        target_tok = AutoTokenizer.from_pretrained(target_tokenizer)

        target_embeddings = FOCUS(
            source_embeddings=source_embeddings,
            source_tokenizer=teacher_tokenizer,
            target_tokenizer=target_tok,
            target_training_data_path=self.params["target_corpus_path"],  # UNCONFIRMED kwarg name
            language_identifier=self.params.get("language"),  # UNCONFIRMED kwarg name
        )

        assert target_embeddings.shape[0] == len(target_tok), (
            f"FOCUS returned {target_embeddings.shape[0]} rows, target tokenizer "
            f"has {len(target_tok)} tokens"
        )
        return target_embeddings
