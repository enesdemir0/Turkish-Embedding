"""FOCUS embedding-init strategy — Dobler & de Melo, 2023 (arXiv:2305.14481),
`pip install deepfocus` (GitHub `konstantinjdobler/focus`). See TURKIS~1.MD §1.2/§5.

FOCUS composes each target-vocab token's embedding as a sparsemax-weighted
combination of the teacher's overlapping-subword embeddings, where the weights come
from an auxiliary embedding space (fasttext) trained on a target-language corpus —
unlike WECHSEL, no bilingual dictionary is required, which is why TURKIS~1.MD flags
it as the most practical drop-in replacement for the paper's mean-composition
`Compose()` (`mean_composition.py`).

UNVERIFIED: the `FOCUS(...)` call below (import path, kwarg names) is a best-guess
based on general knowledge of the public `deepfocus` package, NOT confirmed by
reading its installed source — this project's working agreement forbids installing
heavy ML packages on this machine; all real execution happens on Colab. Before the
first real Colab run, read the installed signature directly
(`import inspect, deepfocus; inspect.signature(deepfocus.FOCUS)`) and patch this file
to match, the same "read installed source, patch, verify" method already used to fix
3 real bugs in this project (see SESSION_NOTES.md's "Bugs found" section) — do not
assume this guess is correct without doing that.

Unlike `mean_composition` (which returns None because transformer-cloner's native
clone already *is* mean-composition), FOCUS has no transformer-cloner equivalent, so
this is the first strategy that actually returns a real matrix for
`model_cloning/clone.py`'s `_maybe_override_embeddings` to inject.
"""

from __future__ import annotations

from src.embedding_init.base import EmbeddingInitStrategy
from src.embedding_init.registry import EMBEDDING_INIT_REGISTRY


@EMBEDDING_INIT_REGISTRY.register("focus_init")
class FocusEmbeddingInit(EmbeddingInitStrategy):
    """params:
        target_corpus_path: path to a plain-text corpus file (one document per
            line) used to fit FOCUS's auxiliary target-language fasttext
            embeddings — see src/data/focus_corpus.py for how exp002 builds this
            from the existing distillation corpus.
        language: target language code, e.g. "tr".
        fasttext_model_path: optional path to a pre-trained fasttext model, if
            FOCUS expects one instead of training from target_corpus_path.
    Param names are a best guess pending confirmation against the real deepfocus
    API on Colab — see module docstring.
    """

    def compose(self, teacher_model, target_tokenizer, teacher_tokenizer):
        # Lazy imports (matches model_cloning/clone.py's convention) so this module
        # stays importable/py_compile-able without deepfocus/torch/transformers
        # installed.
        from transformers import AutoModel, AutoTokenizer

        from deepfocus import FOCUS  # import path UNCONFIRMED — see module docstring

        source_model = AutoModel.from_pretrained(teacher_model)
        source_embeddings = source_model.get_input_embeddings().weight

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
