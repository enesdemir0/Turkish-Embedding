"""KOMBO-style compositional root+suffix embedding (exp009) — the "no fixed
subword lookup table" model_cloning strategy.

Every prior model_cloning strategy (`transformer_clone`, `native_pretrained`)
still starts from a fixed vocabulary: a static table mapping surface
tokens/subwords to embedding rows, however that table was built. This one
doesn't. Each word is split into root + suffix chain by a Turkish
morphological analyzer (`zeyrek`; see `src/data/turkish_morphology.py`), and
its embedding is *composed at runtime* from small root/suffix embedding
tables via a fixed combination operator, instead of being looked up in a
static table. Generalizes to unseen root-suffix combinations at inference —
something no fixed vocabulary, however well pruned, can do. Reference: KOMBO
(arXiv:2604.23948, Korean Jamo-composition analogue), TURKIS~1.MD section
2.2/5 item 5, SESSION_NOTES.md "Future work" #7.

Architecture, confirmed against the actually-installed `sentence_transformers`
5.6.1 source (not assumed/guessed):
  - `RootSuffixEmbedding` is module index 0 in `SentenceTransformer(modules=[...])`.
    Only module[0]'s `preprocess()` is ever called by the pipeline
    (`sentence_transformers.base.model`'s `preprocess()`, `self[0].preprocess(...)`)
    — every other module only sees `forward()`. So `RootSuffixEmbedding.preprocess()`
    does the word-splitting + root/suffix id lookup, and `forward()` composes
    the per-word vectors and writes them into `features["inputs_embeds"]`
    (never `input_ids` — this model has no subword vocabulary to tokenize
    into ids at all).
  - `Transformer.__init__` always adds `"inputs_embeds"` to
    `self.model_forward_params` unconditionally (in addition to whatever the
    wrapped HF model's own `forward()` signature exposes) — so the next
    module downstream (a standard `sentence_transformers.sentence_transformer.modules.Transformer`
    wrapping a plain HF encoder) accepts `inputs_embeds` with zero patching.
    `Transformer.forward()` defaults `features.get("modality", "text")`, so
    no `modality` key needs to be set explicitly as long as the wrapped HF
    model + its (formality-only, see `CompositionalRootSuffixStrategy.build`)
    tokenizer both resolve to the "text" modality, which they do for any
    plain encoder-only text model.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sentence_transformers.base.modules import InputModule

from src.model_cloning.base import ModelCloningStrategy
from src.model_cloning.registry import MODEL_CLONING_REGISTRY

logger = logging.getLogger(__name__)

UNK_ROOT_ID = 0
UNK_SUFFIX_ID = 0


class RootSuffixEmbedding(InputModule):
    """Composes a word's embedding at runtime as
    ``root_embedding + mean(suffix_embeddings)`` — order-invariant,
    parameter-light. Deliberately the simplest defensible operator for a
    first smoke test: getting the mechanism plumbed end-to-end correctly
    matters more than linguistic correctness on attempt #1. A word with zero
    suffixes degenerates to exactly its root embedding, no special-casing
    needed. A suffix-order-aware combinator (concat+MLP, small RNN/attention)
    is real future work once this plumbing is validated.

    Sequences are built at the WORD level (whitespace/punctuation split), not
    subword level — a deliberate, structural difference from every other
    student in this project. Unseen words/roots/suffixes map to id 0
    (<unk_root>/<unk_suffix>), never raise.
    """

    config_file_name = "root_suffix_embedding_config.json"
    config_keys = [
        "root_vocab_size",
        "suffix_vocab_size",
        "hidden_dim",
        "max_suffixes_per_word",
        "max_words_per_text",
        "roots",
        "suffixes",
    ]
    save_in_root = True  # index 0 in the modules list; matches Transformer/StaticEmbedding convention

    def __init__(
        self,
        root_vocab_size: int,
        suffix_vocab_size: int,
        hidden_dim: int,
        roots: dict[str, int] | None = None,
        suffixes: dict[str, int] | None = None,
        max_suffixes_per_word: int = 8,
        max_words_per_text: int = 128,
        analyzer: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.root_vocab_size = root_vocab_size
        self.suffix_vocab_size = suffix_vocab_size
        self.hidden_dim = hidden_dim
        self.max_suffixes_per_word = max_suffixes_per_word
        # Word-level equivalent of a tokenizer's max_seq_length. Without this cap a
        # long document (a full Wikipedia article in the distillation dataset, not a
        # short sentence) produces an arbitrarily long word sequence — hit for real on
        # Colab: a 12,143-word row crashed the tiny BERT body, whose
        # max_position_embeddings/token_type_ids buffer default to 512. Every other
        # student in this project truncates via its tokenizer's max_seq_length; this
        # is the equivalent guard for a student with no tokenizer at all.
        self.max_words_per_text = max_words_per_text
        self.roots = roots or {}
        self.suffixes = suffixes or {}

        self.root_embeddings = nn.Embedding(root_vocab_size, hidden_dim, padding_idx=UNK_ROOT_ID)
        self.suffix_embeddings = nn.Embedding(suffix_vocab_size, hidden_dim, padding_idx=UNK_SUFFIX_ID)

        self._analyzer = analyzer  # lazily created on first preprocess() call if still None

    @classmethod
    def from_vocab_files(
        cls,
        roots_path: str | Path,
        suffixes_path: str | Path,
        hidden_dim: int,
        max_suffixes_per_word: int = 8,
        max_words_per_text: int = 128,
    ) -> "RootSuffixEmbedding":
        """Builds a fresh (randomly-initialized) module from the roots.json/
        suffixes.json produced by src/data/turkish_morphology.py's
        build_root_suffix_vocab(). Only used at first construction — a
        reloaded module goes through Module.load() -> cls(**config), which
        already has roots/suffixes inline in config.json (see config_keys)."""
        roots = json.loads(Path(roots_path).read_text(encoding="utf-8"))
        suffixes = json.loads(Path(suffixes_path).read_text(encoding="utf-8"))
        return cls(
            root_vocab_size=len(roots),
            suffix_vocab_size=len(suffixes),
            hidden_dim=hidden_dim,
            roots=roots,
            suffixes=suffixes,
            max_suffixes_per_word=max_suffixes_per_word,
            max_words_per_text=max_words_per_text,
        )

    def _get_analyzer(self) -> Any:
        if self._analyzer is None:
            from src.data.turkish_morphology import get_analyzer

            self._analyzer = get_analyzer()
        return self._analyzer

    def preprocess(
        self,
        inputs: list[Any],
        prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # src/data/turkish_morphology.py's segment_word() is the single source
        # of truth for root/suffix segmentation elsewhere in this project
        # (vocab building); duplicated here in id-lookup-only form (no
        # OOV-fallback bookkeeping needed at inference time) to avoid a
        # data->model_cloning import direction the project's stage layering
        # doesn't otherwise have.
        from src.data.turkish_morphology import segment_word

        texts = list(inputs)
        if prompt:
            texts = [prompt + text for text in texts]

        analyzer = self._get_analyzer()
        batch_root_ids: list[list[int]] = []
        batch_suffix_ids: list[list[list[int]]] = []

        for text in texts:
            words = text.split()[: self.max_words_per_text]
            root_ids: list[int] = []
            suffix_ids: list[list[int]] = []
            for word in words:
                root, word_suffixes, _is_fallback = segment_word(word, analyzer)
                root_ids.append(self.roots.get(root, UNK_ROOT_ID))
                truncated = word_suffixes[: self.max_suffixes_per_word]
                suffix_ids.append([self.suffixes.get(s, UNK_SUFFIX_ID) for s in truncated])
            batch_root_ids.append(root_ids)
            batch_suffix_ids.append(suffix_ids)

        max_len = max((len(r) for r in batch_root_ids), default=0)
        max_len = max(max_len, 1)  # never emit a zero-length sequence dimension

        root_ids_tensor = torch.zeros((len(texts), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)
        suffix_ids_tensor = torch.zeros((len(texts), max_len, self.max_suffixes_per_word), dtype=torch.long)
        suffix_mask = torch.zeros((len(texts), max_len, self.max_suffixes_per_word), dtype=torch.float32)

        for i, (root_ids, suffix_ids) in enumerate(zip(batch_root_ids, batch_suffix_ids)):
            for j, (root_id, word_suffix_ids) in enumerate(zip(root_ids, suffix_ids)):
                root_ids_tensor[i, j] = root_id
                attention_mask[i, j] = 1
                for k, suffix_id in enumerate(word_suffix_ids):
                    suffix_ids_tensor[i, j, k] = suffix_id
                    suffix_mask[i, j, k] = 1.0

        return {
            "root_ids": root_ids_tensor,
            "suffix_ids": suffix_ids_tensor,
            "suffix_mask": suffix_mask,
            "attention_mask": attention_mask,
        }

    def forward(self, features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        root_vecs = self.root_embeddings(features["root_ids"])  # [B, L, H]
        suffix_vecs = self.suffix_embeddings(features["suffix_ids"])  # [B, L, S, H]
        mask = features["suffix_mask"].unsqueeze(-1)  # [B, L, S, 1]
        suffix_sum = (suffix_vecs * mask).sum(dim=2)  # [B, L, H]
        suffix_count = mask.sum(dim=2).clamp(min=1.0)  # [B, L, 1]
        composed = root_vecs + suffix_sum / suffix_count  # [B, L, H]

        features["inputs_embeds"] = composed
        features.pop("root_ids", None)
        features.pop("suffix_ids", None)
        features.pop("suffix_mask", None)
        return features

    def get_embedding_dimension(self) -> int:
        return self.hidden_dim

    def save(self, output_path: str, *args: Any, safe_serialization: bool = True, **kwargs: Any) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs: Any) -> "RootSuffixEmbedding":
        # The base Module.load() only does cls(**load_config()) — it never
        # actually loads the saved torch weights back in (confirmed by
        # reading sentence_transformers.base.modules.module.Module.load's
        # real source: config -> cls(**config), nothing else). Without this
        # override, a "round-tripped" module silently reverts to fresh random
        # embedding weights — exactly the load-bearing risk flagged in the
        # plan, caught here by an explicit save->load numeric-equality test
        # before this ever touches distil-trainer.
        # The real caller (SentenceTransformer's own module loader) passes extra
        # kwargs (e.g. trust_remote_code) that neither load_config nor
        # load_torch_weights accept — filter down to each method's actual
        # accepted parameter names rather than assuming kwargs is already clean.
        _CONFIG_KWARGS = {"subfolder", "config_filename", "token", "cache_folder", "revision", "local_files_only"}
        _WEIGHTS_KWARGS = {"subfolder", "token", "cache_folder", "revision", "local_files_only"}
        config = cls.load_config(
            model_name_or_path, **{k: v for k, v in kwargs.items() if k in _CONFIG_KWARGS}
        )
        instance = cls(**config)
        cls.load_torch_weights(
            model_name_or_path,
            model=instance,
            **{k: v for k, v in kwargs.items() if k in _WEIGHTS_KWARGS},
        )
        return instance


@MODEL_CLONING_REGISTRY.register("compositional_root_suffix")
class CompositionalRootSuffixStrategy(ModelCloningStrategy):
    """params:
        roots_path, suffixes_path: paths to roots.json/suffixes.json built
            beforehand by src/data/turkish_morphology.py::build_root_suffix_vocab
            (required — no auto-build here; vocab-building is a separate,
            explicit pre-step, since no pipeline stage owns "build a runtime
            vocab a model_cloning strategy consumes" — see exp009 plan).
        hidden_dim: composed embedding dim AND the tiny transformer body's
            hidden_size (must match, since RootSuffixEmbedding's output feeds
            straight into the body as inputs_embeds). Default 64.
        max_suffixes_per_word: pad/truncate length for a word's suffix chain
            (default 8).
        max_words_per_text: word-level equivalent of a tokenizer's
            max_seq_length — truncates each text to this many words before
            composition (default 128). Without this, a long document (a full
            Wikipedia article in the distillation dataset, not a short
            sentence) produces an arbitrarily long word sequence, which
            crashes the tiny BERT body's fixed-size position-embedding/
            token_type_ids buffers once a sequence exceeds
            max_position_embeddings. Kept in sync with tiny_transformer_config's
            implicit BertConfig default (max_position_embeddings=512) below —
            raise both together if you need longer sequences.
        tiny_transformer_config: dict of BertConfig kwargs (num_hidden_layers,
            num_attention_heads, intermediate_size) for a randomly-initialized
            smoke-test transformer body — deliberately NOT a pretrained
            backbone (see module docstring above: a pretrained body would
            confound "is the composition mechanism plumbed correctly" with
            "does a pretrained body tolerate an unfamiliar input
            representation", the same ambiguity that sank exp006/exp007).
        tokenizer_name_or_path: HF id/local path whose tokenizer files satisfy
            sentence_transformers' Transformer module's AutoProcessor.from_pretrained
            call — a formality only, never semantically used since input_ids
            is never populated (RootSuffixEmbedding only ever emits
            inputs_embeds). Default dbmdz/bert-base-turkish-128k-cased
            (public, small, already used elsewhere in this project).
        pooling_mode: passed to Pooling (default "mean").
    """

    requires_tokenizer_surgery = False
    requires_embedding_init = False

    def build(self, *, target_tokenizer: str | None, embedding_init_strategy: Any | None):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
        from transformers import AutoTokenizer, BertConfig, BertModel

        roots_path = self.params.get("roots_path")
        suffixes_path = self.params.get("suffixes_path")
        if not roots_path or not suffixes_path:
            raise ValueError(
                "compositional_root_suffix requires 'roots_path' and 'suffixes_path' "
                "params, built beforehand by "
                "src/data/turkish_morphology.py::build_root_suffix_vocab"
            )

        hidden_dim = self.params.get("hidden_dim", 64)
        max_suffixes_per_word = self.params.get("max_suffixes_per_word", 8)
        max_words_per_text = self.params.get("max_words_per_text", 128)
        tiny_transformer_config = self.params.get("tiny_transformer_config", {})
        tokenizer_name_or_path = self.params.get(
            "tokenizer_name_or_path", "dbmdz/bert-base-turkish-128k-cased"
        )
        pooling_mode = self.params.get("pooling_mode", "mean")

        front_end = RootSuffixEmbedding.from_vocab_files(
            roots_path,
            suffixes_path,
            hidden_dim=hidden_dim,
            max_suffixes_per_word=max_suffixes_per_word,
            max_words_per_text=max_words_per_text,
        )

        num_attention_heads = tiny_transformer_config.get("num_attention_heads", 2)
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"tiny_transformer_config.num_attention_heads ({num_attention_heads})"
            )
        # max_position_embeddings must cover max_words_per_text (RootSuffixEmbedding's
        # own truncation cap) — BERT's default (512) already does for the default
        # max_words_per_text=128, but keep them explicitly tied so a larger
        # max_words_per_text can never silently exceed the body's position buffer
        # (the exact crash this whole guard exists to prevent).
        max_position_embeddings = max(512, max_words_per_text)
        bert_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=tiny_transformer_config.get("num_hidden_layers", 2),
            num_attention_heads=num_attention_heads,
            intermediate_size=tiny_transformer_config.get("intermediate_size", 128),
            max_position_embeddings=max_position_embeddings,
            vocab_size=8,  # never used — RootSuffixEmbedding always supplies inputs_embeds directly
        )
        tiny_body = BertModel(bert_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tiny_body.save_pretrained(tmp_dir)
            AutoTokenizer.from_pretrained(tokenizer_name_or_path).save_pretrained(tmp_dir)
            transformer = Transformer(tmp_dir)

        pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode=pooling_mode)
        model = SentenceTransformer(modules=[front_end, transformer, pooling])

        logger.info(
            "Built compositional_root_suffix student: root_vocab=%d suffix_vocab=%d "
            "hidden_dim=%d tiny_transformer_config=%s",
            front_end.root_vocab_size,
            front_end.suffix_vocab_size,
            hidden_dim,
            tiny_transformer_config,
        )
        return model
