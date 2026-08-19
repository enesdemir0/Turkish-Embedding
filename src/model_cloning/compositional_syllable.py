"""Syllable-based compositional embedding — a sibling to exp009's
morphology-based `compositional_root_suffix.py`, using a different Turkish
linguistic unit: syllables instead of root+suffix chains.

Same overall "no fixed subword vocabulary" idea as exp009 (see that module's
docstring): each word's embedding is *composed at runtime* from a small
syllable embedding table, instead of being looked up in a static table.
What's different: segmentation uses `src/data/turkish_syllable.py`'s
deterministic, rule-based `syllabify_word()` instead of `zeyrek`'s
dictionary-lookup morphological analyzer. This trades away roots' inherent
semantic anchoring (a syllable carries no meaning on its own, unlike a
morphological root) for speed and completeness (syllabification is pure
string logic — no OOV concept, no slow per-word analyzer call, no ~10.5%
fallback rate the way exp009g's root vocab had). Whether that tradeoff nets
out ahead of or behind exp009g's 0.4065 is a genuinely open, untested
question — not assumed to be better OR worse going in.

Composition operator: **mean only**, deliberately — `SyllableEmbedding`
composes a word as `mean(syllable_embeddings)`, the same "simplest
defensible operator first" discipline exp009 used for its own first smoke
test, doubly justified here after exp009l's real lesson: a fancier,
order-aware operator (`kombo_fold`) applied on top of a from-scratch,
randomly-initialized composition made things *worse* on a matched compute
budget, not better. No order-aware option is implemented here; if this
simpler mean-based version ever beats exp009g's 0.4065 by a real, single-
variable comparison, an order-aware follow-up becomes worth considering —
not before.
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

UNK_SYLLABLE_ID = 0


class SyllableEmbedding(InputModule):
    """Composes a word's embedding at runtime as
    ``mean(syllable_embeddings)`` — order-invariant, parameter-light, the
    same "simplest possible operator first" choice exp009's
    `RootSuffixEmbedding` made for its own first smoke test. A word with no
    syllables (empty string) never occurs in practice (`syllabify_word()`
    always returns at least one syllable for any non-empty word), but the
    mask-based mean handles it gracefully (falls back to an all-zero-mask,
    zero vector) regardless.

    Sequences are built at the WORD level (whitespace/punctuation split),
    same as `RootSuffixEmbedding`. Unseen syllables map to id 0
    (<unk_syllable>) — expected to be rare, since syllabification has no
    OOV concept the way a dictionary-based morphological analyzer does;
    id 0 only fires for a syllable that never appeared in the vocab-building
    corpus at all.
    """

    config_file_name = "syllable_embedding_config.json"
    config_keys = [
        "syllable_vocab_size",
        "hidden_dim",
        "max_syllables_per_word",
        "max_words_per_text",
        "syllables",
    ]
    save_in_root = True  # index 0 in the modules list; matches Transformer/StaticEmbedding convention

    def __init__(
        self,
        syllable_vocab_size: int,
        hidden_dim: int,
        syllables: dict[str, int] | None = None,
        max_syllables_per_word: int = 6,
        max_words_per_text: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.syllable_vocab_size = syllable_vocab_size
        self.hidden_dim = hidden_dim
        self.max_syllables_per_word = max_syllables_per_word
        # Word-level equivalent of a tokenizer's max_seq_length — same guard as
        # RootSuffixEmbedding, same real crash it prevents (a long document blowing
        # past the tiny BERT body's max_position_embeddings). See that class's
        # docstring for the full story (bug #1, SESSION_NOTES_EXP009.md).
        self.max_words_per_text = max_words_per_text
        self.syllables = syllables or {}

        self.syllable_embeddings = nn.Embedding(
            syllable_vocab_size, hidden_dim, padding_idx=UNK_SYLLABLE_ID
        )

    @classmethod
    def from_vocab_file(
        cls,
        syllables_path: str | Path,
        hidden_dim: int,
        max_syllables_per_word: int = 6,
        max_words_per_text: int = 128,
    ) -> "SyllableEmbedding":
        """Builds a fresh (randomly-initialized) module from the
        syllables.json produced by
        src/data/turkish_syllable.py::build_syllable_vocab(). Only used at
        first construction — a reloaded module goes through
        Module.load() -> cls(**config), which already has `syllables`
        inline in config.json (see config_keys)."""
        syllables = json.loads(Path(syllables_path).read_text(encoding="utf-8"))
        return cls(
            syllable_vocab_size=len(syllables),
            hidden_dim=hidden_dim,
            syllables=syllables,
            max_syllables_per_word=max_syllables_per_word,
            max_words_per_text=max_words_per_text,
        )

    def preprocess(
        self,
        inputs: list[Any],
        prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        from src.data.turkish_syllable import syllabify_word

        texts = list(inputs)
        if prompt:
            texts = [prompt + text for text in texts]

        batch_syllable_ids: list[list[list[int]]] = []
        for text in texts:
            words = text.split()[: self.max_words_per_text]
            syllable_ids: list[list[int]] = []
            for word in words:
                word_syllables = syllabify_word(word)[: self.max_syllables_per_word]
                syllable_ids.append([self.syllables.get(s, UNK_SYLLABLE_ID) for s in word_syllables])
            batch_syllable_ids.append(syllable_ids)

        max_len = max((len(w) for w in batch_syllable_ids), default=0)
        max_len = max(max_len, 1)  # never emit a zero-length sequence dimension

        syllable_ids_tensor = torch.zeros(
            (len(texts), max_len, self.max_syllables_per_word), dtype=torch.long
        )
        syllable_mask = torch.zeros(
            (len(texts), max_len, self.max_syllables_per_word), dtype=torch.float32
        )
        attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)

        for i, word_syllable_ids in enumerate(batch_syllable_ids):
            for j, syl_ids in enumerate(word_syllable_ids):
                attention_mask[i, j] = 1
                for k, syl_id in enumerate(syl_ids):
                    syllable_ids_tensor[i, j, k] = syl_id
                    syllable_mask[i, j, k] = 1.0

        return {
            "syllable_ids": syllable_ids_tensor,
            "syllable_mask": syllable_mask,
            "attention_mask": attention_mask,
        }

    def forward(self, features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        syllable_vecs = self.syllable_embeddings(features["syllable_ids"])  # [B, L, S, H]
        # Same bf16-dtype-cast discipline as RootSuffixEmbedding.forward() — see that
        # class's own comment (SESSION_NOTES_EXP009.md bug #2) for why this cast is
        # required, not optional, once the trainer casts the model to bf16.
        mask = features["syllable_mask"].unsqueeze(-1).to(syllable_vecs.dtype)  # [B, L, S, 1]
        syllable_sum = (syllable_vecs * mask).sum(dim=2)  # [B, L, H]
        syllable_count = mask.sum(dim=2).clamp(min=1.0)  # [B, L, 1]
        composed = syllable_sum / syllable_count  # [B, L, H]

        features["inputs_embeds"] = composed
        features.pop("syllable_ids", None)
        features.pop("syllable_mask", None)
        return features

    def get_embedding_dimension(self) -> int:
        return self.hidden_dim

    def save(self, output_path: str, *args: Any, safe_serialization: bool = True, **kwargs: Any) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs: Any) -> "SyllableEmbedding":
        # Same real bug the base Module.load() has for RootSuffixEmbedding — see that
        # class's load() override for the full explanation (config -> cls(**config)
        # alone never reloads saved torch weights).
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


@MODEL_CLONING_REGISTRY.register("compositional_syllable")
class CompositionalSyllableStrategy(ModelCloningStrategy):
    """params:
        syllables_path: path to syllables.json built beforehand by
            src/data/turkish_syllable.py::build_syllable_vocab (required —
            no auto-build here, same explicit-pre-step discipline as
            compositional_root_suffix's roots_path/suffixes_path).
        hidden_dim: composed embedding dim AND the tiny transformer body's
            hidden_size (must match). Default 64.
        max_syllables_per_word: pad/truncate length for a word's syllable
            chain (default 6 — Turkish words rarely exceed ~5-6 syllables
            even with heavy suffixation, shorter than
            compositional_root_suffix's default max_suffixes_per_word=8
            since syllables are a finer-grained unit than whole suffixes).
        max_words_per_text: word-level equivalent of a tokenizer's
            max_seq_length (default 128) — same overflow guard as
            compositional_root_suffix, same reasoning (SESSION_NOTES_EXP009.md
            bug #1).
        tiny_transformer_config: dict of BertConfig kwargs
            (num_hidden_layers, num_attention_heads, intermediate_size) for
            a randomly-initialized smoke-test transformer body — same
            "deliberately not pretrained" reasoning as
            compositional_root_suffix.
        tokenizer_name_or_path: HF id/local path whose tokenizer files
            satisfy sentence_transformers' Transformer module's
            AutoProcessor.from_pretrained call — a formality only, never
            semantically used (SyllableEmbedding only ever emits
            inputs_embeds, same as RootSuffixEmbedding). Default
            dbmdz/bert-base-turkish-128k-cased.
        pooling_mode: passed to Pooling (default "mean").
    """

    requires_tokenizer_surgery = False
    requires_embedding_init = False

    def build(self, *, target_tokenizer: str | None, embedding_init_strategy: Any | None):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
        from transformers import AutoTokenizer, BertConfig, BertModel

        syllables_path = self.params.get("syllables_path")
        if not syllables_path:
            raise ValueError(
                "compositional_syllable requires a 'syllables_path' param, built "
                "beforehand by src/data/turkish_syllable.py::build_syllable_vocab"
            )

        hidden_dim = self.params.get("hidden_dim", 64)
        max_syllables_per_word = self.params.get("max_syllables_per_word", 6)
        max_words_per_text = self.params.get("max_words_per_text", 128)
        tiny_transformer_config = self.params.get("tiny_transformer_config", {})
        tokenizer_name_or_path = self.params.get(
            "tokenizer_name_or_path", "dbmdz/bert-base-turkish-128k-cased"
        )
        pooling_mode = self.params.get("pooling_mode", "mean")

        front_end = SyllableEmbedding.from_vocab_file(
            syllables_path,
            hidden_dim=hidden_dim,
            max_syllables_per_word=max_syllables_per_word,
            max_words_per_text=max_words_per_text,
        )

        num_attention_heads = tiny_transformer_config.get("num_attention_heads", 2)
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"tiny_transformer_config.num_attention_heads ({num_attention_heads})"
            )
        max_position_embeddings = max(512, max_words_per_text)
        bert_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=tiny_transformer_config.get("num_hidden_layers", 2),
            num_attention_heads=num_attention_heads,
            intermediate_size=tiny_transformer_config.get("intermediate_size", 128),
            max_position_embeddings=max_position_embeddings,
            vocab_size=8,  # never used — SyllableEmbedding always supplies inputs_embeds directly
        )
        tiny_body = BertModel(bert_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tiny_body.save_pretrained(tmp_dir)
            AutoTokenizer.from_pretrained(tokenizer_name_or_path).save_pretrained(tmp_dir)
            transformer = Transformer(tmp_dir)

        pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode=pooling_mode)
        model = SentenceTransformer(modules=[front_end, transformer, pooling])

        logger.info(
            "Built compositional_syllable student: syllable_vocab=%d hidden_dim=%d "
            "tiny_transformer_config=%s",
            front_end.syllable_vocab_size,
            hidden_dim,
            tiny_transformer_config,
        )
        return model
