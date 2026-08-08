"""From-scratch Turkish tokenizer construction — EMBEDD~3.MD §2. Unlike
`reuse_pretrained.py` (which just points at the paper's already-published
tokenizer), this strategy actually builds one: trains a Turkish-native
SentencePiece-BPE tokenizer on the Cosmos Turkish Corpus, frequency-selects its top
65,536 tokens, unions in frequency-selected multilingual tokens from
Wikipedia-40-langs (after pruning the teacher's vocab for redundancy against the
Turkish-native set), and targets the paper's 131,072-token final vocabulary.

Uses HF `tokenizers`' `SentencePieceBPETokenizer` (Metaspace pre-tokenization, i.e.
`▁`-marked BPE — a native implementation of "SentencePiece-BPE", confirmed
present transitively via transformers/sentence-transformers). Its output wraps
directly into `transformers.PreTrainedTokenizerFast(tokenizer_object=...)` ->
`.save_pretrained(output_dir)`, producing a real `AutoTokenizer.from_pretrained`-
loadable directory, exactly what `TokenizerSurgeryStrategy.build()` must return (see
base.py) and exactly what `model_cloning/clone.py` and both existing
`embedding_init` strategies already consume generically. This avoids a second BPE
code path (raw `sentencepiece` protobuf `.model` + a hand-rolled slow-tokenizer
wrapper).

ASSUMPTIONS (the paper is underspecified past this point — document, don't guess
silently, revisit if a later session finds the paper's real algorithm):

1. **§2.3's "prune the teacher's vocabulary for redundancy" is vague** — no
   precise algorithm is given. Interpretation used here: a teacher token is
   "redundant" if re-encoding its surface string with the from-scratch Turkish
   tokenizer collapses to exactly one token that is already in the Turkish-native
   65,536 set (i.e. a dedicated Turkish-native token already covers it exactly).
   Everything else is a `teacher_survivor`, preferred (but not required) when
   filling the multilingual slots in step 2 below.
2. **Splicing two independently-trained BPE vocabularies is a bigger gap than the
   paper spells out.** A BPE tokenizer's ability to actually produce a given vocab
   string at inference time depends on its *merge rule chain*, not just vocab
   membership — freely unioning vocab strings from two separately-trained
   tokenizers (Turkish-native from Cosmos, multilingual from Wiki-40-langs) does
   not by itself give you a working tokenizer. Resolution used here: walk each
   source tokenizer's own merges (in training order), keep a merge iff its product
   survives into the final vocab and both its inputs are already producible; any
   final-vocab string still unreachable after both merge passes is added via HF's
   `add_tokens` (`added_tokens`) mechanism instead — a first-class way to force an
   exact-string vocab entry to exist without deriving it through merges. The
   fraction of the final vocab that falls into `added_tokens` is logged; a very
   high fraction would mean this approximation is too lossy and a shared tokenizer
   trained on a mixed/oversampled corpus should be tried instead (flagged as
   future work, not fixed here).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from src.tokenizer_surgery.base import TokenizerSurgeryStrategy
from src.tokenizer_surgery.registry import TOKENIZER_SURGERY_REGISTRY

logger = logging.getLogger(__name__)

# Paper's exact target sizes (EMBEDD~3.MD §2.2/§2.5). Overridable via params, but
# doing so means the run is no longer the paper's exact tokenizer recipe.
DEFAULT_TURKISH_NATIVE_VOCAB_SIZE = 65536  # 2**16
DEFAULT_FINAL_VOCAB_SIZE = 131072  # 2**17
DEFAULT_TRAIN_VOCAB_SIZE = 98304  # headroom above the 65536 frequency-selection target

_SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]


def _train_bpe(corpus_path: str | Path, vocab_size: int, min_frequency: int):
    """Trains a from-scratch SentencePiece-style BPE tokenizer over corpus_path.
    Returns the trained tokenizers.implementations.SentencePieceBPETokenizer."""
    from tokenizers.implementations import SentencePieceBPETokenizer

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train(
        [str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=_SPECIAL_TOKENS,
    )
    return tokenizer


def _read_ordered_merges(tokenizer, tmp_dir: str | Path) -> list[tuple[str, str]]:
    """tokenizers' Tokenizer object doesn't expose its trained merge list directly
    as Python objects — save_model() writes it to merges.txt in training order,
    which we read back. Confirmed empirically (not guessed) against tokenizers
    0.22.2: line 0 is a "#version: ..." header, every subsequent line is
    "<left> <right>"."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _, merges_file = tokenizer.save_model(str(tmp_dir))
    with open(merges_file, encoding="utf-8") as f:
        lines = f.read().splitlines()[1:]
    return [tuple(line.split(" ", 1)) for line in lines if line.strip()]


def _base_alphabet(tokenizer) -> set[str]:
    """Single-character vocab entries (with or without the Metaspace '▁' word-
    boundary marker) — force-included regardless of frequency rank, since dropping
    any of these could make some input characters entirely unrepresentable."""
    vocab = tokenizer.get_vocab()
    alphabet = set()
    for token in vocab:
        bare = token.replace("▁", "")
        if len(bare) <= 1:
            alphabet.add(token)
    return alphabet


def _top_k_by_frequency(tokenizer, corpus_path: str | Path, k: int) -> list[str]:
    """Encodes corpus_path with tokenizer, tallies token-id frequency, returns the
    top-k vocab strings by frequency (always including the base alphabet, per
    _base_alphabet's docstring, even if that pushes the returned set slightly above
    k)."""
    inv_vocab = {idx: token for token, idx in tokenizer.get_vocab().items()}
    counts: Counter[int] = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts.update(tokenizer.encode(line).ids)

    ranked = [inv_vocab[token_id] for token_id, _ in counts.most_common()]
    selected = list(dict.fromkeys(ranked[:k]))  # preserve rank order, dedupe
    forced = _base_alphabet(tokenizer) | set(_SPECIAL_TOKENS)
    selected_set = set(selected) | forced
    return sorted(selected_set, key=lambda tok: (tok not in selected, ranked.index(tok) if tok in ranked else 0))


def _is_redundant(token_str: str, turkish_tokenizer, turkish_native_set: set[str]) -> bool:
    """See module docstring, assumption 1: a teacher token is "redundant" if the
    from-scratch Turkish tokenizer already re-encodes its surface string as exactly
    one token that is itself in the Turkish-native set."""
    encoded = turkish_tokenizer.encode(token_str)
    return len(encoded.tokens) == 1 and encoded.tokens[0] in turkish_native_set


def _select_multilingual(
    multilingual_ranked: list[str],
    teacher_survivors: set[str],
    turkish_native_set: set[str],
    remaining_slots: int,
) -> list[str]:
    """See module docstring, §2.4c: fills remaining_slots by preferring
    multilingual tokens that also survived the teacher-vocab pruning pass, then
    backfills from the rest of multilingual_ranked once that pool is exhausted.
    Always skips anything already in turkish_native_set (dedupe)."""
    candidates = [tok for tok in multilingual_ranked if tok not in turkish_native_set]

    preferred = [tok for tok in candidates if tok in teacher_survivors]
    fallback = [tok for tok in candidates if tok not in teacher_survivors]

    selected: list[str] = []
    seen: set[str] = set()
    for tok in preferred + fallback:
        if len(selected) >= remaining_slots:
            break
        if tok in seen:
            continue
        seen.add(tok)
        selected.append(tok)

    if len(selected) < remaining_slots:
        logger.warning(
            "_select_multilingual: only found %d/%d multilingual tokens to fill remaining slots",
            len(selected),
            remaining_slots,
        )
    return selected


def _splice_merges(
    sources: list[tuple[list[tuple[str, str]], set[str]]],
    final_vocab_strings: set[str],
) -> tuple[list[tuple[str, str]], set[str]]:
    """See module docstring, assumption 2. sources is an ordered list of
    (merges, produced_base) pairs — one per source tokenizer, each already trained
    independently — walked in order given. Returns (kept_merges, added_tokens),
    where added_tokens is whatever final_vocab_strings entry never became
    producible through either source's merge chain."""
    produced: set[str] = set()
    for _, base in sources:
        produced |= base

    kept_merges: list[tuple[str, str]] = []
    for merges, _ in sources:
        for a, b in merges:
            ab = a + b
            if ab in final_vocab_strings and ab not in produced and a in produced and b in produced:
                kept_merges.append((a, b))
                produced.add(ab)

    added_tokens = final_vocab_strings - produced
    return kept_merges, added_tokens


def _save_fast_tokenizer(
    final_vocab_strings: set[str],
    kept_merges: list[tuple[str, str]],
    added_tokens: set[str],
    output_dir: str | Path,
):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    ordered = sorted(final_vocab_strings, key=lambda tok: (tok not in _SPECIAL_TOKENS, tok))
    vocab = {tok: idx for idx, tok in enumerate(ordered)}

    bpe = models.BPE(vocab=vocab, merges=kept_merges, unk_token="<unk>")
    tokenizer = Tokenizer(bpe)
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    if added_tokens:
        fast_tokenizer.add_tokens(sorted(added_tokens))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fast_tokenizer.save_pretrained(str(output_dir))
    return output_dir


@TOKENIZER_SURGERY_REGISTRY.register("frequency_pruning")
class FrequencyPruningTokenizer(TokenizerSurgeryStrategy):
    """From-scratch Turkish tokenizer, EMBEDD~3.MD §2. params (all default to the
    paper's own values, so `params: {}` reproduces the paper's recipe):

        cosmos_corpus_repo: HF dataset repo for the Turkish-native training corpus
            (default: ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0).
        wiki40langs_repo: HF dataset repo for the multilingual frequency-analysis
            corpus (default: alibayram/wikipedia-40-langs — NOT the
            -with-embeddings variant, a different repo used elsewhere in this
            project for FOCUS).
        teacher_tokenizer_id: HF model id whose tokenizer vocab is pruned for
            redundancy in §2.3 (default: google/embeddinggemma-300m).
        turkish_native_vocab_size: size of the frequency-selected Turkish-native
            set (default 65536, i.e. 2**16, the paper's exact value).
        final_vocab_size: total target vocabulary size (default 131072, i.e.
            2**17, the paper's exact value).
        turkish_train_vocab_size / multilingual_train_vocab_size: headroom vocab
            sizes used for the two initial from-scratch BPE training passes,
            before frequency-based downselection (default 98304 each).
        corpus_subsample_fraction: optional float in (0, 1], same semantics as
            src/distillation/cosine.py's existing knob (fixed shuffle seed 42) —
            applied to BOTH corpora, for a cheap first pass before committing to
            the full-scale downloads/training. Left unset, uses full corpora.
        min_frequency: passed to the BPE trainer (default 2).
        corpus_output_dir: where extracted plain-text corpus files are written
            (default ./corpus_cache).
        output_dir: where the final built tokenizer is saved (default
            ./tokenizers/frequency_pruning) — this directory's path is what
            build() returns.
    """

    def build(self) -> str:
        from src.data.cosmos_corpus import extract_cosmos_corpus_file
        from src.data.wiki40langs_corpus import extract_multilingual_corpus_file

        cosmos_corpus_repo = self.params.get("cosmos_corpus_repo", "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0")
        wiki40langs_repo = self.params.get("wiki40langs_repo", "alibayram/wikipedia-40-langs")
        teacher_tokenizer_id = self.params.get("teacher_tokenizer_id", "google/embeddinggemma-300m")
        turkish_native_vocab_size = self.params.get("turkish_native_vocab_size", DEFAULT_TURKISH_NATIVE_VOCAB_SIZE)
        final_vocab_size = self.params.get("final_vocab_size", DEFAULT_FINAL_VOCAB_SIZE)
        turkish_train_vocab_size = self.params.get("turkish_train_vocab_size", DEFAULT_TRAIN_VOCAB_SIZE)
        multilingual_train_vocab_size = self.params.get("multilingual_train_vocab_size", DEFAULT_TRAIN_VOCAB_SIZE)
        corpus_subsample_fraction = self.params.get("corpus_subsample_fraction")
        min_frequency = self.params.get("min_frequency", 2)
        corpus_output_dir = Path(self.params.get("corpus_output_dir", "./corpus_cache"))
        output_dir = Path(self.params.get("output_dir", "./tokenizers/frequency_pruning"))

        logger.info("frequency_pruning: extracting Cosmos Turkish corpus (%s)", cosmos_corpus_repo)
        turkish_corpus_path = extract_cosmos_corpus_file(
            corpus_output_dir / "cosmos_turkish.txt",
            dataset_repo=cosmos_corpus_repo,
            subsample_fraction=corpus_subsample_fraction,
        )

        logger.info("frequency_pruning: extracting Wikipedia-40-langs corpus (%s)", wiki40langs_repo)
        multilingual_corpus_path = extract_multilingual_corpus_file(
            corpus_output_dir / "wiki40langs_multilingual.txt",
            dataset_repo=wiki40langs_repo,
            subsample_fraction=corpus_subsample_fraction,
        )

        logger.info("frequency_pruning: training Turkish-native BPE (vocab_size=%d)", turkish_train_vocab_size)
        turkish_tokenizer = _train_bpe(turkish_corpus_path, turkish_train_vocab_size, min_frequency)
        turkish_native_set = set(
            _top_k_by_frequency(turkish_tokenizer, turkish_corpus_path, turkish_native_vocab_size)
        )
        logger.info("frequency_pruning: Turkish-native set has %d tokens", len(turkish_native_set))

        logger.info("frequency_pruning: training multilingual BPE (vocab_size=%d)", multilingual_train_vocab_size)
        multilingual_tokenizer = _train_bpe(multilingual_corpus_path, multilingual_train_vocab_size, min_frequency)
        multilingual_ranked = _top_k_by_frequency(
            multilingual_tokenizer, multilingual_corpus_path, multilingual_train_vocab_size
        )

        logger.info("frequency_pruning: pruning teacher vocab (%s) for redundancy", teacher_tokenizer_id)
        from transformers import AutoTokenizer

        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_id)
        teacher_survivors = {
            tok
            for tok in teacher_tokenizer.get_vocab()
            if not _is_redundant(tok, turkish_tokenizer, turkish_native_set)
        }
        logger.info(
            "frequency_pruning: %d/%d teacher tokens survived redundancy pruning",
            len(teacher_survivors),
            len(teacher_tokenizer.get_vocab()),
        )

        remaining_slots = final_vocab_size - len(turkish_native_set)
        multilingual_selected = _select_multilingual(
            multilingual_ranked, teacher_survivors, turkish_native_set, remaining_slots
        )
        final_vocab_strings = turkish_native_set | set(multilingual_selected)

        if len(final_vocab_strings) > final_vocab_size:
            # Deterministic trim: drop the lowest-priority multilingual entries first.
            overflow = len(final_vocab_strings) - final_vocab_size
            trimmable = [tok for tok in reversed(multilingual_selected) if tok in final_vocab_strings][:overflow]
            final_vocab_strings -= set(trimmable)
        logger.info("frequency_pruning: final vocab has %d tokens (target %d)", len(final_vocab_strings), final_vocab_size)

        turkish_merges = _read_ordered_merges(turkish_tokenizer, corpus_output_dir / "_turkish_model")
        multilingual_merges = _read_ordered_merges(multilingual_tokenizer, corpus_output_dir / "_multilingual_model")

        base = set(_SPECIAL_TOKENS) | _base_alphabet(turkish_tokenizer) | _base_alphabet(multilingual_tokenizer)
        kept_merges, added_tokens = _splice_merges(
            [(turkish_merges, base), (multilingual_merges, set())],
            final_vocab_strings,
        )
        fallback_fraction = len(added_tokens) / max(1, len(final_vocab_strings))
        logger.info(
            "frequency_pruning: %d/%d final-vocab tokens (%.1f%%) fell into the added_tokens "
            "merge-splicing fallback (see module docstring, assumption 2)",
            len(added_tokens),
            len(final_vocab_strings),
            fallback_fraction * 100,
        )

        saved_dir = _save_fast_tokenizer(final_vocab_strings, kept_merges, added_tokens, output_dir)
        logger.info("frequency_pruning: saved final tokenizer to %s", saved_dir)
        return str(saved_dir)
