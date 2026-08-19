"""Builds syllable vocabularies for a new compositional embedding strategy
(`src/model_cloning/compositional_syllable.py`) — a sibling to exp009's
morphology-based `compositional_root_suffix.py`, using a different linguistic
unit: **syllables** instead of morphological root+suffix chains.

Why this exists as a separate idea, not another exp009 variant: exp009's
`zeyrek` morphological analyzer is slow (a real, repeatedly-hit bottleneck —
see SESSION_NOTES_EXP009.md) and imperfect (a real ~10.5% OOV-fallback rate
even at 60k vocab, since it's a dictionary-lookup-based analyzer that can
miss rare/novel words entirely). Turkish syllable structure, by contrast, is
governed by simple, highly regular rules (a direct consequence of Turkish
vowel harmony) — syllabification can be done with pure deterministic string
logic, no ML/dictionary lookup, no OOV concept at all (every word, however
rare or novel, syllabifies), and is orders of magnitude faster to compute.
The tradeoff, made explicit up front: a syllable carries no inherent
semantic meaning on its own (unlike a morphological root), so this is a
genuinely different bet, not a strictly-easier version of the same idea —
see the model_cloning strategy's own docstring for the expected-difficulty
discussion.

Every raw token is first normalized (Turkish-aware lowercase + strip any
non-Turkish-letter character — see `_normalize_word()`) before
syllabification. **Added after a real bug, not designed in from the start**:
without this, a first real Colab run produced 727,154 distinct "syllables"
from a corpus with only 4,042,857 distinct words — nearly word-scale, not
the small, reusable inventory Turkish's regular phonology should produce.
Cause: raw corpus tokens carry punctuation/casing/digits ("kitap," vs
"kitap." vs "Kitap" vs "kitap123"), each producing a distinct, non-reusable
syllable string, and Cosmos (a large, noisy, web-scraped corpus) has enough
long-tail junk like this to nearly double the distinct-string count instead
of collapsing into Turkish's actually-small set of real syllable shapes.
See `_normalize_word()`'s own docstring for the full story.

Turkish syllabification rule implemented here (standard, TDK-consistent),
applied to the normalized string: every syllable contains exactly one
vowel. For a run of N consonants between two vowels: if N <= 1, the whole
run joins the *following* syllable; if N >= 2, all but the last consonant
stay with the *preceding* syllable, and the last consonant joins the
following syllable (e.g. "ıspanak" -> "ıs-pa-nak": the 2-consonant "sp"
cluster splits as "s" staying with "ıs", "p" starting "pa"). Leading
consonants before the first vowel join the first syllable; trailing
consonants after the last vowel join the last syllable. A token that's all
non-letters after normalization (pure digits/punctuation/foreign script) is
treated as a single "syllable" (the original raw token), same never-raise
philosophy as `turkish_morphology.py`'s segment_word().
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

UNK_SYLLABLE = "<unk_syllable>"

TURKISH_VOWELS = set("aeıioöuü")
TURKISH_LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyz")

# Turkish-aware case folding — Python's own str.lower() gets this wrong for
# Turkish's dotted/dotless I distinction: "İ".lower() produces "i̇" (with a
# stray combining dot, not plain "i"), and "I".lower() produces "i" (should
# be the dotless "ı"). Both are real, silent corruptions of a word's actual
# letters if left to the default, confirmed by testing str.lower() directly
# against "İstanbul"/"İSTANBUL" (produces "i̇stanbul", a different string
# than the correct "istanbul").
_TURKISH_CASEFOLD = str.maketrans(
    {"İ": "i", "I": "ı", "Ç": "ç", "Ğ": "ğ", "Ö": "ö", "Ş": "ş", "Ü": "ü"}
)


def _normalize_word(word: str) -> str:
    """Lowercases (Turkish-aware) and strips every non-Turkish-letter
    character (digits, punctuation, foreign-alphabet characters, symbols).

    Added after a real bug found on Colab: without this, `syllabify_word()`
    operating on raw corpus tokens produced 727,154 distinct "syllables"
    from a corpus with only 4,042,857 distinct words — nearly word-scale,
    not the small, reusable syllable inventory Turkish's regular phonology
    should produce. Root cause: punctuation/casing/digits stuck to raw
    tokens ("kitap," vs "kitap." vs "Kitap" vs "kitap123") each produced a
    different, non-reusable syllable string, and Cosmos (a large, noisy,
    web-scraped corpus) has enough of this long-tail junk to nearly double
    the distinct-string count instead of collapsing into Turkish's actually
    small set of real syllable shapes. Normalizing before syllabifying
    fixes this at the source, for both vocab-building and runtime
    inference — every caller goes through `syllabify_word()`, so there's a
    single source of truth, no risk of vocab-build-time and runtime drifting
    out of sync."""
    return "".join(ch for ch in word.translate(_TURKISH_CASEFOLD).lower() if ch in TURKISH_LETTERS)


def syllabify_word(word: str) -> list[str]:
    """Splits a single word into syllables using the rule described in this
    module's docstring, after normalizing (Turkish-aware lowercase, strip
    non-letter characters — see `_normalize_word()`). Pure string logic,
    deterministic, never raises. A word that's all non-letters after
    normalization (pure digits/punctuation/foreign script — e.g. a number,
    a URL fragment) falls back to the original raw word as a single atomic
    "syllable", same never-raise philosophy as `turkish_morphology.py`'s
    `segment_word()`. A normalized word with letters but no vowel (e.g. an
    abbreviation) is likewise returned as a single syllable."""
    if not word:
        return []

    normalized = _normalize_word(word)
    if not normalized:
        return [word]

    vowel_indices = [i for i, ch in enumerate(normalized) if ch in TURKISH_VOWELS]
    if not vowel_indices:
        return [normalized]

    syllables: list[str] = []
    start = 0
    for pos, vowel_i in enumerate(vowel_indices):
        if pos == len(vowel_indices) - 1:
            # Last vowel: its syllable absorbs everything to the end of the word
            # (trailing consonants included).
            syllables.append(normalized[start:])
            break
        next_vowel_i = vowel_indices[pos + 1]
        between = normalized[vowel_i + 1 : next_vowel_i]
        if len(between) <= 1:
            boundary = vowel_i + 1
        else:
            boundary = next_vowel_i - 1
        syllables.append(normalized[start:boundary])
        start = boundary

    return syllables


def build_syllable_vocab(
    corpus_path: str | Path,
    output_dir: str | Path,
    max_syllables: int | None = None,
) -> dict:
    """Reads corpus_path (plain text, one line per doc — same format
    src/data/cosmos_corpus.py's extract_cosmos_corpus_file() produces),
    whitespace-tokenizes, syllabifies every distinct word via
    syllabify_word(), and writes:

      output_dir/syllables.json   — {syllable_string: id}, id 0 reserved for
        <unk_syllable>
      output_dir/vocab_stats.json — word_count, unique_word_count,
        syllable_vocab_size, avg_syllables_per_word

    Unlike build_root_suffix_vocab() in turkish_morphology.py, there is no
    oov_fallback_count here — syllabification never fails to produce
    syllables for a real word, so there is no equivalent failure mode to
    track. No `zeyrek`/analyzer dependency at all, so this runs dramatically
    faster — pure string processing over however many distinct words the
    corpus contains, no per-word rule-based-parser cost.

    max_syllables: optional cap on distinct syllables kept, ranked by real
    corpus frequency (most_common(), same frequency-ranking discipline as
    build_root_suffix_vocab() — see that function's own comment on why
    insertion-order slicing would be a real bug). Expected to rarely matter
    in practice: Turkish's syllable inventory is small and closed (a few
    thousand distinct syllables typically covers the vast majority of real
    usage), unlike the much larger open root/word vocabularies morphology-
    or word-level approaches have to cap.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    word_counts: Counter[str] = Counter()
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            word_counts.update(line.split())

    if not word_counts:
        raise ValueError(f"{corpus_path} produced zero words — refusing to build an empty vocab")

    syllables: dict[str, int] = {UNK_SYLLABLE: 0}
    syllable_counts: Counter[str] = Counter()
    total_syllable_occurrences = 0

    for word, count in word_counts.items():
        word_syllables = syllabify_word(word)
        total_syllable_occurrences += len(word_syllables) * count
        for syl in word_syllables:
            syllable_counts[syl] += count

    ranked_syllables = [syl for syl, _count in syllable_counts.most_common()]
    kept_syllables = ranked_syllables[:max_syllables] if max_syllables is not None else ranked_syllables
    for syl in kept_syllables:
        syllables[syl] = len(syllables)

    unique_word_count = len(word_counts)
    stats = {
        "word_count": sum(word_counts.values()),
        "unique_word_count": unique_word_count,
        "distinct_syllable_count": len(ranked_syllables),
        "syllable_vocab_size": len(syllables),
        "avg_syllables_per_word": (
            total_syllable_occurrences / sum(word_counts.values()) if word_counts else 0.0
        ),
    }

    (output_dir / "syllables.json").write_text(
        json.dumps(syllables, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "vocab_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("build_syllable_vocab: %s", stats)
    return stats
