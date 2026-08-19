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

Turkish syllabification rule implemented here (standard, TDK-consistent):
every syllable contains exactly one vowel. For a run of N consonants between
two vowels: if N <= 1, the whole run joins the *following* syllable; if
N >= 2, all but the last consonant stay with the *preceding* syllable, and
the last consonant joins the following syllable (e.g. "ıspanak" ->
"ıs-pa-nak": the 2-consonant "sp" cluster splits as "s" staying with "ıs",
"p" starting "pa"). Leading consonants before the first vowel join the first
syllable; trailing consonants after the last vowel join the last syllable.
A word with no vowels at all (digits, punctuation, acronyms, foreign
fragments) is treated as a single "syllable" (itself), same
never-raise philosophy as `turkish_morphology.py`'s segment_word().
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

UNK_SYLLABLE = "<unk_syllable>"

TURKISH_VOWELS = set("aeıioöuüAEIİOÖUÜ")


def syllabify_word(word: str) -> list[str]:
    """Splits a single word into syllables using the rule described in this
    module's docstring. Pure string logic, deterministic, never raises —
    a word with no vowels returns `[word]` unchanged."""
    if not word:
        return []

    vowel_indices = [i for i, ch in enumerate(word) if ch in TURKISH_VOWELS]
    if not vowel_indices:
        return [word]

    syllables: list[str] = []
    start = 0
    for pos, vowel_i in enumerate(vowel_indices):
        if pos == len(vowel_indices) - 1:
            # Last vowel: its syllable absorbs everything to the end of the word
            # (trailing consonants included).
            syllables.append(word[start:])
            break
        next_vowel_i = vowel_indices[pos + 1]
        between = word[vowel_i + 1 : next_vowel_i]
        if len(between) <= 1:
            boundary = vowel_i + 1
        else:
            boundary = next_vowel_i - 1
        syllables.append(word[start:boundary])
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
