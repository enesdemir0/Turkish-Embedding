"""Builds root/suffix vocabularies for exp009's compositional root+suffix
embedding (`src/model_cloning/compositional_root_suffix.py`) by running an
already-extracted plain-text Turkish corpus (e.g. `cosmos_corpus.py`'s output)
through `zeyrek`'s morphological analyzer.

Not a corpus extractor like `cosmos_corpus.py`/`focus_corpus.py`/
`wiki40langs_corpus.py` (those write one cleaned text line per row for a
downstream tokenizer trainer). This consumes that output and produces two
small id-lookup vocabularies instead.

zeyrek's real `analyze()` return shape, confirmed by inspecting live output
(not guessed from docs): a list of candidate-groups, each itself a list of
zero or more `Parse` namedtuples (`word`, `lemma`, `pos`, `morphemes`,
`formatted`) — i.e. `list[list[Parse]]`, not `list[Parse]`. An unanalyzable
word still produces at least one group containing a placeholder
`Parse(lemma="Unk", pos="Unk", morphemes="Unk", ...)` (a string, not a list,
for `morphemes` in that case) plus sometimes an extra empty group. Also
confirmed: `morphemes[0]` duplicates the parse's own `pos` tag (e.g. `"Noun"`)
rather than being a real suffix — the real suffix chain is `morphemes[1:]`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UNK_ROOT = "<unk_root>"
UNK_SUFFIX = "<unk_suffix>"


def segment_word(word: str, analyzer: Any) -> tuple[str, list[str], bool]:
    """Returns (root, [suffix, ...], is_fallback) for one word using zeyrek's
    first valid analysis candidate. is_fallback is True whenever zeyrek found
    no valid analysis at all (OOV, proper noun zeyrek's dictionary lacks,
    typo, punctuation, digits, empty string) and the whole surface form was
    used as its own root with an empty suffix chain instead — this is
    distinct from a real analysis that legitimately has zero suffixes (e.g.
    "merhaba" -> ("merhaba", [], False)), which must NOT count as a fallback
    or oov_fallback_fraction below would be meaningless. Never raises on real
    corpus text.
    """
    if not word:
        return word, [], True

    try:
        result = analyzer.analyze(word)
    except Exception:
        logger.debug("zeyrek.analyze() raised for %r; falling back to whole-word root", word)
        return word, [], True

    candidates = [parse for group in result for parse in group]
    valid = [
        parse
        for parse in candidates
        if parse.lemma != "Unk" and isinstance(parse.morphemes, list)
    ]
    if not valid:
        return word, [], True

    first = valid[0]
    suffixes = first.morphemes[1:] if len(first.morphemes) > 1 else []
    return first.lemma, suffixes, False


def build_root_suffix_vocab(
    corpus_path: str | Path,
    output_dir: str | Path,
    max_words: int | None = None,
) -> dict:
    """Reads corpus_path (plain text, one line per doc — the format
    extract_cosmos_corpus_file already produces), whitespace-tokenizes,
    segments every distinct word via segment_word(), and writes:

      output_dir/roots.json       — {root_string: id}, id 0 reserved for <unk_root>
      output_dir/suffixes.json    — {suffix_string: id}, id 0 reserved for <unk_suffix>
      output_dir/vocab_stats.json — word_count, unique_word_count, root_vocab_size,
        suffix_vocab_size, oov_fallback_count, oov_fallback_fraction (fraction of
        distinct words where zeyrek found no valid analysis and segment_word fell
        back to whole-word-as-root — the key go/no-go diagnostic: if this is near
        1.0, the zeyrek integration likely has a real bug, not just normal OOV
        noise)

    Returns the stats dict — callers should log this to the tracker (durable,
    unlike a plain console/logger line — see working agreements in
    SESSION_NOTES.md: a diagnostic that only exists in an ephemeral Colab
    console log has been lost before).

    max_words: optional cap on distinct words processed, for a cheap smoke-test
    pass over a small corpus sample.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import zeyrek

    analyzer = zeyrek.MorphAnalyzer()

    word_counts: Counter[str] = Counter()
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            word_counts.update(line.split())

    if not word_counts:
        raise ValueError(f"{corpus_path} produced zero words — refusing to build an empty vocab")

    distinct_words = list(word_counts.keys())
    if max_words is not None:
        distinct_words = distinct_words[:max_words]

    roots: dict[str, int] = {UNK_ROOT: 0}
    suffixes: dict[str, int] = {UNK_SUFFIX: 0}
    oov_fallback_count = 0

    for word in distinct_words:
        root, word_suffixes, is_fallback = segment_word(word, analyzer)
        if root not in roots:
            roots[root] = len(roots)
        for suffix in word_suffixes:
            if suffix not in suffixes:
                suffixes[suffix] = len(suffixes)
        if is_fallback:
            oov_fallback_count += 1

    unique_word_count = len(distinct_words)
    stats = {
        "word_count": sum(word_counts.values()),
        "unique_word_count": unique_word_count,
        "root_vocab_size": len(roots),
        "suffix_vocab_size": len(suffixes),
        "oov_fallback_count": oov_fallback_count,
        "oov_fallback_fraction": oov_fallback_count / unique_word_count if unique_word_count else 0.0,
    }

    (output_dir / "roots.json").write_text(
        json.dumps(roots, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "suffixes.json").write_text(
        json.dumps(suffixes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "vocab_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("build_root_suffix_vocab: %s", stats)
    return stats
