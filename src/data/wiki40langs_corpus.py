"""Materializes an all-languages plain-text corpus file from Wikipedia-40-langs
(EMBEDD~3.MD §2.4) — the multilingual frequency-analysis source for
`tokenizer_surgery/frequency_pruning.py`.

Deliberately NOT the same helper as `src/data/focus_corpus.py`'s
`extract_turkish_corpus_file`: that function targets a different HF repo
(`alibayram/wikipedia-40-langs-with-embeddings`) and filters down to a single
language (Turkish only), for FOCUS's auxiliary fasttext embeddings. §2.4 needs the
opposite shape — every language, no filter, and the plain-text (no precomputed
embeddings) `alibayram/wikipedia-40-langs` repo — different repo and different
schema, so it gets its own module rather than overloading that file's single-purpose
Turkish-only scope.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

DATASET_REPO = "alibayram/wikipedia-40-langs"


def extract_multilingual_corpus_file(
    output_path: str | Path,
    dataset_repo: str = DATASET_REPO,
    subsample_fraction: float | None = None,
    text_column: str = "text",
) -> Path:
    """Writes one cleaned text line per row (every language, no filter) to a
    plain-text file at output_path.

    subsample_fraction: optional float in (0, 1]. If set, uses a fixed shuffle seed
    (42, matching src/distillation/cosine.py's existing convention) so repeated runs
    at the same fraction draw the same rows — for a cheap first pass before
    committing to the full corpus.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset: Dataset = load_dataset(dataset_repo, split="train")
    logger.info(
        "Loaded %s: %d rows, columns=%s", dataset_repo, len(dataset), dataset.column_names
    )

    if text_column not in dataset.column_names:
        raise ValueError(
            f"{dataset_repo} has no '{text_column}' column — actual columns: "
            f"{dataset.column_names}"
        )

    if subsample_fraction is not None:
        if not 0 < subsample_fraction <= 1:
            raise ValueError(f"subsample_fraction must be in (0, 1], got {subsample_fraction}")
        num_rows = max(1, int(len(dataset) * subsample_fraction))
        dataset = dataset.shuffle(seed=42).select(range(num_rows))
        logger.info("subsample_fraction=%s: using %d rows", subsample_fraction, num_rows)

    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            text = row[text_column]
            if not text:
                continue
            f.write(text.replace("\n", " ").strip() + "\n")

    return output_path
