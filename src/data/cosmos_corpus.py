"""Materializes a plain-text Turkish corpus file from the Cosmos Turkish Corpus v1.0
(EMBEDD~3.MD §2.1) — the from-scratch Turkish tokenizer training source for
`tokenizer_surgery/frequency_pruning.py`.

Kept separate from `src/data/focus_corpus.py`: that file reuses
`alibayram/wikipedia-40-langs-with-embeddings` (the distillation corpus, filtered to
Turkish) for a different purpose (FOCUS's auxiliary fasttext embeddings). This is a
different HF repo entirely, with its own (not yet directly confirmed) schema, so it
gets its own loader rather than being forced into that file's single-purpose scope.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

DATASET_REPO = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0"


def extract_cosmos_corpus_file(
    output_path: str | Path,
    dataset_repo: str = DATASET_REPO,
    subsample_fraction: float | None = None,
    text_column: str = "text",
) -> Path:
    """Writes one cleaned text line per row to a plain-text file at output_path.

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
