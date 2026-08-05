"""Materializes a plain-text Turkish corpus file for FOCUS's auxiliary
target-language fasttext embeddings (embedding_init/focus_init.py), reusing the
same distillation corpus already loaded by teacher_embeddings.py
(`alibayram/wikipedia-40-langs-with-embeddings`, ~100K Turkish rows) rather than
introducing a separate corpus dependency.

Kept as a standalone function rather than added to teacher_embeddings.py since it's
a FOCUS-specific concern (a plain-text file, not a HF Dataset object) — if Colab
reveals the real deepfocus API wants an in-memory Dataset/iterable instead, that's a
change confined to this file and focus_init.py, not the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from src.data.teacher_embeddings import DATASET_REPO, load_teacher_embeddings_dataset


def extract_turkish_corpus_file(
    output_path: str | Path,
    dataset_repo: str = DATASET_REPO,
    language: str = "tur",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_teacher_embeddings_dataset(dataset_repo)
    filtered = dataset.filter(lambda row: row["lang"] == language)

    if len(filtered) == 0:
        raise ValueError(
            f"No rows matched language='{language}' in {dataset_repo}'s 'lang' column — "
            f"refusing to write an empty corpus file (FOCUS would fail on it downstream "
            f"with a much more confusing error). Check the dataset's actual lang values."
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in filtered:
            f.write(row["text"].replace("\n", " ").strip() + "\n")

    return output_path
