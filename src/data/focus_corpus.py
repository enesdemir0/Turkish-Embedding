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

# teacher_embeddings.py's EXPECTED_QUOTA assumes ISO 639-3 "tur", but that's never
# been directly confirmed against the dataset's real `lang` values (only row counts
# were checked) — a real Colab run showed 0 rows matched "tur". Rather than eat
# another ~7-minute `.filter()` pass per guess (confirmed slow on that Colab run:
# "Filter: 580000/580000 [06:52<00:00]"), try these common Turkish-code spellings
# against a single cheap columnar read of the unique values first, and filter only
# once with whichever one actually matches.
_TURKISH_LANGUAGE_CODE_CANDIDATES = ("tur", "tr", "TR", "turkish", "Turkish")


def extract_turkish_corpus_file(
    output_path: str | Path,
    dataset_repo: str = DATASET_REPO,
    language: str = "tur",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_teacher_embeddings_dataset(dataset_repo)

    available_languages = set(dataset["lang"])
    if language not in available_languages:
        for candidate in _TURKISH_LANGUAGE_CODE_CANDIDATES:
            if candidate in available_languages:
                language = candidate
                break
        else:
            raise ValueError(
                f"No Turkish-like language code found in {dataset_repo}'s 'lang' column "
                f"(tried {_TURKISH_LANGUAGE_CODE_CANDIDATES}). Actual values present: "
                f"{sorted(available_languages)}"
            )

    filtered = dataset.filter(lambda row: row["lang"] == language)

    if len(filtered) == 0:
        raise ValueError(
            f"No rows matched language='{language}' in {dataset_repo}'s 'lang' column — "
            f"refusing to write an empty corpus file (FOCUS would fail on it downstream "
            f"with a much more confusing error)."
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in filtered:
            f.write(row["text"].replace("\n", " ").strip() + "\n")

    return output_path
