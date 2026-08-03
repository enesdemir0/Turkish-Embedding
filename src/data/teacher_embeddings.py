"""Loads `alibayram/wikipedia-40-langs-with-embeddings` — the paper's precomputed
distillation corpus (EMBEDD~3.MD §5.1-5.2): EmbeddingGemma-300M embeddings computed
once, ahead of training, over a 40-language Wikipedia corpus, with a language quota
(Turkish/English capped at 100K rows each, the remaining 38 languages capped at 10K
rows each) applied specifically to stop high-resource languages from dominating.

Confirmed by reading the dataset's actual schema (not assumed): its `train` split
already has exactly 580,000 rows — matching the paper's stated quota result — and its
columns (`lang`, `text`, `teacher_embedding_final`, `teacher_embedding_pre_dense`)
already match distil-trainer's `EmbeddingTrainerConfig` defaults exactly. So this
loader trusts the published dataset is already quota-shaped rather than re-filtering
by language, which would mean scanning the full 11GB corpus for no benefit.
`verify_language_quota()` is provided separately as an explicit, on-demand check
against the paper's caps, since the row count alone doesn't strictly prove the
per-language distribution.
"""

from __future__ import annotations

from collections import Counter

from datasets import Dataset, load_dataset

DATASET_REPO = "alibayram/wikipedia-40-langs-with-embeddings"

EXPECTED_QUOTA = {"tur": 100_000, "eng": 100_000}
EXPECTED_DEFAULT_CAP = 10_000
EXPECTED_TOTAL_ROWS = 580_000


def load_teacher_embeddings_dataset(dataset_repo: str = DATASET_REPO, split: str = "train") -> Dataset:
    return load_dataset(dataset_repo, split=split)


def verify_language_quota(dataset: Dataset, language_column: str = "lang") -> dict[str, int]:
    """Returns per-language row counts. Raises if any language exceeds the paper's
    stated cap, so a quota violation is caught explicitly rather than assumed away."""
    counts = Counter(dataset[language_column])
    for lang, count in counts.items():
        cap = EXPECTED_QUOTA.get(lang, EXPECTED_DEFAULT_CAP)
        if count > cap:
            raise ValueError(f"Language '{lang}' has {count} rows, exceeding the paper's cap of {cap}")
    return dict(counts)
