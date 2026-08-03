"""Derives the MLflow run name straight from the resolved config, instead of the
config author hand-typing it — a hand-typed name can silently drift from what the run
actually did (e.g. someone changes `teacher_model` but forgets to update the name).
Auto-deriving it means the name is always an accurate summary.

Convention: exp{NNN}_{teacher_model}_{tokenizer_strategy}_{embedding_init}_{distill_objective}
e.g. exp001_embeddinggemma300m_frequency_pruning_mean_composition_cosine
"""

from __future__ import annotations

import re


def slugify_model_id(model_id: str) -> str:
    """'google/embeddinggemma-300m' -> 'embeddinggemma300m'."""
    name = model_id.split("/")[-1]
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_run_name(
    number: int,
    teacher_model: str,
    tokenizer_strategy: str,
    embedding_init: str,
    distill_objective: str,
) -> str:
    return (
        f"exp{number:03d}_{slugify_model_id(teacher_model)}"
        f"_{tokenizer_strategy}_{embedding_init}_{distill_objective}"
    )
