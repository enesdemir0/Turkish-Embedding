"""Experiment config loading.

A config file is the single source of truth for one experiment run: which strategy
each pipeline stage uses, that strategy's params, and how the run should be named/
tagged in MLflow. Strategy-specific params are kept as plain dicts (not dataclasses)
because each strategy owns its own params and new strategies shouldn't require
changes here — only `TrackingConfig` and `ExperimentConfig`'s top-level shape are
fixed, since the tracker and pipeline orchestrator both depend on those directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageConfig:
    """One pipeline stage's chosen strategy name + its strategy-specific params."""

    strategy: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageConfig":
        return cls(strategy=data["strategy"], params=data.get("params", {}))


@dataclass
class TrackingConfig:
    mlflow_experiment: str
    # Auto-derived from the rest of the config if left unset (see src/tracking/naming.py)
    # — only override this for a one-off reason, since a hand-typed name can drift
    # from what the run actually did.
    run_name: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackingConfig":
        return cls(
            mlflow_experiment=data["mlflow_experiment"],
            run_name=data.get("run_name"),
            tags=data.get("tags", {}),
        )


@dataclass
class ExperimentConfig:
    config_path: Path
    number: int
    description: str
    model_cloning: dict[str, Any]  # must include "teacher_model" (a HF model id)
    distillation: StageConfig
    evaluation: dict[str, Any]
    tracking: TrackingConfig
    # Optional: a model_cloning strategy that supplies its own already-tokenized
    # backbone (e.g. native_pretrained) has no use for these — see
    # ModelCloningStrategy.requires_tokenizer_surgery/requires_embedding_init.
    tokenizer_surgery: StageConfig | None = None
    embedding_init: StageConfig | None = None

    @property
    def teacher_model(self) -> str:
        """The HF model id used as the distillation teacher — pulled out as its own
        property (rather than read ad hoc from model_cloning) because it's part of a
        run's identity: naming/tagging depend on it, and future experiments may swap
        the teacher itself (not just the tokenizer/init/objective)."""
        return self.model_cloning["teacher_model"]

    @property
    def model_cloning_strategy(self) -> str:
        """Defaults to "transformer_clone" (today's only pre-existing behavior) so
        every config written before this strategy field existed keeps working
        unmodified."""
        return self.model_cloning.get("strategy", "transformer_clone")

    @classmethod
    def load(cls, config_path: str | Path) -> "ExperimentConfig":
        config_path = Path(config_path)
        with config_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        experiment = raw["experiment"]
        raw_tokenizer_surgery = raw.get("tokenizer_surgery")
        raw_embedding_init = raw.get("embedding_init")
        return cls(
            config_path=config_path,
            number=experiment["number"],
            description=experiment["description"],
            tokenizer_surgery=StageConfig.from_dict(raw_tokenizer_surgery) if raw_tokenizer_surgery else None,
            embedding_init=StageConfig.from_dict(raw_embedding_init) if raw_embedding_init else None,
            model_cloning=raw["model_cloning"],
            distillation=StageConfig.from_dict(raw["distillation"]),
            evaluation=raw["evaluation"],
            tracking=TrackingConfig.from_dict(raw["tracking"]),
        )
