"""Orchestrates one experiment run: tokenizer surgery -> model cloning (with a chosen
embedding-init strategy) -> distillation -> evaluation, all under one tracked MLflow run.

Stage implementations are resolved by convention: a stage's `strategy` name in the
config must match a module `src/<stage_package>/<strategy_name>.py` that registers
itself with that package's registry on import (see src/registry.py). Adding a new
strategy later (FOCUS, WECHSEL, a contrastive fine-tune objective, ...) means adding
one such file — this orchestrator never changes.

Heavy imports (transformer-cloner, distil-trainer, mteb) are deferred to inside
`run()` rather than module level, so this file — and the config/tracking plumbing
it depends on — can be imported and smoke-tested without those packages installed.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from src.config import ExperimentConfig
from src.distillation.registry import DISTILLATION_REGISTRY
from src.embedding_init.registry import EMBEDDING_INIT_REGISTRY
from src.tokenizer_surgery.registry import TOKENIZER_SURGERY_REGISTRY
from src.tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def _resolve_strategy(package: str, registry: Any, name: str, params: dict) -> Any:
    """Import `<package>.<name>` (registers it as a side effect), then instantiate it."""
    importlib.import_module(f"{package}.{name}")
    strategy_cls = registry.get(name)
    return strategy_cls(**params)


class Pipeline:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tracker = ExperimentTracker(config)

    def run(self) -> dict:
        cfg = self.config
        with self.tracker.start_run():
            self.tracker.log_params(
                {
                    "tokenizer_strategy": cfg.tokenizer_surgery.strategy,
                    "embedding_init": cfg.embedding_init.strategy,
                    "distill_objective": cfg.distillation.strategy,
                    **cfg.model_cloning,
                }
            )

            logger.info("Stage 1/4: tokenizer surgery (%s)", cfg.tokenizer_surgery.strategy)
            tokenizer_strategy = _resolve_strategy(
                "src.tokenizer_surgery",
                TOKENIZER_SURGERY_REGISTRY,
                cfg.tokenizer_surgery.strategy,
                cfg.tokenizer_surgery.params,
            )
            target_tokenizer = tokenizer_strategy.build()

            logger.info("Stage 2/4: model cloning + embedding init (%s)", cfg.embedding_init.strategy)
            embedding_init_strategy = _resolve_strategy(
                "src.embedding_init",
                EMBEDDING_INIT_REGISTRY,
                cfg.embedding_init.strategy,
                cfg.embedding_init.params,
            )
            from src.model_cloning.clone import clone_model  # deferred: heavy deps

            model = clone_model(
                target_tokenizer=target_tokenizer,
                embedding_init_strategy=embedding_init_strategy,
                **cfg.model_cloning,
            )

            logger.info("Stage 3/4: distillation (%s)", cfg.distillation.strategy)
            distillation_strategy = _resolve_strategy(
                "src.distillation",
                DISTILLATION_REGISTRY,
                cfg.distillation.strategy,
                cfg.distillation.params,
            )
            trained_model = distillation_strategy.train(model)

            logger.info("Stage 4/4: evaluation")
            from src.evaluation.mteb_tr_runner import run_evaluation  # deferred: heavy deps

            results = run_evaluation(trained_model, self.tracker, **cfg.evaluation)

            return results
