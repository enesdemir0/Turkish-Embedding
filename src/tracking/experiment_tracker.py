"""DagsHub/MLflow tracking, with a standardized run-naming/tagging convention.

One MLflow *experiment* holds every run for the whole project (see README section on
naming convention) — this is what lets runs be compared side-by-side in DagsHub once
there are many of them. Each *run*'s name encodes what varies
(exp{NNN}_{tokenizer_strategy}_{embedding_init}_{distill_objective}); the underlying
tags carry the same information in a machine-filterable form, plus provenance
(git commit, config file) that the name alone can't capture.

Requires a DagsHub token to actually connect — see README/env setup. Without one,
`init_tracking` raises rather than silently falling back to a local file store, since
a "successful" run that didn't actually reach DagsHub would defeat the point of
centralized experiment comparison.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

from src.config import ExperimentConfig
from src.tracking.naming import build_run_name

REPO_OWNER = "enesdemir0"
REPO_NAME = "Turkish-Embedding"


def init_tracking(repo_owner: str = REPO_OWNER, repo_name: str = REPO_NAME) -> None:
    """Point MLflow at this project's DagsHub-hosted tracking server.

    Requires DAGSHUB_USER_TOKEN (or MLFLOW_TRACKING_USERNAME/PASSWORD) to be set in
    the environment — see the DagsHub docs for how to mint a token. This must be
    called once before any ExperimentTracker.start_run().
    """
    import dagshub

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class ExperimentTracker:
    """Wraps one experiment run's MLflow lifecycle: naming, tagging, artifact logging."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    @contextmanager
    def start_run(self) -> Iterator[None]:
        cfg = self.config
        run_name = cfg.tracking.run_name or build_run_name(
            number=cfg.number,
            teacher_model=cfg.teacher_model,
            tokenizer_strategy=cfg.tokenizer_surgery.strategy,
            embedding_init=cfg.embedding_init.strategy,
            distill_objective=cfg.distillation.strategy,
        )

        mlflow.set_experiment(cfg.tracking.mlflow_experiment)
        with mlflow.start_run(run_name=run_name):
            # Core tags are derived from the config itself, not hand-typed, so they
            # can never drift from what the run actually did. cfg.tracking.tags can
            # add extra one-off tags but should not need to repeat these.
            core_tags = {
                "teacher_model": cfg.teacher_model,
                "tokenizer_strategy": cfg.tokenizer_surgery.strategy,
                "embedding_init": cfg.embedding_init.strategy,
                "distill_objective": cfg.distillation.strategy,
                "git_commit": _git_commit_hash(),
                "config_file": str(cfg.config_path),
            }
            mlflow.set_tags({**core_tags, **cfg.tracking.tags})
            mlflow.log_artifact(str(cfg.config_path))
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)
