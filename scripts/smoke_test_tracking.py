"""Verifies DagsHub/MLflow connectivity and the run-naming/tagging convention, before
any real pipeline stage exists. This is Step 1's own verification: after setting your
DagsHub token (see README), run this and confirm a run named 'exp000_smoketest' shows
up on DagsHub under the 'turkish-embedding' MLflow experiment with the expected tags
and this file logged as an artifact.

Usage:
    python scripts/smoke_test_tracking.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig, StageConfig, TrackingConfig  # noqa: E402
from src.tracking.experiment_tracker import ExperimentTracker, init_tracking  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    init_tracking()

    config = ExperimentConfig(
        config_path=Path(__file__),
        number=0,
        description="Tracking plumbing smoke test — no real training happens here.",
        tokenizer_surgery=StageConfig(strategy="none"),
        embedding_init=StageConfig(strategy="none"),
        model_cloning={"teacher_model": "none"},
        distillation=StageConfig(strategy="none"),
        evaluation={},
        tracking=TrackingConfig(
            mlflow_experiment="turkish-embedding",
            run_name="exp000_smoketest",
            tags={"purpose": "tracking-smoke-test"},
        ),
    )

    tracker = ExperimentTracker(config)
    with tracker.start_run():
        tracker.log_metrics({"dummy_metric": 1.0})

    logging.getLogger(__name__).info(
        "Done. Check https://dagshub.com/%s/%s/experiments for run 'exp000_smoketest'.",
        "enesdemir0",
        "Turkish-Embedding",
    )


if __name__ == "__main__":
    main()
