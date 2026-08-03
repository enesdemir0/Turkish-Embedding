"""CLI entrypoint for running one experiment end-to-end.

Usage:
    python scripts/run_experiment.py --config configs/experiments/exp001_mean_composition_cosine.yaml

Requires DagsHub tracking credentials in the environment (see README) — run
scripts/smoke_test_tracking.py first if you haven't verified connectivity yet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.tracking.experiment_tracker import init_tracking  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    init_tracking()
    config = ExperimentConfig.load(args.config)
    results = Pipeline(config).run()
    logging.getLogger(__name__).info("Run complete. Results: %s", results)


if __name__ == "__main__":
    main()
