"""Evaluation runner — replaces benchmark_classification_colab.py's hardcoded,
Classification-only benchmark with a config-driven runner covering the full Turkish
task suite (Classification, STS, Retrieval, PairClassification, Reranking,
Clustering), with every result logged to MLflow so every experiment lands in the
same place for comparison.

selmanbaysan/mteb_tr's own CLI (mteb_tr_cli.py, read directly from the repo rather
than guessed) doesn't hardcode a task list at all — it calls
`mteb.get_benchmark("MTEB(Turkish)")`, upstream mteb's own maintained Turkish
benchmark registry. That's a better source of truth than any list we could
hardcode ourselves, so this runner does the same by default. `task_names` remains
available to run a narrower, explicit subset (e.g. just the 8 Classification tasks
benchmark_classification_colab.py used) when a full run isn't wanted yet.

The result-parsing logic (mteb.MTEB(tasks=...).run(...) -> parse each task's result
JSON from output_folder) matches what benchmark_classification_colab.py already
validated. mteb itself is intentionally not installed locally (per project
convention: heavy deps/installs happen on Colab) — the pearson/spearman key names
extracted below for STS tasks are a best guess at mteb's result schema and should be
checked against real output on the first Colab run.
"""

from __future__ import annotations

import glob
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# The paper's own headline comparison point (EMBEDD~3.MD §5, model card): student vs.
# teacher on STSbTR specifically (published: Pearson 0.8199 / Spearman 0.7980). Worth
# surfacing as its own metric, not just buried in the per-task table, since it's the
# sharpest single "did we reproduce it" checkpoint against the paper's published numbers.
STSB_TR_TASK_NAME = "STSbTR"

# mteb's per-split result schema varies by task type; these are the key names most
# likely to hold pearson/spearman for an STS task, tried in order. Confirm against
# real output on the first Colab run and trim this list once the real schema is known.
_PEARSON_KEYS = ("pearson", "cosine_pearson")
_SPEARMAN_KEYS = ("spearman", "cosine_spearman")


def run_evaluation(
    model: Any,
    tracker: Any,
    task_names: list[str] | None = None,
    benchmark_name: str | None = "MTEB(Turkish)",
    output_folder: str = "results",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Either pass explicit `task_names` for a narrow subset, or leave it None to
    run the full `benchmark_name` suite (default: upstream mteb's maintained
    Turkish benchmark) — matching how mteb_tr's own CLI runs evaluations."""
    import mteb

    if task_names:
        tasks = mteb.get_tasks(tasks=task_names)
        found_names = {t.metadata.name for t in tasks}
        missing = set(task_names) - found_names
        if missing:
            logger.warning("Requested tasks not found in mteb's task registry: %s", sorted(missing))
        evaluation = mteb.MTEB(tasks=tasks)
    else:
        benchmark = mteb.get_benchmark(benchmark_name)
        evaluation = mteb.MTEB(tasks=benchmark)

    evaluation.run(model, output_folder=output_folder, encode_kwargs={"batch_size": batch_size})

    rows = _parse_result_files(output_folder)
    _log_to_tracker(tracker, rows)

    return {"rows": rows, "task_count": len(tasks)}


def _first_present(entry: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in entry:
            return entry[key]
    return None


def _parse_result_files(output_folder: str) -> list[dict[str, Any]]:
    rows = []
    for path in glob.glob(f"{output_folder}/**/*.json", recursive=True):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        task_name = data.get("task_name")
        if task_name is None:
            continue

        for split_scores in data.get("scores", {}).values():
            for entry in split_scores:
                rows.append(
                    {
                        "task": task_name,
                        "split": entry.get("hf_subset", "default"),
                        "main_score": entry.get("main_score"),
                        "pearson": _first_present(entry, _PEARSON_KEYS),
                        "spearman": _first_present(entry, _SPEARMAN_KEYS),
                    }
                )
    return rows


def _log_to_tracker(tracker: Any, rows: list[dict[str, Any]]) -> None:
    if tracker is None:
        return

    for row in rows:
        if row["main_score"] is not None:
            tracker.log_metrics({f"mteb/{row['task']}/{row['split']}": row["main_score"]})

    for row in rows:
        if row["task"] != STSB_TR_TASK_NAME:
            continue
        if row["pearson"] is not None:
            tracker.log_metrics({f"stsb_tr_pearson_{row['split']}": row["pearson"]})
        if row["spearman"] is not None:
            tracker.log_metrics({f"stsb_tr_spearman_{row['split']}": row["spearman"]})

    scores = [row["main_score"] for row in rows if row["main_score"] is not None]
    if scores:
        tracker.log_metrics({"mteb_mean_score": sum(scores) / len(scores)})
