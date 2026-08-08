"""Second-stage contrastive fine-tune with mined hard negatives (TURKIS~1.MD
§3.2/§3.3/§3.5) — a separate, independent objective from `cosine.py`'s teacher-
embedding regression (L_cos). Where cosine distillation teaches "my vector for
this sentence should look like the teacher's vector for this sentence", this
strategy teaches relative geometry directly: "this sentence pair means the same
thing, this other one doesn't" — the signal retrieval/STS tasks actually need,
and exactly what collapsed hardest in the zero-distillation ablation (exp003,
see SESSION_NOTES.md).

Uses sentence-transformers' own native SentenceTransformerTrainer (5.x) —
entirely unrelated to distil-trainer, so none of cosine.py's three monkeypatches
(cosine LR scheduler, device-transfer fix, log-line scraping) apply here. This
file does not import distil_trainer at all.

Intended usage (see configs/experiments/exp004_contrastive_finetune_smoke.yaml):
point tokenizer_surgery + model_cloning.teacher_model at the SAME already-trained
checkpoint (e.g. the paper's published alibayram/embeddingmagibu-200m) with
embedding_init.strategy: mean_composition (whose compose() returns None) — since
source and target tokenizer are then identical, SentenceTransformerCloner performs
an effective identity clone, so the model entering train() below is weight-
identical to that published checkpoint. Mining hard negatives against `model`
itself is therefore mining against the base checkpoint, as intended.
"""

from __future__ import annotations

import logging
from typing import Any

from src.distillation.base import DistillationObjective
from src.distillation.registry import DISTILLATION_REGISTRY

logger = logging.getLogger(__name__)

_TRAINING_ARG_KEYS = {
    "num_epochs": "num_train_epochs",
    "batch_size": "per_device_train_batch_size",
    "learning_rate": "learning_rate",
    "warmup_ratio": "warmup_ratio",
}


def _load_pair_dataset(source: dict, seed: int):
    """Loads one (anchor, positive) pair source and returns a Dataset with
    exactly those two columns. `source` is one entry of the `sources` param
    (see ContrastiveFinetuneDistillationObjective docstring) — a dict with
    `dataset_repo` and optional `dataset_config`, `subsample_size`,
    `anchor_column`, `positive_column`, `score_column`, `min_score`.

    Two known shapes are auto-detected if `anchor_column`/`positive_column`
    aren't given explicitly:
      - NLI-style: {premise, hypothesis, label} -> filters label == 0
        (entailment) and renames premise/hypothesis to anchor/positive.
      - Already pair-shaped: {anchor, positive} -> used as-is.
    A third shape, STS-style {sentence1, sentence2, score}, needs
    `anchor_column`/`positive_column`/`score_column` given explicitly (score
    scale varies by dataset, so it isn't guessed) — `min_score` then filters
    down to only the near-paraphrase pairs, since MultipleNegativesRankingLoss
    needs a binary "these mean the same thing" pair, not a graded similarity.

    Some HF dataset repos (confirmed for emrecan/all-nli-tr, via a real Colab
    error) bundle several views of the same underlying data as separate
    "configs" (e.g. "pair", "pair-class", "pair-score", "triplet") and refuse
    to load without one being named explicitly — pass `dataset_config` for
    those.

    Column names for a new dataset_repo aren't guessed beyond the two known
    shapes above — fails loud if the expected columns aren't present, rather
    than silently misreading garbage into the trainer (same discipline as
    src/data/focus_corpus.py's language-code detection)."""
    from datasets import load_dataset

    dataset_repo = source["dataset_repo"]
    dataset_config = source.get("dataset_config")
    subsample_size = source.get("subsample_size")
    anchor_column = source.get("anchor_column")
    positive_column = source.get("positive_column")
    score_column = source.get("score_column")
    min_score = source.get("min_score")

    dataset = load_dataset(dataset_repo, dataset_config, split="train") if dataset_config else load_dataset(
        dataset_repo, split="train"
    )

    if score_column is not None:
        if min_score is None:
            raise ValueError(f"{dataset_repo!r}: score_column given without min_score to filter on")
        dataset = dataset.filter(lambda row: row[score_column] >= min_score)

    if anchor_column and positive_column:
        dataset = dataset.rename_columns({anchor_column: "anchor", positive_column: "positive"})
    else:
        columns = set(dataset.column_names)
        if {"premise", "hypothesis", "label"} <= columns:
            # Standard NLI numeric encoding: 0 = entailment.
            dataset = dataset.filter(lambda row: row["label"] == 0)
            dataset = dataset.rename_columns({"premise": "anchor", "hypothesis": "positive"})
        elif {"anchor", "positive"} <= columns:
            pass
        else:
            raise ValueError(
                f"Don't know how to extract (anchor, positive) pairs from "
                f"{dataset_repo!r} (config={dataset_config!r}) — got columns "
                f"{sorted(columns)}. Either it matches a known auto-detected shape "
                f"({{'premise', 'hypothesis', 'label'}} or {{'anchor', 'positive'}}), "
                f"or pass anchor_column/positive_column explicitly in this source."
            )

    dataset = dataset.select_columns(["anchor", "positive"])

    if subsample_size is not None:
        num_rows = min(subsample_size, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(num_rows))
        logger.info("%s: subsample_size=%s -> %d pairs", dataset_repo, subsample_size, num_rows)
    else:
        logger.info("%s: using all %d pairs", dataset_repo, len(dataset))

    return dataset


def _load_pairs(params: dict, default_subsample_size: int, seed: int):
    """Builds the full training pair set from `params["sources"]` (a list of
    source dicts, see _load_pair_dataset) if given, else falls back to the
    single-source dataset_repo/dataset_config/subsample_size params (backward
    compatible with configs written before multi-source support existed —
    exp004/exp004b/exp004c all still work unmodified)."""
    from datasets import concatenate_datasets

    sources = params.get("sources")
    if sources is None:
        sources = [
            {
                "dataset_repo": params.get("dataset_repo", "emrecan/all-nli-tr"),
                "dataset_config": params.get("dataset_config", "pair"),
                "subsample_size": params.get("subsample_size", default_subsample_size),
            }
        ]

    datasets_per_source = [_load_pair_dataset(source, seed) for source in sources]
    if len(datasets_per_source) == 1:
        return datasets_per_source[0]
    return concatenate_datasets(datasets_per_source).shuffle(seed=seed)


@DISTILLATION_REGISTRY.register("contrastive_finetune")
class ContrastiveFinetuneDistillationObjective(DistillationObjective):
    """params:
        sources: optional list of source dicts to blend together before hard-
            negative mining, e.g.:
                sources:
                  - dataset_repo: emrecan/all-nli-tr
                    dataset_config: pair
                    subsample_size: 5000
                  - dataset_repo: emrecan/stsb-mt-turkish
                    dataset_config: null
                    subsample_size: 2000
                    anchor_column: sentence1
                    positive_column: sentence2
                    score_column: score
                    min_score: 4.0
            Each source dict is passed to _load_pair_dataset (see its
            docstring for the accepted keys and the two shapes it
            auto-detects). Blending in a second, more general/diverse source
            (e.g. Turkish STS-B, not just NLI) is the fix for a real finding
            from this project's own experiments: training on NLI pairs alone
            improves retrieval/STS but measurably degrades NLI-as-classification
            and general sentiment/topic classification tasks, and the damage
            scales with total training steps (not fixable by learning_rate
            alone) — see exp004/exp004b/exp004c in SESSION_NOTES.md.
            If `sources` is omitted, falls back to the single-source
            dataset_repo/dataset_config/subsample_size params below
            (backward compatible with exp004/exp004b/exp004c's configs).
        dataset_repo, dataset_config, subsample_size: single-source shorthand,
            used only if `sources` is not given. dataset_repo default:
            emrecan/all-nli-tr, dataset_config default: "pair" (some repos,
            confirmed for emrecan/all-nli-tr, bundle several views of the same
            data and require a config name to be picked explicitly).
        num_negatives, range_min, range_max, margin, sampling_strategy: passed to
            sentence_transformers.util.mine_hard_negatives(). Defaults are
            conservative (range_min=10) since short/generic NLI sentences risk
            false negatives at aggressive ranks.
        num_epochs, batch_size, learning_rate, warmup_ratio: standard training
            hyperparameters (defaults are a first-guess starting point, not a
            paper-derived constant like cosine.py's PAPER_HYPERPARAMETERS).
        output_dir: where the fine-tuned model is saved (default: ./distilled_model_contrastive).
        seed: fixed subsample/training seed (default: 42, project-wide convention).
        use_bf16: default True (matches cosine.py, targets A100 training).
    (tracker is passed to .train() by the pipeline, not read from params here.)
    """

    def train(self, model: Any, tracker: Any | None = None) -> Any:
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.util import mine_hard_negatives
        from transformers import TrainerCallback

        num_negatives = self.params.get("num_negatives", 5)
        range_min = self.params.get("range_min", 10)
        range_max = self.params.get("range_max", 50)
        margin = self.params.get("margin")
        sampling_strategy = self.params.get("sampling_strategy", "top")
        output_dir = self.params.get("output_dir", "./distilled_model_contrastive")
        seed = self.params.get("seed", 42)
        use_bf16 = self.params.get("use_bf16", True)
        mining_batch_size = self.params.get("mining_batch_size", 64)

        training_args_kwargs = {
            arg_name: self.params.get(param_name, default)
            for param_name, arg_name, default in (
                ("num_epochs", "num_train_epochs", 1),
                ("batch_size", "per_device_train_batch_size", 32),
                ("learning_rate", "learning_rate", 2e-5),
                ("warmup_ratio", "warmup_ratio", 0.1),
            )
        }

        if tracker is not None:
            tracker.log_params(
                {
                    "sources": self.params.get(
                        "sources",
                        [
                            {
                                "dataset_repo": self.params.get("dataset_repo", "emrecan/all-nli-tr"),
                                "dataset_config": self.params.get("dataset_config", "pair"),
                                "subsample_size": self.params.get("subsample_size", 5000),
                            }
                        ],
                    ),
                    "num_negatives": num_negatives,
                    "range_min": range_min,
                    "range_max": range_max,
                    "margin": margin,
                    "sampling_strategy": sampling_strategy,
                    "seed": seed,
                    **training_args_kwargs,
                }
            )

        pairs = _load_pairs(self.params, default_subsample_size=5000, seed=seed)

        triplets = mine_hard_negatives(
            pairs,
            model,
            anchor_column_name="anchor",
            positive_column_name="positive",
            num_negatives=num_negatives,
            range_min=range_min,
            range_max=range_max,
            margin=margin,
            sampling_strategy=sampling_strategy,
            batch_size=mining_batch_size,
            use_faiss=False,
            output_format="triplet",
        )
        logger.info("Mined %d hard-negative triplets from %d pairs", len(triplets), len(pairs))

        loss = MultipleNegativesRankingLoss(model)

        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            seed=seed,
            bf16=use_bf16,
            **training_args_kwargs,
        )

        callbacks = []
        if tracker is not None:

            class _TrackerLogCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not logs:
                        return
                    metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                    if metrics:
                        tracker.log_metrics(metrics, step=state.global_step)

            callbacks.append(_TrackerLogCallback())

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=triplets,
            loss=loss,
            callbacks=callbacks or None,
        )
        trainer.train()

        model.save(output_dir)
        logger.info("Contrastive fine-tune complete, saved to %s", output_dir)
        return model
