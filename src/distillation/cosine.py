"""The paper's exact offline distillation objective (EMBEDD~3.MD §5): cosine
distillation loss (L_cos) against precomputed EmbeddingGemma-300M sentence vectors,
using the published hyperparameters (§5.4).

Wraps distil-trainer's EmbeddingDistillationTrainer/EmbeddingTrainerConfig, whose
defaults already line up with several of the paper's values (target_type="final",
text_column="text", final_embedding_column="teacher_embedding_final",
weight_decay=0.01, max_grad_norm=1.0) — confirmed by reading the installed
package's real source (v0.1.27), not assumed from documentation.

Three things needed a deliberate fix rather than a plain config pass-through, all
found the same way (reading the source, not guessing):

1. Its LR scheduler is hardcoded to "linear" with no config option — the paper
   specifies cosine decay from the peak LR to zero. `_force_cosine_lr_schedule`
   patches the one `get_scheduler(...)` call the library makes internally, rather
   than forking its ~150-line train() loop to change one string.
2. It has no MLflow/DagsHub integration (only optional Weights & Biases). Its
   per-step loss lines are logged via the stdlib `logging` module, so
   `_MlflowLossLogHandler` attaches to that logger and forwards each step's loss/lr
   to our tracker — giving a real loss curve to compare against the paper's
   reported trajectory (~0.09 -> ~0.07 within 200 steps -> ~0.05 by epoch end),
   rather than only the final summary dict `.train()` returns.
3. `_forward_with_gradients`/`_forward_pre_dense` do `v.to(self.device) for k, v in
   features.items()` over every key in the tokenizer's output unconditionally.
   Newer sentence-transformers (5.x, confirmed against the installed 5.6.1) added a
   `"modality"` key to that output (a plain string, for multi-modal support) — so
   `.to()` on a `str` crashes with `AttributeError: 'str' object has no attribute
   'to'`. `_patch_device_transfer` replaces both methods with a version that only
   calls `.to()` on values that actually have it.
"""

from __future__ import annotations

import logging
import re
import tempfile
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from src.distillation.base import DistillationObjective
from src.distillation.registry import DISTILLATION_REGISTRY

logger = logging.getLogger(__name__)

# Paper's exact hyperparameters (EMBEDD~3.MD §5.4). Any of these keys can be
# overridden via the strategy's `params`, but doing so means the run is no longer
# the exact paper reproduction.
PAPER_HYPERPARAMETERS: dict[str, Any] = dict(
    num_epochs=1,
    batch_size=256,
    learning_rate=5e-5,
    warmup_ratio=0.01,
    weight_decay=0.01,
    max_grad_norm=1.0,
    loss_type="cosine",
    target_type="final",
    use_bf16=True,
    compile_model=True,
    gradient_checkpointing=True,
    save_steps=100,
)

_STEP_LOG_PATTERN = re.compile(r"Step (\d+): loss = ([\d.]+), lr = ([\d.eE+-]+)")


class _MlflowLossLogHandler(logging.Handler):
    """Forwards distil-trainer's per-step "Step N: loss = ..., lr = ..." log lines
    to our own tracker, since the library has no callback/metrics-hook API."""

    def __init__(self, tracker: Any):
        super().__init__(level=logging.INFO)
        self._tracker = tracker

    def emit(self, record: logging.LogRecord) -> None:
        match = _STEP_LOG_PATTERN.search(record.getMessage())
        if not match:
            return
        step, loss, lr = match.groups()
        self._tracker.log_metrics({"train_loss": float(loss), "learning_rate": float(lr)}, step=int(step))


@contextmanager
def _force_cosine_lr_schedule():
    from transformers import get_scheduler as hf_get_scheduler

    def cosine_get_scheduler(name, *args, **kwargs):
        return hf_get_scheduler("cosine", *args, **kwargs)

    with patch("distil_trainer.core.embedding_trainer.get_scheduler", cosine_get_scheduler):
        yield


def _forward_with_gradients(self, texts: list[str]):
    """Drop-in replacement for EmbeddingDistillationTrainer._forward_with_gradients
    (confirmed against the installed source, v0.1.27) — identical except the device
    transfer only touches values that are actually tensors (see module docstring,
    point 3)."""
    model = self._original_model
    features = model.tokenize(texts)
    features = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in features.items()}

    for module in model:
        features = module(features)

    embedding = features.get("sentence_embedding")
    if embedding is None:
        embedding = features.get("token_embeddings")
        if embedding is not None and len(embedding.shape) == 3:
            embedding = embedding.mean(dim=1)

    return embedding


def _forward_pre_dense(self, texts: list[str]):
    """Drop-in replacement for EmbeddingDistillationTrainer._forward_pre_dense —
    same fix as _forward_with_gradients above."""
    model = self._original_model
    pre_dense_idx = self._get_pre_dense_index()

    features = model.tokenize(texts)
    features = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in features.items()}

    for idx in range(pre_dense_idx):
        features = model[idx](features)

    embedding = features.get("sentence_embedding")
    if embedding is None:
        embedding = features.get("token_embeddings")
        if embedding is not None and len(embedding.shape) == 3:
            embedding = embedding.mean(dim=1)

    return embedding


@contextmanager
def _patch_device_transfer():
    from distil_trainer.core.embedding_trainer import EmbeddingDistillationTrainer

    original_gradients = EmbeddingDistillationTrainer._forward_with_gradients
    original_pre_dense = EmbeddingDistillationTrainer._forward_pre_dense
    EmbeddingDistillationTrainer._forward_with_gradients = _forward_with_gradients
    EmbeddingDistillationTrainer._forward_pre_dense = _forward_pre_dense
    try:
        yield
    finally:
        EmbeddingDistillationTrainer._forward_with_gradients = original_gradients
        EmbeddingDistillationTrainer._forward_pre_dense = original_pre_dense


@contextmanager
def _log_loss_to_tracker(tracker: Any | None):
    if tracker is None:
        yield
        return

    handler = _MlflowLossLogHandler(tracker)
    trainer_logger = logging.getLogger("distil_trainer.core.embedding_trainer")
    # A logger only reaches its handlers if its own effective level admits the
    # record first; root defaults to WARNING, which would otherwise silently drop
    # every .info() call before our handler ever sees it.
    previous_level = trainer_logger.level
    trainer_logger.setLevel(logging.INFO)
    trainer_logger.addHandler(handler)
    try:
        yield
    finally:
        trainer_logger.removeHandler(handler)
        trainer_logger.setLevel(previous_level)


@DISTILLATION_REGISTRY.register("cosine")
class CosineDistillationObjective(DistillationObjective):
    """params:
        dataset_repo: HF dataset repo id with precomputed teacher embeddings
            (default: alibayram/wikipedia-40-langs-with-embeddings).
        output_dir: where the trained student model is saved (default: ./distilled_model).
        Any PAPER_HYPERPARAMETERS key may be overridden here too.
    (tracker is passed to .train() by the pipeline, not read from params here.)
    """

    def train(self, model: Any, tracker: Any | None = None) -> Any:
        from distil_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig
        from sentence_transformers import SentenceTransformer

        from src.data.teacher_embeddings import DATASET_REPO, load_teacher_embeddings_dataset

        dataset_repo = self.params.get("dataset_repo", DATASET_REPO)
        output_dir = self.params.get("output_dir", "./distilled_model")

        hyperparameters = {
            **PAPER_HYPERPARAMETERS,
            **{k: v for k, v in self.params.items() if k in PAPER_HYPERPARAMETERS},
        }

        with tempfile.TemporaryDirectory() as student_dir:
            model.save(student_dir)

            config = EmbeddingTrainerConfig(
                student_model=student_dir,
                output_dir=output_dir,
                **hyperparameters,
            )
            trainer = EmbeddingDistillationTrainer(config)

            dataset = load_teacher_embeddings_dataset(dataset_repo)

            with _force_cosine_lr_schedule(), _patch_device_transfer(), _log_loss_to_tracker(tracker):
                metrics = trainer.train(dataset)

        logger.info("Distillation complete: %s", metrics)
        return SentenceTransformer(output_dir)
