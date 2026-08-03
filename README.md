# Turkish-Embedding

A modular framework for building Turkish sentence-embedding models via cross-lingual
tokenizer surgery + offline distillation, following and then extending
"Adapting Multilingual Embedding Models to Turkish via Cross-Lingual Tokenizer Surgery
and Offline Distillation" (arXiv:2605.29992). See `EMBEDD~3.MD` for a full breakdown of
the paper's method and `TURKIS~1.MD` for a survey of extensions this framework is
designed to support later (FOCUS/WECHSEL embedding init, morphology-aware tokenizer
surgery, contrastive fine-tuning, Matryoshka loss, ...).

## Why this is structured as a framework, not a script

Every stage of the pipeline is a swappable strategy, selected by name from a YAML
config:

| Stage | Package | First strategy (paper reproduction) |
|---|---|---|
| Tokenizer surgery | `src/tokenizer_surgery/` | `frequency_pruning` |
| Embedding initialization | `src/embedding_init/` | `mean_composition` |
| Distillation objective | `src/distillation/` | `cosine` |
| Evaluation | `src/evaluation/` | `mteb_tr_runner` (fixed — the ~26-task Turkish MTEB suite) |

Adding a new strategy later (e.g. `embedding_init/focus_init.py`) means writing one
file that registers itself with that package's registry — `src/pipeline.py` and every
other stage are untouched. See each package's `base.py` for the interface a new
strategy must implement.

## Experiment tracking (DagsHub + MLflow)

All runs log to one MLflow experiment (`turkish-embedding`) on
[DagsHub](https://dagshub.com/enesdemir0/Turkish-Embedding), so every experiment can be
compared side-by-side rather than living in separate silos.

**Run naming convention:** `exp{NNN}_{teacher_model}_{tokenizer_strategy}_{embedding_init}_{distill_objective}`
— e.g. `exp001_embeddinggemma300m_frequency_pruning_mean_composition_cosine`. The
teacher model is included because it's itself something you might swap later (not
just the tokenizer/init/objective). **This name is auto-derived from the config, not
hand-typed** (`src/tracking/naming.py`) — leave `tracking.run_name` unset in a config
and it's generated for you, so the name can never drift from what the run actually
did. The same fields (`teacher_model`, `tokenizer_strategy`, `embedding_init`,
`distill_objective`, plus `git_commit`) are also set as MLflow **tags** automatically —
those are what you filter/sort by once there are many runs; `tracking.tags` in a
config is only for extra one-off tags.

**Setup:**
1. Get a token: https://dagshub.com/user/settings/tokens
2. Copy `.env.example` to `.env` and fill in `DAGSHUB_USER_TOKEN` (already gitignored).
3. Run `python scripts/smoke_test_tracking.py` — confirms connectivity and that a run
   named `exp000_smoketest` shows up correctly tagged on DagsHub, before any real
   pipeline code runs.

## Running an experiment

```bash
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/experiments/exp001_mean_composition_cosine.yaml
```

## Status

- [x] Step 1 — tracking/config plumbing, pipeline skeleton (this commit)
- [ ] Step 2 — tokenizer surgery (`frequency_pruning`)
- [ ] Step 3 — model cloning + `mean_composition` embedding init
- [ ] Step 4 — `cosine` distillation objective + teacher-embeddings data loader
- [ ] Step 5 — `mteb_tr` evaluation runner (replaces `benchmark_classification_colab.py`)
- [ ] Step 6 — `exp001` run: exact paper reproduction, verified against published STSbTR
      numbers (Pearson 0.8199 / Spearman 0.7980)
