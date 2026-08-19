# Session Notes — exp010 (Syllable-Based Compositional Embedding)

Read this alongside `SESSION_NOTES.md` and `SESSION_NOTES_EXP009.md`. exp009
(compositional root+suffix, morphology-based) is **closed** — final result
0.4065 (exp009g), documented in `SESSION_NOTES_EXP009.md`, not being
reopened. This file covers exp010, a related-but-distinct idea started
right after exp009 closed.

## What exp010 is, and why

The user's own follow-up idea after exp009 closed: same overall
"no fixed subword vocabulary, compose the embedding at runtime" concept as
exp009, but using a **different Turkish linguistic unit** — syllables
instead of morphological root+suffix chains. Motivation, discussed directly
with the user:

- **exp009's `zeyrek` morphological analyzer was a real, repeatedly-hit
  bottleneck** (slow per-word, ~10.5% OOV-fallback rate even at 60k vocab,
  a dictionary-lookup approach that can miss rare/novel words entirely).
- **Turkish syllable structure is governed by simple, highly regular rules**
  (a consequence of vowel harmony) — syllabification is pure deterministic
  string logic, no ML/dictionary lookup, no OOV concept at all, dramatically
  faster to compute than a morphological analyzer.
- **The honest tradeoff, flagged to the user before building anything**: a
  syllable carries no inherent semantic meaning on its own (unlike a
  morphological root, which is a real lexical unit tied to word meaning).
  This is NOT assumed to be a strictly easier version of exp009's idea —
  genuinely uncertain whether faster/cleaner segmentation offsets the loss
  of semantic anchoring. Both directions were stated plainly before starting
  (see "Real expectation, stated before building" below).

**Code**: `src/data/turkish_syllable.py` (`syllabify_word`,
`build_syllable_vocab`) and `src/model_cloning/compositional_syllable.py`
(`SyllableEmbedding` + `CompositionalSyllableStrategy`, registered
`"compositional_syllable"`). Architecture mirrors
`compositional_root_suffix.py`'s exactly (`SyllableEmbedding` is module
index 0 in a `SentenceTransformer`, doing word-splitting + syllable-id
lookup in `preprocess()` and composing `inputs_embeds` in `forward()`,
feeding into the same kind of randomly-initialized `BertModel` blueprint
body) — same working plumbing, different segmentation front-end.

## Turkish syllabification rule implemented (deterministic, TDK-consistent)

Every syllable contains exactly one vowel. For a run of N consonants
between two vowels: N<=1 joins the *following* syllable; N>=2, all but the
last consonant stay with the *preceding* syllable, the last joins the
*following* syllable (e.g. "ıspanak" -> "ıs-pa-nak"). Leading/trailing
consonants join the first/last syllable respectively. A word with no vowels
(digits, punctuation, acronyms) is treated as one syllable (itself), never
raises — same never-raise philosophy as `turkish_morphology.py`'s
`segment_word()`.

Verified locally against known TDK examples (`ıspanak` -> `ıs-pa-nak`,
`kitap` -> `ki-tap`, `elma` -> `el-ma`, `kitaplar` -> `ki-tap-lar`,
`Türkiye` -> `Tür-ki-ye`) — all correct.

## Composition operator: mean only, deliberately

`SyllableEmbedding` composes a word as `mean(syllable_embeddings)` —
**no order-aware operator implemented**, on purpose. Directly learned from
exp009l/exp009m (`SESSION_NOTES_EXP009.md`): jumping straight to a fancier,
order-aware operator (`kombo_fold`) on a from-scratch, randomly-initialized
composition made results *worse*, not better, even with a well-motivated
design and later an identity-init fix. This time, start simple, same
discipline exp009's *first* smoke test used — an order-aware follow-up is
only worth considering if this simpler version already beats exp009g's
0.4065 in a real single-variable comparison.

## Verified locally (pure Python/torch, no downloads — per standing "no
local heavy downloads" preference)

All in one throwaway script, no `zeyrek`, no corpus download:
- `syllabify_word()` correctness against 5 known TDK examples + empty-string
  and no-vowel edge cases.
- `build_syllable_vocab()` on a tiny synthetic corpus — correct word/
  syllable counts, correct vocab size.
- `SyllableEmbedding.forward()` shape/dtype correctness, gradient flow into
  `syllable_embeddings`.
- Exact mean-composition value check (`mean(ki, tap)` for "kitap" matches
  bit-for-bit).
- Save→reload round-trip numeric equality.
- `preprocess()` end-to-end on real strings (no corpus needed — pure string
  splitting + syllabification).

All checks passed on first run.

## New configs

- `configs/experiments/exp010_compositional_syllable_smoke.yaml` — cheap
  plumbing check (tiny vocab, 2-layer body, 1% data), same discipline as
  every prior smoke test in this project. `colab/run_experiment_colab.py`'s
  `CONFIG_PATH` currently points here — **run this first**.
- `configs/experiments/exp010b_compositional_syllable_30pct.yaml` — real
  comparison against exp009g: 30% data, 4-layer body, lr=2e-5, MSE, mean
  composition. NOT byte-identical to exp009g the way exp009l/m were to each
  other — the vocab concept differs (small closed syllable inventory vs. a
  capped-at-60k open root vocabulary), so check `vocab_stats.json`'s real
  `syllable_vocab_size` before assuming this is "the same scale" as exp009g.
  Run only after the smoke config passes.

## Real expectation, stated before building (calibration, not hindsight)

Discussed directly with the user before writing any code: genuinely
uncertain whether this beats exp009g's 0.4065, and if anything a slight
lean toward **not** beating it on the first attempt — a syllable is a
phonological unit with no inherent meaning, closer to character-level
modeling than morpheme-level, and character/syllable-level models
generally need *more* training to reach the semantic quality that
morpheme-aware models get more cheaply, not less. The real advantage is
speed/completeness of segmentation (no zeyrek bottleneck, no OOV concept),
not obviously better final embedding quality. Worth testing because it's
cheap to find out, not because a win is expected.

## Not yet run

Neither the smoke config nor exp010b has been run on Colab yet. Next
session should run the smoke config first, confirm the go/no-go signals
(vocab builds fast and looks sane, pipeline runs to completion, no
crashes), then run exp010b and compare `mteb_mean_score` against exp009g's
0.4065 and the zero-distillation floor (~0.289).
