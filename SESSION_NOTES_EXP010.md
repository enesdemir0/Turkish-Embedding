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

## Bug found and fixed: uncapped syllable vocab exploded to near-word-scale

First real Colab smoke run (uncapped `build_syllable_vocab`, no
`max_syllables`): **727,154 distinct "syllables"** from a corpus with only
4,042,857 distinct words — nearly word-scale, not the small, reusable
inventory Turkish's regular phonology should produce. `727,154 × 768 ≈
558M parameters` just for the syllable lookup table — bigger than the
teacher model (300M), completely defeating the "small, cheap model" point
of this idea. The smoke run still completed without crashing and even
scored `mteb_mean_score = 0.40184` — dangerously close to exp009g's real,
fully-trained result, from only 23 training steps on 1% data. **This score
was diagnosed as not real signal**: with the syllable table nearly
one-to-one with distinct words, the "model" was closer to a giant
per-word lookup table (far more capacity than intended) than a genuine
small compositional model, so of course a wildly-oversized model can
partially fit data even in 23 steps — this says nothing about whether the
syllable-composition idea itself is good.

**Root cause**: `syllabify_word()` operated on raw corpus tokens with no
normalization — punctuation, casing, and digits stuck to words (`"kitap,"`
vs `"kitap."` vs `"Kitap"` vs `"kitap123"`) each produced a different,
non-reusable syllable string. Cosmos (a large, noisy, web-scraped corpus)
has enough of this long tail to nearly double the distinct-string count.

**Fix, part 1 (normalization)**: added `_normalize_word()` to
`turkish_syllable.py` — Turkish-aware lowercase (confirmed Python's own
`str.lower()` is wrong for Turkish: `"İ".lower()` produces a stray
combining-dot artifact, `"I".lower()` produces `"i"` when it should produce
dotless `"ı"` — built an explicit translation table instead) plus stripping
every non-Turkish-letter character. `syllabify_word()` now normalizes
before segmenting, so vocab-building and runtime inference automatically
stay in sync (single source of truth, no separate call sites to keep
consistent). Re-ran on Colab: **727,154 → 173,715** — real, ~4.2x
reduction, confirms normalization was a genuine part of the problem.

**Fix, part 2 (frequency-ranked capping — the actual fix)**: 173,715 is
still far too large — normalization alone can't fully separate real Turkish
syllables from non-Turkish/foreign/junk content that happens to use only
Turkish-alphabet letters (e.g. an English word sails through the letter
filter untouched). Rather than chase every possible contamination source
(endless whack-a-mole), applied the same fix exp009 already needed for its
own root vocab: frequency-ranked capping via the existing (previously
unused) `max_syllables` param — real Turkish syllables dominate the head of
a Zipfian frequency distribution, junk sits in the long tail, so capping to
the N most frequent syllables cuts the junk without needing perfect text
cleaning. Also added a new stat, `syllable_oov_fraction` — the syllable-
level analogue of `build_root_suffix_vocab`'s `oov_fallback_fraction`:
fraction of real syllable *occurrences* (not distinct types) that fall
outside the cap. Verified locally (synthetic corpus, exact math check):
capping to the 2 most frequent syllables in a 2-word corpus correctly
computes `oov_fraction = 2/22`.

**Configs updated** to pass explicit caps: `exp010`'s smoke config now
builds with `max_syllables=3000`; `exp010b`'s real-comparison config with
`max_syllables=5000`. Both configs' header comments document the real
727,154/173,715 numbers as the reason this is required, not optional.

## Corrected smoke run: passed vocab checks, but the score raised a new,
more important question than the vocab bug did

Ran the corrected smoke config (`max_syllables=3000`, normalization fix
live): `syllable_vocab_size=3001`, `syllable_oov_fraction=0.0108` (1.08% —
excellent coverage, well under exp009g's 10.5% root-fallback rate). The
vocab-size bug is genuinely fixed.

But `mteb_mean_score` landed at **~0.41** — NOT near exp009's ~0.29-0.31
smoke-test floor (the correct apples-to-apples reference: same 23 steps,
same 1% data, same 2-layer body). That's a real, unexpected gap worth
investigating on its own, separate from the vocab-size bug.

## exp010c — zero-distillation ablation (same methodology as exp003):
the real, final finding

Built `configs/experiments/exp010c_compositional_syllable_zero_distillation.yaml`
(`distillation.strategy: skip`, same exact model/vocab as the smoke test,
zero training at all) to isolate how much of that ~0.41 was learned versus
architectural. **Result: 0.4113 — statistically the same as the trained
smoke test's ~0.41.** The 23 training steps taught the model essentially
nothing; the score is coming from the architecture itself.

**Diagnosis**: with only 3,001 syllables shared across millions of words
(vs. exp009's 16,838 roots, close to one-per-lexical-item), even
completely untrained mean-of-syllable-vector composition captures a real
"bag of sub-words" lexical-overlap signal — two sentences sharing common
words/syllables get similar mean-pooled vectors purely from vocabulary
overlap, regardless of whether anything was learned. Several MTEB tasks
(STS, retrieval, some classification) partially reward exactly this kind
of overlap. This is a known category of effect (a naive averaging baseline
scoring non-trivially on certain benchmarks without real understanding),
not a code bug — the vocab, composition math, and training loop are all
confirmed correct (see local unit tests above and the real, working
zero-distillation run itself).

**Why this matters, stated plainly**: `0.4113` (zero training) is *higher*
than exp009g's `0.4065` (the best real, fully-trained result from the
entire morphology-based track). This means "beating exp009g's number" is
not a meaningful target for the syllable approach — an untrained model
already clears it by doing nothing. The real bar for any future syllable-
based training result would have to be ~0.41-0.42, not 0.4065 — a
significantly higher, harder bar than originally understood when exp010b
was designed.

## FINAL STATUS: exp010 concluded — real, negative-but-honest finding,
not run further

**exp010b (the real 30%-data comparison) was never run.** Given the
zero-distillation finding, running it would spend the largest remaining
chunk of a tight 100-Colab-unit budget on a result that couldn't be
cleanly interpreted as "real learning" even if the raw number looked good —
the same ambiguity exp008's BERTurk result already taught this project to
watch for. Decided, discussed directly with the user, not to spend that
budget chasing it.

**What this session actually produced, for a future session's benefit**:
- A working, tested Turkish syllabification implementation
  (`src/data/turkish_syllable.py`) — deterministic, fast, Turkish-aware
  casefold, correctly handles real-world corpus noise via frequency
  capping. Reusable for any future idea that needs syllable-level Turkish
  text processing, independent of this specific compositional-embedding
  application.
- A real, well-diagnosed negative result: composing word embeddings from a
  small, highly-shared unit (syllables) produces a benchmark score that is
  substantially an artifact of lexical overlap, not learned semantics, at
  this scale — a genuine, transferable finding about *why* exp009's
  morphological roots (a much more word-specific unit) were a better
  choice of decomposition unit than syllables, now confirmed empirically
  rather than just theoretically predicted.
- Two real, caught-before-they-cost-real-budget bugs (the uncapped vocab
  explosion, and the misleading high smoke score) — both caught via cheap
  diagnostics (a coverage stat, a zero-distillation ablation) before the
  expensive real run, exactly the cheap-first discipline this project has
  used throughout.

Not being reopened without an explicit new idea for a different
composition unit or fix — same treatment as exp009's closure.
