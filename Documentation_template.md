# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Semantic Error
**Team Members:** Manasa Gopisetty (Team Leader), Ishita Punjani, Aditi Garg, Tanvi Paithankar
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We match each Source 1 business to zero or more Source 2/Source 3 records with a blocking + two-stage
LightGBM classifier pipeline. Text is normalised across scripts and countries (including French, unseen in
training) before candidate generation. The key lesson driving our final design: a matcher validated on a
small down-sampled world scored 0.9785 locally but only 0.903 on the leaderboard; re-validating at the
true test scale (every training S1 entity against the full candidate pool) closed most of that gap and
raised the leaderboard score to our final, submitted **0.91818**. After submission we continued the error
analysis and found and validated a further, consistent improvement (a label-free "name ambiguity"
feature); it is not part of the submitted model and is documented as immediate future work (section 5.5),
not claimed as part of this entry's score.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale dominates.** Training has 2.2M Source 1 records against a combined ~10.3M Source 2/3 pool; test
  is a similar order of magnitude. All-pairs comparison is infeasible; blocking quality caps everything
  downstream (candidate recall at our chosen top-30 blocking width is ~93% at full scale).
- **Noise matches the brief exactly**: legal-suffix variants (Pvt/Private, Ltd/Limited), punctuation (`&`
  vs `and`), word-order transpositions, typos, abbreviated street types, missing PIN/state, landmark
  references, reordered address components.
- **Script diversity.** A meaningful share of Indian Source 2/3 names are in Devanagari, Kannada, Gujarati,
  Tamil, Odia, etc.; matching against Latin-script Source 1 names requires transliteration, and
  off-the-shelf transliteration (`unidecode`) is lossy enough that pure name-token overlap under-recalls
  these pairs — the single largest source of blocking misses in our error analysis.
- **France (test-only, unseen in training).** ~15% of test Source 1 entities. Address sharing is far
  higher than in training countries: 73% of French Source 1 records share a street+city with another
  French Source 1 record, vs. 12% (US) and 2% (India) in training — i.e. the training distribution
  under-represents how ambiguous an address can be as a signal, which we address with a label-free
  address-ambiguity feature that works without needing French labels.
- **Deliberate decoys.** Manual review of ground-truth non-matches found engineered near-misses: identical
  address with a different house number, an extra legal-form/holding word added to an otherwise-identical
  name, single-character-substituted names ("Prairie Grand" vs "Prairie Gtrdh").
- **Missing fields**: ~3.3% of Source 2/3 addresses are empty; this interacts badly with generic business
  names (see 5.2).
- **Records are exclusive**: every ground-truth Source 2/3 id appears under at most one Source 1 entity —
  informs the final assignment rule (4, "Decision").

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier (LightGBM), pairwise, with a
final "each Source 2/3 record goes to at most one Source 1 entity" assignment step.

**Core Innovation:** Two things outside the standard blocking+classifier recipe:
1. **Validating at true test scale before trusting any number.** A model tuned on a 12% down-sampled
   world looked excellent locally (F0.5 0.9785) but scored only 0.903 on the leaderboard. Retraining and
   re-thresholding against the *entire* training set as queries against the *entire* candidate pool (not a
   sample) recovered nearly all of that gap (0.918 leaderboard) purely from correct calibration and
   candidate-competition density — no new features. See section 5 and the ablation log in the appendix.
2. **Label-free ambiguity features.** Two features count, per country and using only the input files (no
   ground truth), how many other records share this record's exact address / street+city / normalised
   name. These generalise to the unseen test country (France) without requiring any French training data,
   and the name-ambiguity feature was found and confirmed to help via manual inspection of the deployed
   model's actual false positives (see 5.2), not by trying a generic technique speculatively.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** per-country TF-IDF cosine similarity (never comparing across country labels,
  but the code treats country as an open string set — no hard-coded {US, India}), over a token bag built
  from:
  - normalised name words (legal-form/stopword-stripped "core" name)
  - a phonetic consonant skeleton of the name (collapses vowels and common consonant confusions —
    `ph→f, ck→k, x→ks, c→k, q→k, w→v, z→s` — so transliterated non-Latin names and vowel typos land on
    the same key as their Latin counterpart)
  - normalised address words
  - adjacent address-word bigrams (captures house-number / city-name pairs, e.g. `88_5`, `bangalore_north`)
  - top-30 nearest neighbours per Source 1 record retained.
- **Candidate pairs generated:** ~30.0 per Source 1 entity on average (52.0M pairs for the 1,732,544-row
  test set; 66.2M pairs when blocking the full training set for validation).
- **How we ensured true matches were not lost:**
  - Candidate (pair) recall against training ground truth measured at the *same scale* as test — 93.1%
    overall (India 91.4%, US 94.2%) — with an oracle F0.5 ceiling of 0.974 given those candidates, so the
    matcher operates close to (measured ~95–98% of) that ceiling rather than being blocking-bound.
  - Tried and rejected: unioning three extra retrieval channels (name-only, address-only, character
    3-gram) raised recall to 94.0% (ceiling 0.978) but the 46% larger candidate set fed the matcher more
    decoys than it gained in recall, giving a *lower* validated F0.5 — not deployed.
  - Tried and rejected: doubling the main channel's width (K=30→60) raised the recall ceiling by only
    +0.31pt for 2x the candidates — not worth the compute or the extra decoy exposure.
  - `candidate_pairs.tsv` records the exact candidate set the classifier scored (so blocking quality is
    independently auditable), and every id in `matching_results.tsv` is a subset of it.

---

## 4. Matching Model

**Features used** (~110 total; full list and code in `src/features.py`):
- **Name features:** Levenshtein-family ratios (`rapidfuzz`: ratio, token-sort, token-set, partial),
  Jaro-Winkler, Jaccard on tokens, first-token-only Jaro-Winkler, phonetic-skeleton ratio/token-set/Jaccard,
  length ratio, count and fraction of tokens present on one side only (decoy-word signal).
- **Address features:** the same string-similarity family (skipped/-1-sentinelled when empty), house/PIN
  number agreement (first number equal, Jaccard of all numbers, PIN/ZIP overlap), an "address-ambiguity"
  feature (records per million sharing this exact address / this street+city, computed separately for the
  Source-1 side and the candidate side, label-free).
- **Other:** non-Latin-script flag; blocking cosine score, its rank and its ratio to the entity's best
  candidate; count of candidates for this entity; rank and score-gap of this candidate against *rival
  Source-1 entities* competing for the same Source 2/3 record (a genuine Source 2/3 record belongs to at
  most one Source 1 entity, so this competition is informative); a second-stage set of features built from
  the first stage's own predicted probabilities (best/sum/count of an entity's other candidate scores, and
  the strength of the record's strongest rival Source-1 claimant). *(A "name-ambiguity" feature was also
  explored and validated, but only after this model was submitted — see section 5.5 — so it is not among
  the features above.)*

**Model type:** LightGBM (gradient-boosted decision trees; MIT licence, well under the 8B-parameter cap),
in two stages:
1. **Stage 1** scores every candidate pair from the ~110 base features.
2. **Stage 2** rescoring adds "context" features derived from stage 1's own probabilities (described
   above) and produces the final probability. Both stages are 2-fold, out-of-fold by Source 1 entity (no
   entity is ever scored by a model that trained on it).

At inference, both fold models of a stage are averaged.

**Threshold selection method:** grid + local refinement search over the decision threshold, maximising the
*official* out-of-fold macro F0.5 (not accuracy/F1/AUC), always measured at test scale. A candidate
alternative — a stricter threshold specifically for Source 1 entities whose address is shared with other
entities (a label-free, country-agnostic proxy for "France-like" ambiguity, so it needs no French training
data) — was tested but gave no measurable improvement over a single global threshold and was not deployed.

**Decision rule:** keep pairs with probability ≥ threshold, then, since ground truth never assigns one
Source 2/3 record to two Source 1 entities, give each record only to its highest-probability Source 1
claimant (a small but consistent + ~0.0002–0.0014 F0.5 gain over not doing this).

---

## 5. Results & Error Analysis

### 5.1 Headline numbers

| Stage | What changed | Held-out F0.5 (macro) |
|---|---|---|
| v1 | Two-stage LightGBM, validated on a 12% down-sampled world | 0.9785 (local) / **0.903** (leaderboard) |
| v2 | Same architecture, retrained + re-thresholded at *true test scale* | **0.91797** (leaderboard) |
| **v3 — submitted** | More training data for the matcher (10%→18% of Source 1 entities) | 0.9515 (test-scale held-out) / **0.91818 (final leaderboard score)** |

v3 is the model that produced this entry's `output/matching_results.tsv` and leaderboard score; it is what
`code/business_entity_resolution/` reproduces exactly. A further feature (name-ambiguity, section 5.5) was
validated *after* this submission and is documented as immediate future work, not part of the submitted
model or its score.

The v1→v2 gap is the central finding of our process: F0.5 measured on a smaller world is systematically
optimistic, because candidate competition (how many similar-looking records a Source 2/3 record and a
Source 1 entity each have to compete against) is density-dependent and only realistic at true scale. Every
subsequent change in this project was validated at full scale (`src/run_fullscale.py`: every training
Source 1 entity queried against the complete pool) specifically to avoid repeating that mistake.

### 5.2 Common false positives (wrong merges)

A structured review of the submitted (v3) model's actual false positives (not a generic literature
technique) found that **77% of them shared one trait: the wrongly-matched candidate had a completely empty
address**. With no address to corroborate or refute, the model over-trusted name similarity alone for
common/generic business names — e.g. one "Mountain Alliance" wrongly matched to an unrelated "Mountain
Alliance LLC", one "Modern Media Private Limited" matched to a different company with the identical name.
This finding led to the name-ambiguity feature described in section 5.5 — validated, but produced *after*
this entry was submitted, so it is not reflected in v3's features or score above.

Other false-positive patterns, addressed earlier in development: identical street address with a
different house number; an extra legal-form or holding-company word added to an otherwise-matching name
(handled by the extra-token-count features in 4).

### 5.3 Common false negatives (missed matches)

- **Transliteration quality.** The largest remaining recall loss: Devanagari/Kannada/Gujarati/Tamil/Odia
  names whose ASCII transliteration (via `unidecode`) doesn't overlap enough with the Latin-script Source
  1 name's tokens or phonetic skeleton to rank in the retrieved top-30, especially when combined with a
  short or missing address. This is a genuine blocking-recall ceiling, not a matcher problem — see 3 for
  the (rejected, cost > benefit) attempts to relax it.
- **OCR/typo-style corruption** in source data itself (e.g. "Mumbai" → "Marlbmai", digit-for-letter swaps)
  occasionally drags the name-similarity score for a genuine match below threshold, particularly when
  combined with a differing house number (which is otherwise a strong, correctly-weighted decoy signal —
  see 5.2's opposite failure mode). We judged this an acceptable, roughly symmetric trade-off rather than
  a one-sided bug, given F0.5's precision weighting.

### 5.4 Ideas tested and explicitly rejected (with evidence)

For transparency: the following were implemented and evaluated with proper held-out validation, and not
deployed, because they did not survive that evaluation.

| Idea | Result |
|---|---|
| Extra blocking channels (name/address/char-3gram union) | Recall +1pt, ceiling +0.4pt, but F0.5 *lower* (more decoys than gains) |
| Two-threshold "uncertain zone" rules | ≤ the single-threshold F0.5 in every variant tried |
| S2↔S3 "consensus" features (candidate similarity to an entity's other candidates) | No improvement (0.9775 vs 0.9785) |
| Label-free per-country word-frequency "decoy word" feature | +0.0005 AUC only — noise |
| Country-conditioned rank-normalised features | +1.5pt on one random sample, *negative* on a second — rejected as sample-specific overfitting |
| Self-training on the unseen country's own confident pseudo-labels, anchored on labelled source data | Leaderboard score *fell* (0.91818 → 0.91734) |
| Multilingual sentence embeddings (`multilingual-e5-small`, MIT-licensed) as a semantic similarity feature | No consistent gain in either direction of an unseen-country transfer test; full-corpus embedding was separately measured infeasible on commodity CPU hardware (~375 sentences/sec → 8-10+ h) |
| Doubling blocking width (K=30→60) | Ceiling +0.31pt only, for 2x the candidate volume — not pursued |

### 5.5 Future work: a validated improvement not yet in the submitted model

After submitting v3, continued error analysis (5.2) found a specific, fixable pattern and a feature to
address it: a label-free "name-ambiguity" column (records per million, within a country, sharing this
exact normalised name — the same recipe already used for the address-ambiguity feature in section 4), plus
its interaction with "candidate address is empty". Unlike every rejected idea in 5.4, this one was positive
and consistent across three independent checks, never flipping sign:

| Test | Baseline F0.5 | + name-ambiguity | Gain |
|---|---|---|---|
| Small held-out sample A | 0.9624 | 0.9691 | +0.0067 |
| Small held-out sample B (different random world) | 0.9807 | 0.9836 | +0.0029 |
| Full test-scale training (all 2.2M Source 1 entities, 2-fold out-of-fold) | 0.9515 | 0.9529 | +0.0014 |

This was deliberately **not** substituted into the submitted model or score: it was found and validated
after v3 had already been submitted, and we chose to document it transparently as the clear next step
rather than quietly swap it into this write-up's headline numbers. The function (`add_name_ambiguity()` in
`src/features.py`) ships in the code package for reference but is not called by the submitted pipeline
(`run_fullscale.py` / `run_test.py`); `src/exp_name_amb.py` reproduces the validation above.

---

## 6. Conclusion

Our submitted solution (v3) is a TF-IDF blocking + two-stage LightGBM classifier pipeline, engineered
around one central discipline: every design decision was validated at the true scale and competitive
density of the actual test problem, because smaller-scale validation was demonstrably (and substantially)
optimistic — this discipline alone closed most of the gap between our first (0.903) and submitted (0.91818)
leaderboard scores. Several further ideas (extra blocking channels, semantic embeddings, self-training,
rank-normalised features) were implemented, honestly evaluated, and correctly discarded when held-out
evidence did not support them (5.4) — which we consider as much a result of this process as the features
that were kept. One further idea, found by manually inspecting the submitted model's real mistakes rather
than by trying generic techniques speculatively, gave a small but consistent, three-times-confirmed gain
(5.5); we report it transparently as validated future work rather than retroactively folding it into this
entry's score.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline ships under `code/business_entity_resolution/` (`src/`, `README.md`,
`requirements.txt`). Entry points:

```bash
python src/run_fullscale.py --fracT 0.18 --save_model models/g   # train + validate at test scale
python src/run_test.py --model models/g --out output              # produce the two submission files
```
Format-checked with the organiser-provided `utils/validate_submission.py` from the challenge's own
`student_resource/` package (not shipped in this zip — it is the organisers' own script, run against this
run's `output/` folder).

Analysis/ablation scripts behind section 5.4's table (`exp_*.py`, `error_scan.py`, `diag_*.py`,
`probe_k.py`, `france_selftrain.py`, `build_source.py`, `make_france_variants.py`) are included for
reproducibility but are not part of the production pipeline.

### B. Additional Results

- Candidate recall at full test scale, main blocking channel only: **93.1%** overall (India 91.4%, US
  94.2%); oracle F0.5 ceiling given those candidates: **0.974**.
- Final model's macro F0.5 reaches ~95–98% of that ceiling on held-out training data at test scale,
  indicating the matcher, not blocking, is close to fully exploiting the available candidate signal for
  the countries present in training.
- Test-set composition: India 46.8%, US 38.3%, France 15.0% of Source 1 entities (France unseen in
  training).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
