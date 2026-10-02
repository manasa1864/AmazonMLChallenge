# Business Entity Resolution — pipeline

Matches every Source 1 business to its Source 2 / Source 3 records (ML Challenge 2026).
Everything runs offline on the provided data only: no external lookups, geocoders or APIs.
The only learned model is LightGBM (MIT licence, gradient-boosted trees, far below the 8B-parameter limit).

## Layout

```
src/
  text.py          normalisation (unidecode transliteration, abbreviation / state canonicalisation, phonetic skeleton)
  data.py          TSV loading + coherent down-sampled "mini world" of the training data
  blocking.py      candidate generation: per-country TF-IDF over name / skeleton / address tokens + address bigrams
  features.py      pair features (rapidfuzz similarities, extra-token / number features, rival competition, stage-2 context)
  decision.py      threshold + "each S2/S3 record belongs to at most one S1 entity" assignment
  metric.py        official macro F0.5
  run_validation.py  train + validate on a training sample, saves models   (step 1)
  run_test.py        apply the saved models to the test set, writes submission files (step 2)
  exp_blocking.py, eda.py, inspect_dbg.py   analysis helpers
utils/validate_submission.py   organiser's format checker
data/                          <- put the dataset here: data/train/*.tsv, data/test/*.tsv
```

## Reproduce end to end

```bash
pip install -r requirements.txt            # Python 3.12
export PYTHONPATH=$PWD/src                 # (PowerShell: $env:PYTHONPATH="$PWD\src")

# 1) train + validate (12% coherent sample of the training data, 2-fold out-of-fold by S1 entity).
#    Prints the official macro F0.5, tunes the decision threshold and saves models/m_*.txt + models/m.json.
python src/run_validation.py --frac 0.12 --save_model models/m

# 2) generate the submission from the test set
python src/run_test.py --model models/m --out output

# 3) check the format
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/test
```

Runtime on a 22-core / 31 GB Windows machine: step 1 about 20 min, step 2 about 1 hour
(1.7M S1 x 10M S2/S3 records; ~50M candidate pairs scored in chunks). Peak RAM around 20 GB.

## Validation results (12% coherent training sample, 2-fold out-of-fold by S1 entity, official macro F0.5)

| Variant | F0.5 |
|---|---|
| TF-IDF cosine only | 0.9039 |
| Logistic regression on the pair features | 0.9610 |
| LightGBM stage 1 | 0.9713 |
| **LightGBM stage 2 (submitted)** | **0.9785** |
| + S2<->S3 consensus features | 0.9775 |
| two-threshold "uncertain zone" rules | 0.9763 - 0.9775 |
| union with name / address / char-3gram blocking channels (`--channels`) | 0.9761 (recall up, precision and singletons down) |

## Method in one paragraph

1. **Normalise** names and addresses (transliterate every script to ASCII, expand Pvt/Ltd/Rd/St..., canonical state codes,
   drop legal-form words for a "core" name, phonetic consonant skeleton for transliteration-robust name keys).
2. **Block** within each country label (open set, so unseen countries such as France work): TF-IDF over name words,
   name skeleton, address words and adjacent address word pairs; top-30 cosine neighbours among S2/S3 per S1 record.
3. **Stage-1 matcher**: LightGBM on ~100 pair features (fuzzy name / address similarities, house-number and PIN
   agreement, extra words / numbers on either side, decoy-word flags, blocking-score context, competition between
   S1 rivals for the same S2/S3 record).
4. **Stage-2 matcher**: second LightGBM on the stage-1 probabilities' context (best / sum / count of the entity's other
   candidates, strength of rival S1 entities for the same record), which sharpens singleton detection and decoy rejection.
5. **Decision**: keep pairs with probability >= tuned threshold, then give each S2/S3 record only to its most probable S1
   entity (the ground truth never assigns one record to two S1 entities). The threshold maximises out-of-fold macro F0.5.
