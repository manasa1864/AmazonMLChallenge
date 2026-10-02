# Business Entity Resolution — reproduction guide

Matches every Source 1 business to its Source 2 / Source 3 records. Offline only: no external lookups,
geocoders or APIs. The only learned model is LightGBM (MIT licence, gradient-boosted trees, far under the
8B-parameter cap).

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src            # PowerShell: $env:PYTHONPATH="$PWD\src"
```

Place the challenge data in a `dataset/` folder next to this `README.md` (i.e. a sibling of `src/`), laid
out exactly as the challenge provides it:
```
dataset/train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
dataset/test/test_source1.tsv, test_source2.tsv, test_source3.tsv
```
No code changes needed — `src/data.py`'s `DATA` constant already points at this `dataset/` folder.

## Reproduce end to end

```bash
# 1) train + validate at TEST SCALE (every train S1 entity vs the full S2/S3 pool; a small-sample validation
#    is optimistic and does not transfer — see Documentation_template.md, section 5).
#    Prints the official macro F0.5, tunes the decision threshold and saves models/g_*.txt + models/g.json.
python src/run_fullscale.py --fracT 0.18 --save_model models/g

# 2) generate the submission from the test set
python src/run_test.py --model models/g --out output

# 3) check the format using the organiser-provided validator (utils/validate_submission.py from the
#    challenge's own student_resource/ package - not shipped in this zip; run it from that directory,
#    pointing --matching/--candidate at this run's output/ folder)
python utils/validate_submission.py --matching /path/to/output/matching_results.tsv \
    --candidate /path/to/output/candidate_pairs.tsv --test-dir dataset/test
```

Runtime on a 22-core / 31 GB Windows machine: step 1 ~1.8h, step 2 ~2h (1.7M S1 x 10M S2/S3 records;
~52M candidate pairs scored in chunks). Set the environment variable `AML_JOBS` (e.g. `8`) to cap worker
processes / threads on a smaller or thermally-limited machine, at the cost of a longer run.

## Source layout (`src/`)

| File | Role |
|---|---|
| `text.py` | normalisation: transliteration (unidecode), abbreviation / US+India state canonicalisation, French street/legal-form handling, phonetic consonant skeleton |
| `data.py` | TSV loading + a coherent down-sampled "mini world" of the training set for fast iteration |
| `blocking.py` | candidate generation: per-country TF-IDF over name / skeleton / address tokens + address bigrams, top-K cosine |
| `features.py` | pair features: rapidfuzz string similarities, extra-token / extra-number counts, address- and name-ambiguity (label-free "how common is this address/name"), blocking-score context, rival-S1 competition, stage-2 context-of-stage-1-probabilities |
| `decision.py` | threshold (optionally per-pair) + "each S2/S3 record belongs to at most one S1 entity" assignment |
| `metric.py` | official macro F0.5 |
| `run_fullscale.py` | **main entry point**: trains + validates at test scale, saves the model |
| `run_test.py` | applies a saved model to the test set, writes the two submission files |
| `run_validation.py` | earlier, smaller-scale trainer (superseded by `run_fullscale.py`; kept for reference / fast iteration) |
| `exp_*.py`, `error_scan.py`, `diag_*.py`, `probe_k.py`, `france_selftrain.py`, `build_source.py`, `make_france_variants.py` | analysis / ablation scripts behind the choices in `Documentation_template.md`, including post-submission exploratory work (section 5.4 "Future work") — **not part of the submitted pipeline**, not required to reproduce `output/` |

`features.py` additionally defines `add_name_ambiguity()`, used only by `exp_name_amb.py`: a feature explored
*after* the submitted model was trained (see Documentation_template.md, "Future work"). It is not called by
`run_fullscale.py` / `run_test.py`, so it has no effect on the submitted `output/` or on reproducing it.

## Method in one paragraph

1. **Normalise** names and addresses (transliterate every script to ASCII, expand Pvt/Ltd/Rd/St/French
   street types..., canonical state codes, drop legal-form words for a "core" name, phonetic consonant
   skeleton for transliteration-robust name keys).
2. **Block** within each country label (open set, so unseen countries such as France work): TF-IDF over
   name words, name skeleton, address words and adjacent address word pairs; top-30 cosine neighbours
   among S2/S3 per S1 record.
3. **Stage-1 matcher**: LightGBM on ~110 pair features (fuzzy name / address similarities, house-number and
   PIN agreement, extra words / numbers on either side, address- and name-ambiguity, blocking-score
   context, competition between S1 rivals for the same S2/S3 record).
4. **Stage-2 matcher**: second LightGBM on the stage-1 probabilities' context (best / sum / count of the
   entity's other candidates, strength of rival S1 entities for the same record), which sharpens singleton
   detection and decoy rejection.
5. **Decision**: keep pairs with probability >= the tuned threshold, then give each S2/S3 record only to
   its most probable S1 entity (the ground truth never assigns one record to two S1 entities).

See `Documentation_template.md` for the full write-up, including the ablations that were tried and
rejected (extra blocking channels, semantic embeddings, self-training, per-country rank features).
