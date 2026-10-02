# Where things stand (paused on 2026-09-26)

## Submissions (all safe on disk)
| Folder | What | Leaderboard |
|---|---|---|
| `output_v1_scored_0.903/` (read-only) | first pipeline, trained on a 12% world | 0.903 |
| `output_v2_scored_0.91797/` (read-only) | scale-matched training (`models/f`) | 0.91797 |
| `output/` | currently identical to v2 | 0.91797 |

## Models trained (all saved in `models/`)
- `models/m`  - first pipeline (12% world).  Old feature set; cannot be used with the current code.
- `models/f`  - full-scale training, fracT 0.10. Held-out F0.5 0.9489 at test scale. Produced v2.
- `models/g`  - full-scale training, fracT 0.18. Held-out F0.5 **0.9515** (US 0.9594, India 0.9397). NOT yet applied to the test set.
  Uses the current code (French cleanup in `text.py`), single threshold 0.66.

## NEXT STEP: generate test predictions with `models/g`
Runs ~2 hours on all cores (~3-4 h with the cap). Keep the laptop plugged in and awake (sleep breaks the worker pool).
```powershell
cd C:\Users\Manasa\Documents\aml-er
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUNBUFFERED="1"; $env:PYTHONPATH="$PWD\src"
$env:AML_JOBS="8"          # optional: cap workers / threads so the laptop runs cooler; remove for full speed
python src\run_test.py --model models\g --out output_v3
python utils\validate_submission.py --matching output_v3\matching_results.tsv --candidate output_v3\candidate_pairs.tsv --test-dir data\test
```
Then compare `output_v3` with v2, and only if you want it copy it over `output/`. Never write into the read-only backup folders.

## Findings worth remembering
- Validation on a small sample (12%) was far too optimistic (0.9785) versus the leaderboard (0.903). Always validate at test scale
  (`src/run_fullscale.py`, all 2.2M train S1 vs the full 10M pool).
- Candidate recall at test scale is only ~93% (India 91.4%, US 94.2%); F0.5 ceiling given candidates 0.974.
- Extra blocking channels (`--channels`): recall 93.0% -> 94.0%, ceiling 0.974 -> 0.978, but 45% more candidates and ~2x compute; not adopted.
- Ambiguity-conditioned threshold (stricter for S1 whose street+city is shared, i.e. France-like): no gain on training data, so not used.
- France (15% of test) has no training data; 73% of French S1 share a street+city with another S1 (US 12%, India 2%).
- Background jobs need a generous timeout (a 3 h limit killed one run). Parallel workers can crash under memory pressure or sleep.

## Still to do for the final submission package
- team name + members, `<team_name>_submission.zip` = `output/` (2 TSVs) + `code/business_entity_resolution/{src,README.md,requirements.txt}` + filled `Documentation_template.md`.
- README.md in this folder describes the pipeline; update its results table with the full-scale numbers above.

## Update 2026-09-26 (later): leaderboard results and per-country diagnosis
- v3 (`models/g`) scored **0.91818** (v2 0.91797): essentially no gain. Saved read-only as `output_v3_scored_0.91818/`.
- Per-country diagnostic (v3 with ONE country's rows left empty; only that country changes): France-empty 0.801, India-empty 0.548, US-empty 0.544.
  Drops: France 0.117, India 0.370, US 0.374 (they add up, with the singleton credit ~0.056, to the v3 score -> the method is sound).
  Splitting by country needs the country mix of the scored subset, which is not the full-test mix (global shares would give US > 1.0).
  If US/India perform as in validation (0.959 / 0.940), France is about 0.76-0.79; if they are weaker, up to ~0.85. Either way France is the weakest country.
- Train vs test inputs are nearly identical for India and US (non-Latin share, empty addresses, lengths). The only shift: test has ~24% more S2+S3 records per S1.
- Rejected this round: label-free "decoy word" ratio feature (+0.0005 AUC only), France-only threshold tweaks (France predictions barely move).
- In progress when paused: `src/exp_domain.py` (out_exp_domain.txt): unseen-country transfer test (train US -> score India and back) comparing LightGBM,
  RandomForest, LR, kNN, per-country rank features and self-training.  Use its result to pick the France technique.
- Diagnostic files (not for final submission): output_v3_france_empty / india_empty / us_empty / fr045 / fr085.
