"""Generate the submission files from the TEST set using models trained on the training set.

    python src/run_test.py --model models/m --out output

Writes output/matching_results.tsv (scored) and output/candidate_pairs.tsv (blocking candidates fed to the model).
The stage-1 / stage-2 fold models saved by run_validation.py are averaged. Pair features are built in chunks
(the test set has ~50M candidate pairs); comparison features across rival S1 entities are computed globally.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
import pandas as pd

from data import DATA, read_tsv
from text import normalise_frame
from blocking import block
from features import loop_features, group_features, sim_avg_only, context_features, add_ambiguity
from decision import assign, ambiguity_threshold
from par import JOBS_CAP

PRED_KW = {"num_threads": JOBS_CAP} if JOBS_CAP else {}      # LightGBM prediction threads (AML_JOBS caps them)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/m", help="prefix used with run_validation.py --save_model")
ap.add_argument("--out", default="output")
ap.add_argument("--thr", type=float, default=None, help="override the threshold stored with the model")
ap.add_argument("--limit", type=int, default=None, help="debug only: use the first N test S1 rows (output is NOT submittable)")
ap.add_argument("--chunk", type=int, default=3_000_000, help="pairs per feature/prediction chunk")
args = ap.parse_args()
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

meta = json.loads(Path(args.model + ".json").read_text())
thr = args.thr if args.thr is not None else meta["thr"]                       # kept for logging / --thr override
use_amb_thr = args.thr is None and meta.get("use_amb_thr", False)
load = lambda stage: [lgb.Booster(model_file=f"{args.model}_{stage}_fold{i}.txt") for i in (0, 1)]
m1, m2 = load("s1"), load("s2")
tick(f"models {args.model}: thr={thr:.3f}, trained on frac={meta['frac']}, val F0.5={meta['val_f05']:.4f}")

s1 = read_tsv(DATA / "test/test_source1.tsv")
if args.limit:
    s1 = s1.head(args.limit).copy()
pool = pd.concat([read_tsv(DATA / f"test/test_source{i}.tsv") for i in (2, 3)], ignore_index=True)
tick(f"test: {len(s1)} S1, {len(pool)} pool records; countries: {s1.country.value_counts().to_dict()}")
normalise_frame(s1)
normalise_frame(pool)
add_ambiguity(s1)
add_ambiguity(pool)
tick("normalised")

pairs = block(s1, pool, K=meta["K"], max_df_abs=meta["max_df_abs"], channels=meta.get("channels", False))
pairs = pairs.astype({"s1_idx": np.int32, "c_idx": np.int32, "blk_rank": np.int16}).reset_index(drop=True)
pairs = pairs.sort_values(["s1_idx", "blk_rank"], kind="stable").reset_index(drop=True)   # grouped by S1 entity
tick(f"blocking: {len(pairs)} candidate pairs ({len(pairs) / len(s1):.1f} per S1)")
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()

# write the candidate set right away (it is what the model is about to score)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def write(path, header, d):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for sid in s1_ids:                      # exactly one row per test S1 entity, in file order
            f.write(f"{sid}\t{','.join(d.get(sid, []))}\n")


cands = {}
for s, c in zip(pairs.s1_idx.to_numpy(), pairs.c_idx.to_numpy()):
    cands.setdefault(s1_ids[s], []).append(c_ids[c])
write(out / "candidate_pairs.tsv", "candidate_entity_ids", cands)
del cands
tick("candidate_pairs.tsv written")

sim = sim_avg_only(s1, pool, pairs)
G = group_features(pairs, sim, s1, pool)
del sim
tick("global comparison features done")

s1_cols, s2_cols = meta["s1_features"], meta["s2_features"]
bounds = [(lo, min(lo + args.chunk, len(pairs))) for lo in range(0, len(pairs), args.chunk)]


def chunk_frame(lo, hi):
    sub = pairs.iloc[lo:hi]
    return sub, pd.concat([loop_features(s1, pool, sub, meta["vocab"]), G.iloc[lo:hi]], axis=1)


p1 = np.zeros(len(pairs), dtype=np.float32)
for lo, hi in bounds:
    _, F = chunk_frame(lo, hi)
    X = F[s1_cols].to_numpy(dtype=np.float32)
    p1[lo:hi] = np.mean([m.predict(X, **PRED_KW) for m in m1], axis=0)
    tick(f"stage-1 scored {hi}/{len(pairs)}")

C = context_features(pairs, p1)
prob = np.zeros(len(pairs), dtype=np.float32)
for lo, hi in bounds:
    _, F = chunk_frame(lo, hi)
    F2 = pd.concat([F, C.iloc[lo:hi]], axis=1)
    X = F2[s2_cols].to_numpy(dtype=np.float32)
    prob[lo:hi] = np.mean([m.predict(X, **PRED_KW) for m in m2], axis=0)
    tick(f"stage-2 scored {hi}/{len(pairs)}")
if use_amb_thr:
    # stricter threshold for S1 entities whose address is shared with other S1 businesses (most of France; a
    # small share of US/India) - tuned on training data only, applies uniformly regardless of country.
    thr_vec = ambiguity_threshold(s1.amb_street.to_numpy()[pairs.s1_idx.to_numpy()], meta["thr_low"], meta["thr_high"])
    tick(f"scored; {(prob >= thr_vec).sum()} pairs above the ambiguity-conditioned threshold "
         f"(low={meta['thr_low']:.2f}, high={meta['thr_high']:.2f})")
else:
    thr_vec = thr
    tick(f"scored; {(prob >= thr).sum()} pairs above threshold {thr:.3f}")
work = Path("cache")                        # kept for re-thresholding without recomputation (not part of output/)
work.mkdir(exist_ok=True)
tag = "dbg" if args.limit else "full"
np.save(work / f"test_probs_{tag}.npy", prob)
pairs[["s1_idx", "c_idx"]].to_parquet(work / f"test_pairs_{tag}.parquet")

matches = assign(pairs, prob, thr_vec, s1_ids, c_ids)
write(out / "matching_results.tsv", "matched_entity_ids", matches)
n_match = sum(len(v) for v in matches.values())
tick(f"wrote {out}/: {len(s1_ids)} rows; {n_match} matched ids; "
     f"{sum(1 for k in s1_ids if k not in matches)} empty (singleton) rows; "
     f"avg matches/S1 = {n_match / len(s1_ids):.2f}")
