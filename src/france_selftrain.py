"""Country-only self-training on a country's own UNLABELED test pairs (default: France).  Only that country's rows change.

    python src/france_selftrain.py --model models/g --base output_v3 --out output_v4_france_selftrain

Steps (no test labels, no leaderboard feedback anywhere):
 1. Rebuild the country's candidate pairs and scores exactly as run_test.py does (same models, same features), for the
    country alone.  GATE: the resulting matches must reproduce the base submission's rows for that country
    (>= 99.5% identical as sets) - otherwise this script stops, because the rest would be built on something wrong.
 2. Pseudo-label the country's pairs the model is very sure about (p >= 0.95 -> match, p <= 0.02 -> no match).
 3. Refit the stage-1 model on [labeled TRAIN source sample at test scale (build_source.py)  +  the pseudo-labeled pairs]:
    the labeled source anchors the model, the pseudo-labels align it with the new country's feature distribution.
 4. Recompute stage-2 context features from the adapted stage-1 probabilities, rescore with the ORIGINAL stage-2 models,
    apply the ORIGINAL training-tuned threshold, write the base submission with only this country's rows replaced.
Evidence for the technique (train-only, unseen-country proxy, two random samples): US->India +1.6 / +0.8 points,
India->US -0.3 / -0.1 points versus plain LightGBM.
"""
import argparse
import gc
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
from decision import assign
from par import JOBS_CAP

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/g")
ap.add_argument("--base", default="output_v3", help="submission whose other countries are kept unchanged")
ap.add_argument("--out", default="output_v4_france_selftrain")
ap.add_argument("--country", default="France")
ap.add_argument("--source", default="cache/source_stage1.npz")
ap.add_argument("--src_rows", type=int, default=1_500_000)
ap.add_argument("--tgt_rows", type=int, default=1_500_000)
ap.add_argument("--chunk", type=int, default=2_000_000)
ap.add_argument("--no_gate", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
NT = JOBS_CAP or -1
LGBP = dict(n_estimators=350, learning_rate=0.07, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, n_jobs=NT, verbose=-1)

meta = json.loads(Path(args.model + ".json").read_text())
thr = meta["thr"]
s1_cols, s2_cols = meta["s1_features"], meta["s2_features"]
load = lambda stage: [lgb.Booster(model_file=f"{args.model}_{stage}_fold{i}.txt") for i in (0, 1)]
m1, m2 = load("s1"), load("s2")
PRED = {"num_threads": JOBS_CAP} if JOBS_CAP else {}

# ---- 1. rebuild the country's candidates and scores exactly like run_test.py ---------------------------------------
s1 = read_tsv(DATA / "test/test_source1.tsv")
s1 = s1[s1.country == args.country].reset_index(drop=True)
pool = pd.concat([read_tsv(DATA / f"test/test_source{i}.tsv") for i in (2, 3)], ignore_index=True)
pool = pool[pool.country == args.country].reset_index(drop=True)
tick(f"{args.country}: {len(s1)} S1, {len(pool)} pool records")
normalise_frame(s1); normalise_frame(pool); add_ambiguity(s1); add_ambiguity(pool)
pairs = block(s1, pool, K=meta["K"], max_df_abs=meta["max_df_abs"], channels=meta.get("channels", False))
pairs = pairs.astype({"s1_idx": np.int32, "c_idx": np.int32, "blk_rank": np.int16})
pairs = pairs.sort_values(["s1_idx", "blk_rank"], kind="stable").reset_index(drop=True)
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()
tick(f"blocking: {len(pairs)} pairs")
sim = sim_avg_only(s1, pool, pairs)
G = group_features(pairs, sim, s1, pool)
del sim
n = len(pairs)
F1 = np.zeros((n, len(s1_cols)), dtype=np.float32)
for lo in range(0, n, args.chunk):
    sl = slice(lo, min(lo + args.chunk, n))
    sub = pairs.iloc[sl]
    F1[sl] = pd.concat([loop_features(s1, pool, sub, meta["vocab"]), G.iloc[sl]], axis=1)[s1_cols].to_numpy(dtype=np.float32)
    tick(f"features {sl.stop}/{n}")
del G
gc.collect()


def score_stage1(models, X):
    return np.mean([m.predict(X, **PRED) if isinstance(m, lgb.Booster) else m.predict_proba(X)[:, 1] for m in models], axis=0)


def score_stage2(p1):
    C = context_features(pairs, p1).to_numpy(dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for lo in range(0, n, args.chunk):
        sl = slice(lo, min(lo + args.chunk, n))
        X2 = np.hstack([F1[sl], C[sl]])
        out[sl] = np.mean([m.predict(X2, **PRED) for m in m2], axis=0)
    return out


p1 = np.zeros(n, dtype=np.float32)
for lo in range(0, n, args.chunk):
    sl = slice(lo, min(lo + args.chunk, n))
    p1[sl] = score_stage1(m1, F1[sl])
p2 = score_stage2(p1)
base_matches = {k: set(v) for k, v in assign(pairs, p2, thr, s1_ids, c_ids).items()}
tick(f"baseline rebuilt: {sum(len(v) for v in base_matches.values())} matched ids")

# ---- GATE: must reproduce the base submission for this country -----------------------------------------------------
base_file = pd.read_csv(Path(args.base) / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
base_rows = {k: (frozenset(v.split(",")) if v else frozenset()) for k, v in zip(base_file.source1_entity_id, base_file.matched_entity_ids)}
same = np.mean([base_rows[k] == frozenset(base_matches.get(k, ())) for k in s1_ids])
tick(f"GATE: {args.country} rows reproduced exactly (as sets) = {same:.4f}")
if same < 0.995 and not args.no_gate:
    raise SystemExit(f"GATE FAILED ({same:.4f} < 0.995): the rebuilt pipeline does not reproduce {args.base}; stopping.")

# ---- 2-3. self-training on the country's own unlabeled pairs, anchored by labeled train source data -----------------
rng = np.random.default_rng(args.seed)
src = np.load(args.source)
assert list(src["cols"]) == s1_cols, "source sample was built with a different feature set"
si = rng.choice(len(src["y"]), min(args.src_rows, len(src["y"])), replace=False)
Xs, ys = src["X"][si], src["y"][si]
conf = np.flatnonzero((p2 >= 0.95) | (p2 <= 0.02))
ti = rng.choice(conf, min(args.tgt_rows, len(conf)), replace=False)
yt = (p2[ti] >= 0.5).astype(np.int8)
tick(f"self-training set: {len(ys)} labeled source pairs ({ys.mean():.3f} pos) + {len(ti)} pseudo-labeled {args.country} pairs "
     f"({yt.mean():.3f} pos; {len(conf) / n:.2f} of {args.country} pairs were confident)")
Xtr, ytr = np.vstack([Xs, F1[ti]]), np.concatenate([ys, yt])
del Xs, src
models = []
for seed in (0, 1):
    idx = rng.choice(len(ytr), int(0.85 * len(ytr)), replace=False)
    models.append(lgb.LGBMClassifier(**{**LGBP, "random_state": seed}).fit(Xtr[idx], ytr[idx]))
    tick(f"adapted stage-1 model {seed} fitted")
del Xtr
p1n = np.zeros(n, dtype=np.float32)
for lo in range(0, n, args.chunk):
    sl = slice(lo, min(lo + args.chunk, n))
    p1n[sl] = score_stage1(models, F1[sl])
p2n = score_stage2(p1n)
new_matches = {k: set(v) for k, v in assign(pairs, p2n, thr, s1_ids, c_ids).items()}

# ---- report ----------------------------------------------------------------------------------------------------------------
cnt = lambda m: np.array([len(m.get(k, ())) for k in s1_ids])
b, a = cnt(base_matches), cnt(new_matches)
changed = np.mean([base_matches.get(k, set()) != new_matches.get(k, set()) for k in s1_ids])
added = sum(len(new_matches.get(k, set()) - base_matches.get(k, set())) for k in s1_ids)
removed = sum(len(base_matches.get(k, set()) - new_matches.get(k, set())) for k in s1_ids)
print(f"\n{args.country}: avg matches/S1 {b.mean():.3f} -> {a.mean():.3f} | empty rows {np.mean(b == 0):.3f} -> {np.mean(a == 0):.3f} | "
      f"rows changed {changed:.3f} | pairs added {added}, removed {removed}")
print(f"  mean prob of pairs above 0.5: base {p2[p2 > .5].mean():.4f}, adapted {p2n[p2n > .5].mean():.4f}; "
      f"pairs above threshold: base {(p2 >= thr).sum()}, adapted {(p2n >= thr).sum()}", flush=True)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
allrows = pd.read_csv(Path(args.base) / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
new_str = {k: ",".join(new_matches.get(k, [])) for k in s1_ids}
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for k, v in zip(allrows.source1_entity_id, allrows.matched_entity_ids):
        f.write(f"{k}\t{new_str.get(k, v)}\n")
tick(f"wrote {out}/matching_results.tsv (only {args.country} rows differ from {args.base}; candidates unchanged)")
