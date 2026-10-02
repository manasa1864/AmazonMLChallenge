"""Train + validate at TEST SCALE, using training labels only.

    python src/run_fullscale.py --save_model models/f                 # all 2.2M train S1 vs the full 10M S2/S3 pool
    python src/run_fullscale.py --world_frac 0.005 --save_model models/dbg   # smoke test on a small coherent sample

Why: a 12% world has 8x fewer look-alike records than the test set, so candidate recall, score distributions and the
decision threshold measured there do not transfer (validation 0.9785 vs leaderboard 0.903). Here EVERY train S1 entity is a
query against the FULL pool, so competition between S1 rivals for the same S2/S3 record is at test density.
Only a random subset T of S1 entities (--fracT) is used to fit / evaluate the matcher (2-fold out-of-fold by S1 entity);
all other pairs are scored by the two fold models, exactly like unseen test entities, so the stage-2 context features
are computed under test conditions. Nothing here touches the test set.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from data import DATA, read_tsv, parse_gt, load_train_sample
from text import normalise_frame
from blocking import block
from features import loop_features, group_features, sim_avg_only, context_features, build_vocab, add_ambiguity
from metric import f05_macro
from decision import assign, ambiguity_threshold

LGB_PARAMS = dict(n_estimators=350, learning_rate=0.07, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, n_jobs=-1, verbose=-1)
ap = argparse.ArgumentParser()
ap.add_argument("--world_frac", type=float, default=1.0, help="<1: use the small coherent sample (smoke test)")
ap.add_argument("--fracT", type=float, default=0.15, help="share of S1 entities used to fit / evaluate the matcher")
ap.add_argument("--K", type=int, default=30)
ap.add_argument("--max_df_abs", type=int, default=12000)
ap.add_argument("--channels", action="store_true", help="union with name/address/char-3gram blocking channels")
ap.add_argument("--chunk", type=int, default=3_000_000)
ap.add_argument("--save_model", default="models/f")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

# ---- data ---------------------------------------------------------------------------------------------------------
if args.world_frac < 1:
    s1, pool, truth = load_train_sample(args.world_frac, args.seed)
else:
    s1 = read_tsv(DATA / "train/train_source1.tsv")
    pool = pd.concat([read_tsv(DATA / f"train/train_source{i}.tsv") for i in (2, 3)], ignore_index=True)
    truth = None            # ground truth is loaded after blocking (keeps peak memory down)
tick(f"train world: {len(s1)} S1, {len(pool)} pool records; countries {s1.country.value_counts().to_dict()}")
normalise_frame(s1)
normalise_frame(pool)
add_ambiguity(s1)
add_ambiguity(pool)
for d in (s1, pool):        # raw text is not needed any more
    d.drop(columns=["business_name", "business_address"], inplace=True)
gc.collect()
tick("normalised")

# ---- blocking (all S1 as queries) ----------------------------------------------------------------------------------
pairs = block(s1, pool, K=args.K, max_df_abs=args.max_df_abs, channels=args.channels)
pairs = pairs.astype({"s1_idx": np.int32, "c_idx": np.int32, "blk_rank": np.int16})
pairs = pairs.sort_values(["s1_idx", "blk_rank"], kind="stable").reset_index(drop=True)
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()

if truth is None:
    truth = parse_gt(read_tsv(DATA / "train/train_ground_truth.tsv"))
c2s = {c: s for s, cs in truth.items() for c in cs}
s1pos = pd.Series(np.arange(len(s1), dtype=np.int64), index=s1.entity_id)
owner = pool.entity_id.map(c2s).map(s1pos).to_numpy(dtype=np.float64)
owner = np.nan_to_num(owner, nan=-1).astype(np.int32)
del c2s, s1pos
y = (owner[pairs.c_idx.to_numpy()] == pairs.s1_idx.to_numpy()).astype(np.int8)
del owner
n_true = sum(len(v) for v in truth.values())
ctry = s1.country.to_numpy()[pairs.s1_idx.to_numpy()]
n_true_c = {c: sum(len(truth[k]) for k in s1_ids[s1.country.to_numpy() == c]) for c in sorted(set(ctry))}
tick(f"blocking: {len(pairs)} pairs ({len(pairs) / len(s1):.1f}/S1); pair recall {y.sum() / n_true:.4f}; "
     + "; ".join(f"{c} {y[ctry == c].sum() / n_true_c[c]:.4f}" for c in sorted(set(ctry))))
del ctry
# subset T of S1 entities used to fit / evaluate the matcher; keep ground truth only for those, free the rest
rng = np.random.default_rng(args.seed)
inT_s1 = rng.random(len(s1)) < args.fracT
fold_s1 = rng.integers(0, 2, len(s1))
truthT = {k: truth[k] for k in s1_ids[inT_s1]}
del truth
gc.collect()
iT = np.flatnonzero(inT_s1[pairs.s1_idx.to_numpy()])
oracle = {k: set() for k in truthT}
for s, c in zip(pairs.s1_idx.to_numpy()[iT][y[iT] == 1], pairs.c_idx.to_numpy()[iT][y[iT] == 1]):
    oracle[s1_ids[s]].add(c_ids[c])
print(f"          oracle F0.5 ceiling given candidates (on T) = {f05_macro(truthT, oracle):.4f}", flush=True)
del oracle

# ---- global (cheap) comparison features ------------------------------------------------------------------------------
sim = sim_avg_only(s1, pool, pairs)
G = group_features(pairs, sim, s1, pool)
del sim
tick("global comparison features")

# ---- training subset T ---------------------------------------------------------------------------------------------------
iN = np.flatnonzero(~inT_s1[pairs.s1_idx.to_numpy()])
vocab = build_vocab(s1, pool, pairs.iloc[iT])
LT = loop_features(s1, pool, pairs.iloc[iT], vocab)
FT = pd.concat([LT, G.iloc[iT]], axis=1)
del LT
s1_cols = list(FT.columns)
XT = FT.to_numpy(dtype=np.float32)
del FT
yT = y[iT]
foldT = fold_s1[pairs.s1_idx.to_numpy()[iT]]
tick(f"T: {inT_s1.sum()} S1 entities, {len(iT)} pairs, {len(s1_cols)} features")


def fit_oof(X, tag):
    oof, models = np.zeros(len(yT), dtype=np.float32), []
    for k in (0, 1):
        tr, te = foldT != k, foldT == k
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(X[tr], yT[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        models.append(m)
        tick(f"{tag}: fold {k} fitted")
    return oof, models


p1 = np.zeros(len(pairs), dtype=np.float32)
p1T, models1 = fit_oof(XT, "stage-1")
p1[iT] = p1T
# every non-T pair is scored like an unseen test pair (average of the two fold models)
for lo in range(0, len(iN), args.chunk):
    sl = iN[lo:lo + args.chunk]
    sub = pairs.iloc[sl]
    F = pd.concat([loop_features(s1, pool, sub, vocab), G.iloc[sl]], axis=1)[s1_cols].to_numpy(dtype=np.float32)
    p1[sl] = np.mean([m.predict_proba(F)[:, 1] for m in models1], axis=0)
    tick(f"stage-1 scored non-T pairs {min(lo + args.chunk, len(iN))}/{len(iN)}")

C = context_features(pairs, p1)
X2T = np.hstack([XT, C.iloc[iT].to_numpy(dtype=np.float32)])
s2_cols = s1_cols + list(C.columns)
del XT, C
p2T, models2 = fit_oof(X2T, "stage-2")

# ---- evaluation on T (official metric, out-of-fold) -------------------------------------------------------------------
pairsT = pairs.iloc[iT]
idsT = s1_ids[inT_s1]
ctryT = dict(zip(s1_ids, s1.country.to_numpy()))


def evaluate(prob, thr, keys=None):
    pred = {k: set(v) for k, v in assign(pairsT, prob, thr, s1_ids, c_ids).items()}
    tr = truthT if keys is None else {k: truthT[k] for k in keys}
    return f05_macro(tr, pred)


def sweep(name, prob):
    best = max((evaluate(prob, t), t) for t in np.arange(0.3, 0.96, 0.05))
    best = max((evaluate(prob, t), t) for t in np.arange(max(best[1] - 0.05, 0.05), min(best[1] + 0.051, 0.99), 0.01))
    print(f"  {name:<28} F0.5 = {best[0]:.4f} @ thr {best[1]:.2f}", flush=True)
    return best


print("\n=== Validation at TEST SCALE (held-out train S1 entities, out-of-fold) ===")
sweep("cosine only", G.blk_score.to_numpy()[iT])
sweep("stage-1", p1T)
f_best, t_best = sweep("stage-2 (final)", p2T)
for c in sorted(set(ctryT.values())):
    keys = [k for k in idsT if ctryT[k] == c]
    print(f"  stage-2 on {c:<8} ({len(keys)} S1): F0.5 = {evaluate(p2T, t_best, keys):.4f}")
# address-ambiguous subset: S1 sharing its street+city with other S1 records (the France-like condition, but
# general and label-free: applies to any country, so it needs no French training data to be tuned).
amb = s1.amb_street.to_numpy()
hi = [k for k, a in zip(s1_ids[inT_s1], amb[inT_s1]) if a > 0]
if hi:
    print(f"  stage-2 on street-ambiguous S1 ({len(hi)}): F0.5 = {evaluate(p2T, t_best, hi):.4f}")

# ---- ambiguity-conditioned threshold: require more evidence for S1 entities whose address is shared ------------------
amb_pairT = s1.amb_street.to_numpy()[pairsT.s1_idx.to_numpy()]


def evaluate_amb(t_low, t_high):
    thr = ambiguity_threshold(amb_pairT, t_low, t_high)
    pred = {k: set(v) for k, v in assign(pairsT, p2T, thr, s1_ids, c_ids).items()}
    return f05_macro(truthT, pred)


coarse = np.arange(0.35, 0.91, 0.1)
best_amb = max(((evaluate_amb(tl, th), tl, th) for tl in coarse for th in coarse), key=lambda x: x[0])
fine = lambda t: np.arange(max(t - 0.1, 0.05), min(t + 0.101, 0.97), 0.02)
best_amb = max(((evaluate_amb(tl, th), tl, th) for tl in fine(best_amb[1]) for th in fine(best_amb[2])),
              key=lambda x: x[0])
f_amb, t_low, t_high = best_amb
print(f"  ambiguity-conditioned threshold: t_low={t_low:.2f} (normal), t_high={t_high:.2f} (address shared "
      f"with other S1) -> F0.5 = {f_amb:.4f}  (single-threshold was {f_best:.4f})")
use_amb = f_amb > f_best + 0.0005
if use_amb:
    print("  -> using the ambiguity-conditioned threshold")
    f_best = f_amb
else:
    print("  -> no real improvement, keeping the single threshold")

thrT = ambiguity_threshold(amb_pairT, t_low, t_high) if use_amb else t_best
pred = {k: set(v) for k, v in assign(pairsT, p2T, thrT, s1_ids, c_ids).items()}
sing = [k for k, v in truthT.items() if not v]
tp = sum(len(pred.get(k, set()) & v) for k, v in truthT.items())
npred = sum(len(v) for v in pred.values())
nt = sum(len(v) for v in truthT.values())
print(f"  singletons handled = {np.mean([not pred.get(k) for k in sing]):.3f}; pair precision = {tp / max(npred, 1):.4f}; "
      f"pair recall = {tp / nt:.4f}; avg #pred/S1 = {npred / len(truthT):.2f} (true {nt / len(truthT):.2f})")

# ---- save --------------------------------------------------------------------------------------------------------------------
pre = args.save_model
Path(pre).parent.mkdir(parents=True, exist_ok=True)
for i, m in enumerate(models1):
    m.booster_.save_model(f"{pre}_s1_fold{i}.txt")
for i, m in enumerate(models2):
    m.booster_.save_model(f"{pre}_s2_fold{i}.txt")
Path(pre + ".json").write_text(json.dumps(
    {"thr": float(t_best), "thr_low": float(t_low if use_amb else t_best), "thr_high": float(t_high if use_amb else t_best),
     "use_amb_thr": bool(use_amb), "val_f05": float(f_best), "vocab": vocab, "frac": args.world_frac, "fracT": args.fracT,
     "K": args.K, "max_df_abs": args.max_df_abs, "channels": args.channels, "s1_features": s1_cols, "s2_features": s2_cols}))
tick(f"models saved with prefix {pre} (thr={t_best:.3f}, amb thr={use_amb})")
