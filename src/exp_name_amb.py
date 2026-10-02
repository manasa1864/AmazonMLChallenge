"""Cheap gate: does the name-ambiguity feature (from the error scan finding) actually help, in a proper 2-fold
held-out-by-S1-entity test, over BOTH the full held-out F0.5 and specifically the 'empty candidate address' cases
where the error scan found the false positives concentrated?"""
import argparse
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from data import CACHE
from text import normalise_frame

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="norm_f0.25.pkl", help="cached world file, or 'train_f0.05_s1.pkl' for a genuinely different sample")
ap.add_argument("--nq", type=int, default=45000)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()
from blocking import block
from features import pair_features, build_vocab, add_ambiguity, add_name_ambiguity
from metric import f05_macro
from decision import assign

T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

s1, pool, truth = pd.read_pickle(CACHE / args.world)
if "name_core" not in s1.columns:       # a raw load_train_sample() cache, not yet normalised
    normalise_frame(s1); normalise_frame(pool)
if "addr_street" not in s1.columns:
    for d in (s1, pool):
        d["addr_street"] = [" ".join(t for t in a.split() if not any(ch.isdigit() for ch in t)) for a in d.addr]
add_ambiguity(s1); add_ambiguity(pool)
add_name_ambiguity(s1); add_name_ambiguity(pool)

rng = np.random.default_rng(args.seed)
qidx = np.sort(rng.choice(len(s1), min(args.nq, len(s1)), replace=False))
q = s1.iloc[qidx].reset_index(drop=True)
pairs = block(q, pool, K=30, n_jobs=10)
s1_ids, c_ids = q.entity_id.to_numpy(), pool.entity_id.to_numpy()
y = np.array([int(c_ids[c] in truth[s1_ids[s]]) for s, c in zip(pairs.s1_idx, pairs.c_idx)], dtype=np.int8)
tick(f"pairs={len(pairs)}, pos rate={y.mean():.4f}")

vocab = build_vocab(q, pool, pairs)
F = pair_features(q, pool, pairs, vocab, n_jobs=10)          # includes the new name-ambiguity columns
base_cols = [c for c in F.columns if not c.startswith(("s1_name_amb", "c_name_amb"))]
tick(f"features {F.shape} ({len(F.columns) - len(base_cols)} new)")

fold = pairs.s1_idx.to_numpy() % 2
LGBP = dict(n_estimators=300, learning_rate=0.07, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.6, n_jobs=10, verbose=-1)


def oof(cols):
    X = F[cols].to_numpy(dtype=np.float32)
    o = np.zeros(len(y), dtype=np.float32)
    for k in (0, 1):
        tr, te = fold != k, fold == k
        o[te] = lgb.LGBMClassifier(**LGBP).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return o


def best_thr(prob, mask=None):
    idx = np.arange(len(y)) if mask is None else np.flatnonzero(mask)
    keys = np.unique(s1_ids[pairs.s1_idx.to_numpy()[idx]])
    tr = {k: truth[k] for k in keys}

    def f(t):
        pred = {k: set(v) for k, v in assign(pairs.iloc[idx], prob[idx], t, s1_ids, c_ids).items()}
        return f05_macro(tr, pred)
    return max((f(t), t) for t in np.arange(0.3, 0.96, 0.05))


p_base = oof(base_cols)
p_new = oof(list(F.columns))
tick("both models trained (2-fold OOF)")

f_base, t_base = best_thr(p_base)
f_new, t_new = best_thr(p_new)
print(f"\nOverall (all pairs):  base F0.5 = {f_base:.4f} @ {t_base:.2f}   +name_amb F0.5 = {f_new:.4f} @ {t_new:.2f}   gain = {f_new - f_base:+.4f}")

empty = (pool.addr.to_numpy()[pairs.c_idx.to_numpy()] == "")
f_base_e, _ = best_thr(p_base, empty)
f_new_e, _ = best_thr(p_new, empty)
print(f"S1 with an empty-address candidate ({len(np.unique(s1_ids[pairs.s1_idx.to_numpy()[empty]]))} S1 entities): "
      f"base F0.5 = {f_base_e:.4f}   +name_amb F0.5 = {f_new_e:.4f}   gain = {f_new_e - f_base_e:+.4f}")

imp = pd.Series(lgb.LGBMClassifier(**LGBP).fit(F.to_numpy(dtype=np.float32), y).feature_importances_,
                index=F.columns).sort_values(ascending=False)
print("\nname-ambiguity feature ranks:\n" + imp.loc[["s1_name_amb", "c_name_amb", "c_name_amb_x_addr_empty"]].to_string())
tick("done")
