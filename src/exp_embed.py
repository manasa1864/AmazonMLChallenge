"""Cheap gate BEFORE any large embedding run: does a semantic-similarity feature (multilingual-e5-small) help the
unseen-country transfer test (proxy for France), on top of the existing 107 features?  Same discipline as exp_domain.py:
both directions, and a second random sample, before trusting any gain.

    python src/exp_embed.py --seed 0
    python src/exp_embed.py --seed 1
"""
import argparse
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from data import load_train_sample
from text import normalise_frame
from blocking import block
from features import pair_features, build_vocab, add_ambiguity
from embed import pair_cosine
from decision import assign
from metric import f05_macro

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frac", type=float, default=0.05)
args = ap.parse_args()
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
rng = np.random.default_rng(args.seed)

s1, pool, truth = load_train_sample(args.frac, args.seed)
normalise_frame(s1); normalise_frame(pool); add_ambiguity(s1); add_ambiguity(pool)
pairs = block(s1, pool, K=30, n_jobs=10)
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()
y = np.array([int(c_ids[c] in truth[s1_ids[s]]) for s, c in zip(pairs.s1_idx, pairs.c_idx)], dtype=np.int8)
ctry = s1.country.to_numpy()[pairs.s1_idx.to_numpy()]
tick(f"pairs={len(pairs)}, countries={dict(pd.Series(ctry).value_counts())}, pos rate={y.mean():.4f}")

vocab = build_vocab(s1, pool, pairs)
F = pair_features(s1, pool, pairs, vocab, n_jobs=10)
X = F.to_numpy(dtype=np.float32)
tick(f"base features {X.shape}")

sem = pair_cosine(s1, pool, pairs, text_col="name_full")
tick("semantic name-similarity feature built")
Xs = np.hstack([X, sem[:, None]])

LGBP = dict(n_estimators=300, learning_rate=0.07, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.6, n_jobs=10, verbose=-1)
BUDGET = 500_000


def sub(idx):
    return rng.choice(idx, min(BUDGET, len(idx)), replace=False)


def best_thr(idx, prob):
    f = lambda t: f05_macro({k: truth[k] for k in np.unique(s1_ids[pairs.s1_idx.to_numpy()[idx]])},
                            {k: set(v) for k, v in assign(pairs.iloc[idx], prob, t, s1_ids, c_ids).items()})
    return max((f(t), t) for t in np.arange(0.3, 0.96, 0.05))


def cross(Xm, i_src, i_tgt):
    s = sub(i_src)
    m = lgb.LGBMClassifier(**LGBP).fit(Xm[s], y[s])
    p_src = m.predict_proba(Xm[i_src])[:, 1]
    _, t_src = best_thr(i_src, p_src)             # in-domain (2-fold-free approx: refit threshold on its own preds is
    p_tgt = m.predict_proba(Xm[i_tgt])[:, 1]       # optimistic for in-domain, but src_thr is what matters for the target
    keys = np.unique(s1_ids[pairs.s1_idx.to_numpy()[i_tgt]])
    pred = {k: set(v) for k, v in assign(pairs.iloc[i_tgt], p_tgt, t_src, s1_ids, c_ids).items()}
    return f05_macro({k: truth[k] for k in keys}, pred), t_src


rows = []
for src, tgt in (("US", "India"), ("India", "US")):
    i_src, i_tgt = np.flatnonzero(ctry == src), np.flatnonzero(ctry == tgt)
    for name, Xm in (("base (107 feats)", X), ("+ semantic name sim", Xs)):
        f, t = cross(Xm, i_src, i_tgt)
        rows.append((f"{src}->{tgt}", name, f, t))
        tick(f"{src}->{tgt} {name:<22} F0.5 = {f:.4f} @ thr {t:.2f}")

R = pd.DataFrame(rows, columns=["direction", "technique", "F05", "thr"])
print("\n" + R.round(4).to_string(index=False))
piv = R.pivot(index="direction", columns="technique", values="F05")
piv["gain"] = piv["+ semantic name sim"] - piv["base (107 feats)"]
print("\n" + piv.round(4).to_string())
tick("done")
