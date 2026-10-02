"""Which technique transfers best to a country it has never seen?  (proxy for France, using training labels only)

Leave-one-country-out on the training data: fit on US -> score India, fit on India -> score US, versus in-domain
(2-fold by S1 entity inside one country).  The decision threshold is tuned on the SOURCE country (as it would have to be
for France, where no labels exist) and applied unchanged to the target; we also report the oracle threshold to separate
'ranking got worse' from 'calibration shifted'.

Techniques (same pair features, same candidates, same training-row budget):
  LightGBM | RandomForest | Logistic regression | kNN (15 core features) |
  LightGBM on per-country percentile-rank features (label-free normalisation of each country's own feature distribution) |
  LightGBM + one round of self-training on the target country's own unlabeled pairs (confident pseudo-labels).
"""
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from data import load_train_sample
from text import normalise_frame
from blocking import block
from features import pair_features, build_vocab, add_ambiguity
from metric import f05_macro
from decision import assign

NJ = 8
SEED = int(os.environ.get('AML_SEED', '0'))                      # which random sample of the training data
ONLY = [t.strip().lower() for t in os.environ.get('AML_MODELS', '').split(',') if t.strip()]   # e.g. 'lightgbm,rank,self'
DIRS = [t.strip() for t in os.environ.get('AML_DIRS', '').split(',') if t.strip()]             # e.g. 'India->US'
want = lambda name: (not ONLY) or any(t in name.lower() for t in ONLY)
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)

s1, pool, truth = load_train_sample(0.05, SEED)
normalise_frame(s1); normalise_frame(pool); add_ambiguity(s1); add_ambiguity(pool)
pairs = block(s1, pool, K=30, n_jobs=NJ)
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()
y = np.array([int(c_ids[c] in truth[s1_ids[s]]) for s, c in zip(pairs.s1_idx, pairs.c_idx)], dtype=np.int8)
vocab = build_vocab(s1, pool, pairs)
F = pair_features(s1, pool, pairs, vocab, n_jobs=NJ)
X = F.to_numpy(dtype=np.float32)
ctry = s1.country.to_numpy()[pairs.s1_idx.to_numpy()]
tick(f"pairs={len(pairs)}, features={X.shape[1]}, countries={dict(pd.Series(ctry).value_counts())}")

# per-country percentile-rank version of every feature (label-free, uses only that country's own pairs)
XR = F.groupby(ctry).rank(pct=True).to_numpy(dtype=np.float32)
KNN_COLS = [list(F.columns).index(c) for c in ("nm_tset", "nm_jw", "ad_tset", "num_first_eq", "sk_ratio", "blk_score", "blk_ratio",
                                                "ex_c", "ex_s", "num_jac", "zip_match", "nm_full_tset", "ad_jac", "nm_lenratio", "ad_empty")]

LGBP = dict(n_estimators=300, learning_rate=0.07, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, n_jobs=NJ, verbose=-1)
BUDGET = 500_000          # training rows per model (same for every technique)
rng = np.random.default_rng(SEED)


def sub(idx):
    """One random training-row subset. ALWAYS call once and reuse the result for both X and y (a second call draws
    different rows and silently misaligns features and labels)."""
    return rng.choice(idx, min(BUDGET, len(idx)), replace=False)


def fit_on(fit, Xm, idx):
    s = sub(idx)
    return fit(Xm[s], y[s])


def f05(idx, prob, thr):
    pred = {k: set(v) for k, v in assign(pairs.iloc[idx], prob, thr, s1_ids, c_ids).items()}
    keys = np.unique(s1_ids[pairs.s1_idx.to_numpy()[idx]])
    return f05_macro({k: truth[k] for k in keys}, pred)


def best_thr(idx, prob):
    return max((f05(idx, prob, t), t) for t in np.arange(0.3, 0.96, 0.05))


class Model:
    def __init__(self, name, fit, xsel=None):
        self.name, self.fit, self.xsel = name, fit, xsel


def lgb_fit(Xtr, ytr, Xu=None):
    return lgb.LGBMClassifier(**LGBP).fit(Xtr, ytr).predict_proba


def rf_fit(Xtr, ytr, Xu=None):
    return RandomForestClassifier(n_estimators=120, min_samples_leaf=3, max_features="sqrt", n_jobs=NJ, random_state=0).fit(Xtr, ytr).predict_proba


def lr_fit(Xtr, ytr, Xu=None):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
    return lambda Z: m.predict_proba(sc.transform(Z))


def knn_fit(Xtr, ytr, Xu=None):
    Xk = Xtr[:, KNN_COLS]
    sc = StandardScaler().fit(Xk)
    m = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=NJ).fit(sc.transform(Xk[:200_000]), ytr[:200_000])
    return lambda Z: m.predict_proba(sc.transform(Z[:, KNN_COLS]))


MODELS = [("LightGBM", lgb_fit, X), ("RandomForest", rf_fit, X), ("LogisticReg", lr_fit, X), ("kNN(25)", knn_fit, X),
          ("LightGBM on country-rank feats", lgb_fit, XR)]


def in_domain(idx_c, Xm, fit):
    """2-fold out-of-fold by S1 within one country -> (oof prob over idx_c, tuned threshold, F0.5)."""
    fold = pairs.s1_idx.to_numpy()[idx_c] % 2
    oof = np.zeros(len(idx_c))
    for k in (0, 1):
        tr, te = idx_c[fold != k], np.flatnonzero(fold == k)
        pred = fit_on(fit, Xm, tr)
        oof[te] = pred(Xm[idx_c[te]])[:, 1]
    f, t = best_thr(idx_c, oof)
    return f, t


rows = []
for src, tgt in (("US", "India"), ("India", "US")):
    if DIRS and f"{src}->{tgt}" not in DIRS:
        continue
    i_src, i_tgt = np.flatnonzero(ctry == src), np.flatnonzero(ctry == tgt)
    # evaluation target subset: 20k S1 entities keeps kNN / RF prediction tractable and identical for all models
    ts = np.unique(pairs.s1_idx.to_numpy()[i_tgt])
    ts = rng.choice(ts, min(20000, len(ts)), replace=False)
    i_tgt = i_tgt[np.isin(pairs.s1_idx.to_numpy()[i_tgt], ts)]
    for name, fit, Xm in MODELS:
        if not want(name):
            continue
        f_in, t_src = in_domain(i_src, Xm, fit)                       # in-domain score + threshold from the SOURCE country
        assert f_in > 0.5, f"sanity check failed: {name} in-domain F0.5 = {f_in:.3f} is implausibly low (bug, not a result)"
        pred = fit_on(fit, Xm, i_src)
        p = pred(Xm[i_tgt])[:, 1]
        f_cross = f05(i_tgt, p, t_src)
        f_orc, t_orc = best_thr(i_tgt, p)
        rows.append((f"{src}->{tgt}", name, f_in, f_cross, f_orc, t_src, t_orc))
        tick(f"{src}->{tgt} {name:<32} in-domain {f_in:.4f} | cross(src thr {t_src:.2f}) {f_cross:.4f} | cross(oracle thr {t_orc:.2f}) {f_orc:.4f}")
    if not want('self-training'):
        continue
    # self-training: LightGBM + confident pseudo-labels on the target's own unlabeled pairs
    s_src = sub(i_src)
    m0 = lgb_fit(X[s_src], y[s_src])
    _, t_src = in_domain(i_src, X, lgb_fit)
    p0 = m0(X[i_tgt])[:, 1]
    conf = (p0 >= 0.95) | (p0 <= 0.02)
    Xst = np.vstack([X[s_src], X[i_tgt][conf]])
    yst = np.concatenate([y[s_src], (p0[conf] >= 0.5).astype(np.int8)])               # pseudo-labels: NO target labels used
    p1 = lgb_fit(Xst, yst)(X[i_tgt])[:, 1]
    f_cross = f05(i_tgt, p1, t_src)
    f_orc, t_orc = best_thr(i_tgt, p1)
    rows.append((f"{src}->{tgt}", "LightGBM + self-training", np.nan, f_cross, f_orc, t_src, t_orc))
    tick(f"{src}->{tgt} LightGBM + self-training ({conf.mean():.2f} of target pairs pseudo-labeled) cross(src thr) {f_cross:.4f} | oracle {f_orc:.4f}")

R = pd.DataFrame(rows, columns=["direction", "technique", "in_domain_F05", "cross_F05_src_thr", "cross_F05_oracle_thr", "src_thr", "oracle_thr"])
print("\n=== Unseen-country transfer (proxy for France) ===")
print(R.round(4).to_string(index=False))
avg = R.groupby("technique")[["cross_F05_src_thr", "cross_F05_oracle_thr"]].mean().sort_values("cross_F05_src_thr", ascending=False)
print("\nAverage over the two directions (higher is better):")
print(avg.round(4).to_string())
