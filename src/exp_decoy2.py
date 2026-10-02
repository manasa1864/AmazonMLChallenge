"""Does the label-free per-country word-ratio feature add signal beyond the existing 107 features?
2-fold (by S1 entity) LightGBM with vs without the new features on candidate pairs of 15k S1 from the cached 25% world.
Reports overall AUC, AUC on the hard subset (similar names + extra words in the candidate), and pair-level precision at
90% recall of the without/with models."""
import time
from collections import Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from data import CACHE
from blocking import block
from features import pair_features, build_vocab, add_ambiguity

T0 = time.time()
s1, pool, truth = pd.read_pickle(CACHE / "norm_f0.25.pkl")
if "addr_street" not in s1.columns:                       # cached frames predate the street key
    for d in (s1, pool):
        d["addr_street"] = [" ".join(t for t in a.split() if not any(ch.isdigit() for ch in t)) for a in d.addr]
add_ambiguity(s1)
add_ambiguity(pool)

A = 5.0
ratio = {}
for c in s1.country.unique():
    a, b = s1[s1.country == c], pool[pool.country == c]
    ca, cb = Counter(), Counter()
    for n in a.name_core:
        ca.update(set(n.split()))
    for n in b.name_core:
        cb.update(set(n.split()))
    scale = len(b) / len(a)
    ratio[c] = {t: float(np.log((cb[t] + A) / (ca.get(t, 0) * scale + A))) for t in cb}

rng = np.random.default_rng(0)
q = s1.iloc[np.sort(rng.choice(len(s1), 15000, replace=False))].reset_index(drop=True)
pairs = block(q, pool, K=30)
ids_q, ids_p = q.entity_id.to_numpy(), pool.entity_id.to_numpy()
y = np.array([int(ids_p[c] in truth[ids_q[s]]) for s, c in zip(pairs.s1_idx, pairs.c_idx)])
vocab = build_vocab(q, pool, pairs)
F = pair_features(q, pool, pairs, vocab)
print(f"[{time.time() - T0:.0f}s] base features {F.shape}", flush=True)


def ratio_feats(s1_core, p_core, country):
    r = ratio[country]
    ta, tb = set(s1_core.split()), set(p_core.split())
    ex_c = [r.get(t, 0.0) for t in tb - ta]
    ex_s = [r.get(t, 0.0) for t in ta - tb]
    return (max(ex_c) if ex_c else -9, float(np.mean(ex_c)) if ex_c else -9, sum(v > 1.0 for v in ex_c),
            max(ex_s) if ex_s else -9, float(np.mean(ex_s)) if ex_s else -9)


NEW = ["rt_c_max", "rt_c_mean", "rt_c_n_hi", "rt_s_max", "rt_s_mean"]
R = pd.DataFrame([ratio_feats(q.name_core.iat[s], pool.name_core.iat[c], q.country.iat[s])
                  for s, c in zip(pairs.s1_idx.to_numpy(), pairs.c_idx.to_numpy())], columns=NEW, index=F.index)
print(f"[{time.time() - T0:.0f}s] ratio features built", flush=True)

fold = (pairs.s1_idx.to_numpy() % 2)
hard = ((F.nm_tset >= 80) & (F.ex_c >= 1)).to_numpy()


def cv(X):
    o = np.zeros(len(y))
    for k in (0, 1):
        tr, te = fold != k, fold == k
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.07, num_leaves=63, min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.6, n_jobs=10, verbose=-1).fit(X[tr], y[tr])
        o[te] = m.predict_proba(X[te])[:, 1]
    return o


def report(name, o):
    # precision at 90% recall over all pairs
    order = np.argsort(-o)
    tp = np.cumsum(y[order])
    k90 = np.searchsorted(tp, 0.9 * y.sum())
    print(f"  {name:<22} AUC all {roc_auc_score(y, o):.4f} | AUC hard-subset ({hard.sum()} pairs) "
          f"{roc_auc_score(y[hard], o[hard]):.4f} | precision at 90% recall {tp[k90] / (k90 + 1):.4f}", flush=True)


print("\n=== 2-fold CV by S1 entity, same pairs, same LightGBM ===")
report("existing features", cv(F.to_numpy(dtype=np.float32)))
report("+ word-ratio features", cv(pd.concat([F, R], axis=1).to_numpy(dtype=np.float32)))
print(f"[{time.time() - T0:.0f}s] done")
