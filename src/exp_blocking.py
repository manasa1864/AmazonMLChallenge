"""Blocking experiments on the large (25%) pool: pair recall at top-K for several token/weighting variants."""
import sys
import time
import numpy as np
import pandas as pd

from data import load_train_sample, CACHE
from text import normalise_frame
from blocking import block

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

cp = CACHE / f"norm_f{FRAC}.pkl"
if cp.exists():
    s1, pool, truth = pd.read_pickle(cp)
else:
    s1, pool, truth = load_train_sample(FRAC, 0)
    normalise_frame(s1)
    normalise_frame(pool)
    pd.to_pickle((s1, pool, truth), cp)
tick(f"loaded {len(s1)} S1, {len(pool)} pool")

rng = np.random.default_rng(0)
q = s1.iloc[np.sort(rng.choice(len(s1), NQ, replace=False))].reset_index(drop=True)
c_ids = pool.entity_id.to_numpy()
n_true = sum(len(truth[k]) for k in q.entity_id)


def doc_factory(use_k, use_bi, k_nonlatin_only=False):
    def fn(df):
        out = []
        for n, k, a, nl, eid in zip(df["name_core"].tolist(), df["name_skel"].tolist(), df["addr"].tolist(),
                                    df["nonlatin"].tolist(), df["entity_id"].tolist()):
            t = ["n_" + x for x in n.split()] + ["a_" + x for x in a.split()]
            if use_k and (not k_nonlatin_only or eid.startswith("S1") or nl):
                t += ["k_" + x for x in k.split() if len(x) >= 3]
            if use_bi:
                at = a.split()
                t += ["b_" + at[i] + "_" + at[i + 1] for i in range(len(at) - 1)]
            out.append(" ".join(t))
        return out
    return fn


variants = {
    "A n+a (old)":            dict(doc_fn=doc_factory(0, 0)),
    "B n+k+a":                dict(doc_fn=doc_factory(1, 0)),
    "C n+a+addr-bigrams":     dict(doc_fn=doc_factory(0, 1)),
    "D n+k+a+bigrams":        dict(doc_fn=doc_factory(1, 1)),
    "E D, name weight .7":    dict(doc_fn=doc_factory(1, 1), weights={"n_": 0.7, "k_": 0.7}),
    "F D, cap 12000":         dict(doc_fn=doc_factory(1, 1), max_df_abs=12000),
    "G n+a+bi, k on nonlatin": dict(doc_fn=doc_factory(1, 1, True)),
    "H G, cap 12000":         dict(doc_fn=doc_factory(1, 1, True), max_df_abs=12000),
}
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
for name, kw in variants.items():
    if only and name[0] not in only:
        continue
    t = time.time()
    pr = block(q, pool, K=60, **kw)
    pr["y"] = [int(c_ids[c] in truth[q.entity_id[s]]) for s, c in zip(pr.s1_idx, pr.c_idx)]
    rec = {K: pr[(pr.blk_rank < K)].y.sum() / n_true for K in (10, 20, 40, 60)}
    print(f"{name:<24} recall@10/20/40/60 = " + " / ".join(f"{v:.4f}" for v in rec.values()) +
          f"   ({time.time() - t:.0f}s)", flush=True)
tick("done")
