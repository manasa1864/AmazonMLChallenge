"""Compare ideas on the SAME held-out training world (same sample, same 2-fold split by S1 entity, same F0.5).

    python src/exp_variants.py --frac 0.12

Variants: logistic-regression baseline, stage-1 LightGBM, assignment on/off, + S2<->S3 consensus features,
two-threshold "uncertain zone" rule, and blocking-channel union recall (name / address / char-n-gram channels).
"""
import argparse
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import load_train_sample
from text import normalise_frame
from blocking import block
from features import pair_features, build_vocab, context_features, consensus_features
from metric import f05_macro
from decision import assign

LGB_PARAMS = dict(n_estimators=350, learning_rate=0.07, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, n_jobs=-1, verbose=-1)
ap = argparse.ArgumentParser()
ap.add_argument("--frac", type=float, default=0.12)
ap.add_argument("--K", type=int, default=30)
ap.add_argument("--max_df_abs", type=int, default=12000)
args = ap.parse_args()
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

s1, pool, truth = load_train_sample(args.frac, 0)
normalise_frame(s1)
normalise_frame(pool)
pairs = block(s1, pool, K=args.K, max_df_abs=args.max_df_abs)
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()
y = np.array([int(c_ids[c] in truth[s1_ids[s]]) for s, c in zip(pairs.s1_idx, pairs.c_idx)])
n_true = sum(len(v) for v in truth.values())
vocab = build_vocab(s1, pool, pairs)
F = pair_features(s1, pool, pairs, vocab)
X = F.to_numpy(dtype=np.float32)
fold = pairs.s1_idx.to_numpy() % 2
tick(f"pairs={len(pairs)}, features={F.shape[1]}, blocking pair recall={y.sum() / n_true:.4f}")


def decide(prob, thr, use_assign=True):
    if use_assign:
        return {k: set(v) for k, v in assign(pairs, prob, thr, s1_ids, c_ids).items()}
    out = {}
    for s, c in zip(pairs.s1_idx.to_numpy()[prob >= thr], pairs.c_idx.to_numpy()[prob >= thr]):
        out.setdefault(s1_ids[s], set()).add(c_ids[c])
    return out


def sweep(name, prob, grid=np.arange(0.2, 0.97, 0.05), use_assign=True):
    best = max(((f05_macro(truth, decide(prob, t, use_assign)), t) for t in grid))
    fine = np.arange(max(best[1] - 0.05, 0.05), min(best[1] + 0.051, 0.99), 0.01)
    best = max(((f05_macro(truth, decide(prob, t, use_assign)), t) for t in fine))
    print(f"  {name:<46} F0.5 = {best[0]:.4f} @ thr {best[1]:.2f}", flush=True)
    return best


def oof(fit_predict, tag):
    o = np.zeros(len(y))
    for k in (0, 1):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        o[te] = fit_predict(tr, te)
    tick(tag)
    return o


rng = np.random.default_rng(0)


def lr_fit_predict(tr, te, Xm=X):
    sub = rng.choice(tr, min(2_000_000, len(tr)), replace=False)
    sc = StandardScaler().fit(Xm[sub])
    m = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Xm[sub]), y[sub])
    return m.predict_proba(sc.transform(Xm[te]))[:, 1]


lgb_fit_predict = lambda Xm: (lambda tr, te: lgb.LGBMClassifier(**LGB_PARAMS).fit(Xm[tr], y[tr]).predict_proba(Xm[te])[:, 1])

print("\n=== Results (same held-out world; official macro F0.5) ===")
sweep("cosine only (TF-IDF blocking score)", F.blk_score.to_numpy())
p_lr = oof(lambda tr, te: lr_fit_predict(tr, te), "LR done")
sweep("Logistic Regression (stage-1 features)", p_lr)
p1 = oof(lgb_fit_predict(X), "stage-1 LightGBM done")
sweep("LightGBM stage-1", p1)
sweep("LightGBM stage-1, NO one-S1-per-record rule", p1, use_assign=False)

C = context_features(pairs, p1)
CS = consensus_features(pool, pairs, p1)
tick("context + consensus features built")
X2 = np.hstack([X, C.to_numpy(dtype=np.float32)])
X2c = np.hstack([X2, CS.to_numpy(dtype=np.float32)])
p2c = oof(lgb_fit_predict(X2c), "stage-2 (+consensus) done")
b_c = sweep("LightGBM stage-2 + consensus features", p2c)
sweep("  same, NO one-S1-per-record rule", p2c, use_assign=False)
print("  (reference: stage-2 without consensus features scored 0.9785 in run_validation.py)")

# two-threshold "uncertain zone": between T_low and T_high accept only if extra deterministic evidence holds
cols = list(F.columns)
ix = {c: cols.index(c) for c in ("num_first_eq", "nm_tset", "ad_tset", "zip_match")}
T_high = b_c[1]
for name, rule in {
    "house-number equal": X[:, ix["num_first_eq"]] == 1,
    "house-number equal & name>=60": (X[:, ix["num_first_eq"]] == 1) & (X[:, ix["nm_tset"]] >= 60),
    "address>=90 & name>=50": (X[:, ix["ad_tset"]] >= 90) & (X[:, ix["nm_tset"]] >= 50),
}.items():
    for t_low in (0.3, 0.5):
        padj = np.where((p2c >= t_low) & (p2c < T_high) & rule, T_high, p2c)
        f = f05_macro(truth, decide(padj, T_high))
        print(f"  two-threshold: [{t_low:.1f},{T_high:.2f}) accepted if {name:<32} F0.5 = {f:.4f}")
    # stricter: inside the zone below T_high+0.15 require the rule (i.e. reject if rule fails)
    for t_hi2 in (T_high + 0.1, T_high + 0.2):
        padj = np.where((p2c >= T_high) & (p2c < t_hi2) & ~rule, 0.0, p2c)
        f = f05_macro(truth, decide(padj, T_high))
        print(f"  two-threshold: [{T_high:.2f},{t_hi2:.2f}) rejected unless {name:<32} F0.5 = {f:.4f}")

# ---- blocking channel union: recall / candidates per S1 ------------------------------------
q_ids = s1.sample(60000, random_state=0).index.to_numpy()
qs = s1.loc[q_ids].reset_index(drop=True)
main = pairs[np.isin(pairs.s1_idx.to_numpy(), q_ids)]
pos = {int(o): i for i, o in enumerate(q_ids)}
n_true_q = sum(len(truth[k]) for k in qs.entity_id)


def key(pr_s, pr_c):
    return set(zip(pr_s.tolist(), pr_c.tolist()))


def recall_of(pset):
    return sum(1 for s, c in pset if c_ids[c] in truth[qs.entity_id[s]]) / n_true_q


main_set = key(np.array([pos[int(o)] for o in main.s1_idx.to_numpy()]), main.c_idx.to_numpy())


def doc_name(df):
    return [" ".join("n_" + t for t in n.split()) for n in df.name_core.tolist()]


def doc_addr(df):
    o = []
    for a in df.addr.tolist():
        t = a.split()
        o.append(" ".join(["a_" + x for x in t] + ["b_" + t[i] + "_" + t[i + 1] for i in range(len(t) - 1)]))
    return o


def doc_chr(df):
    o = []
    for n in df.name_core.tolist():
        s = n.replace(" ", "")
        o.append(" ".join("c_" + s[i:i + 3] for i in range(max(len(s) - 2, 1))))
    return o


print("\n=== Blocking channels (60k queries, same pool) ===")
print(f"  current combined channel, top-{args.K}: recall {recall_of(main_set):.4f}, {len(main_set) / len(qs):.1f} cand/S1")
union = set(main_set)
for name, fn, cap in (("name-word channel top-10", doc_name, 12000), ("address channel top-10", doc_addr, 12000),
                      ("char-3gram name channel top-10", doc_chr, 6000)):
    t = time.time()
    pr = block(qs, pool, K=10, max_df_abs=cap, doc_fn=fn)
    ch = key(pr.s1_idx.to_numpy(), pr.c_idx.to_numpy())
    union |= ch
    print(f"  {name:<32} alone recall {recall_of(ch):.4f}; union so far: recall {recall_of(union):.4f}, "
          f"{len(union) / len(qs):.1f} cand/S1  ({time.time() - t:.0f}s)", flush=True)
tick("done")
