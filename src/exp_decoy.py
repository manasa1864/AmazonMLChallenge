"""Does a label-free per-country 'decoy word' signal separate true matches from decoys?

For every name token we compute, inside one country and using only the unlabeled input files,
    log_ratio(t) = log( (count in S2/S3 names + a) / (expected count if the token were as common as in S1 names + a) )
Words that decoy generation injects into S2/S3 names (Holdings, Groupe, Participations...) should have a high ratio.
Compare, on candidate pairs whose names are already similar, how well (a) the existing extra-word count and
(b) the new ratio features rank true matches above decoys (AUC).  Uses the cached 25% world (train labels only).
"""
import time
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from rapidfuzz import fuzz

from data import CACHE
from blocking import block

T0 = time.time()
s1, pool, truth = pd.read_pickle(CACHE / "norm_f0.25.pkl")
print(f"[{time.time() - T0:.0f}s] world: {len(s1)} S1, {len(pool)} pool", flush=True)

# ---- per-country token ratio table (label-free) ------------------------------------------------------------------
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
    top = sorted(cb, key=lambda t: -ratio[c][t] * (cb[t] > 300))[:12]
    print(f"  {c}: highest-ratio frequent tokens: {[(t, round(ratio[c][t], 2)) for t in top]}", flush=True)

# ---- candidate pairs for a subset of S1 ----------------------------------------------------------------------------
rng = np.random.default_rng(0)
q = s1.iloc[np.sort(rng.choice(len(s1), 15000, replace=False))].reset_index(drop=True)
pairs = block(q, pool, K=30)
print(f"[{time.time() - T0:.0f}s] {len(pairs)} pairs", flush=True)
ids_q, ids_p = q.entity_id.to_numpy(), pool.entity_id.to_numpy()
y = np.array([int(ids_p[c] in truth[ids_q[s]]) for s, c in zip(pairs.s1_idx, pairs.c_idx)])

rows = []
for s, c in zip(pairs.s1_idx.to_numpy(), pairs.c_idx.to_numpy()):
    na, nb = q.name_core.iat[s], pool.name_core.iat[c]
    ta, tb = set(na.split()), set(nb.split())
    country = q.country.iat[s]
    r = ratio[country]
    ex_c = [r.get(t, 0.0) for t in tb - ta]
    ex_s = [ratio[country].get(t, 0.0) for t in ta - tb]
    rows.append((fuzz.token_set_ratio(na, nb), len(ex_c), max(ex_c) if ex_c else -9, float(np.mean(ex_c)) if ex_c else -9,
                 sum(v > 1.0 for v in ex_c), len(ex_s), max(ex_s) if ex_s else -9))
F = pd.DataFrame(rows, columns=["name_sim", "ex_count", "ex_max_ratio", "ex_mean_ratio", "ex_n_decoyish", "exs_count", "exs_max_ratio"])
print(f"[{time.time() - T0:.0f}s] features built", flush=True)

# ---- compare: among pairs with SIMILAR names AND a candidate with extra words (the hard 'decoy vs variant' cases) ---
sub = (F.name_sim >= 80) & (F.ex_count >= 1)
print(f"\nhard pairs (name_sim>=80 and candidate has extra words): {sub.sum()} pairs, {y[sub].mean():.3f} are true matches")
for col, sign in (("ex_count", -1), ("ex_max_ratio", -1), ("ex_mean_ratio", -1), ("ex_n_decoyish", -1)):
    auc = roc_auc_score(y[sub], sign * F.loc[sub, col])
    print(f"  AUC of '{col}' for telling true matches from decoys: {auc:.3f}")
# both countries' extra-word table in France-like conditions: show what the ratio looks like for a few common decoy-ish words
for c in ratio:
    print(c, {t: round(ratio[c].get(t, float('nan')), 2) for t in ("holdings", "enterprises", "services", "group", "center", "industries", "international", "solutions")})
