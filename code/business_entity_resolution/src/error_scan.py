"""Fresh, honest look at what the CURRENT best model (models/g) actually gets wrong, on held-out training data,
hunting for a specific pattern not already covered by existing features (rather than another generic ML trick).

Uses the cached 25% world; ~18% of its S1 entities may overlap models/g's own training subset T (one of the two
averaged fold models could have seen them) - a mild optimism bias, noted, not hidden. Errors found here are still
real errors (if the model gets something wrong even having possibly partly seen it, that is not a good sign)."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import json
from pathlib import Path

from data import CACHE
from blocking import block
from features import pair_features, context_features, add_ambiguity
from decision import assign
from metric import f05_macro

T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
MODEL = "models/g"
meta = json.loads(Path(MODEL + ".json").read_text())
load = lambda stage: [lgb.Booster(model_file=f"{MODEL}_{stage}_fold{i}.txt") for i in (0, 1)]
m1, m2 = load("s1"), load("s2")

s1, pool, truth = pd.read_pickle(CACHE / "norm_f0.25.pkl")
if "addr_street" not in s1.columns:           # cached frame predates the street key added to text.py
    for d in (s1, pool):
        d["addr_street"] = [" ".join(t for t in a.split() if not any(ch.isdigit() for ch in t)) for a in d.addr]
add_ambiguity(s1); add_ambiguity(pool)
rng = np.random.default_rng(3)
qidx = np.sort(rng.choice(len(s1), 45000, replace=False))
q = s1.iloc[qidx].reset_index(drop=True)
tick(f"world: {len(q)} S1 sample (of {len(s1)}), {len(pool)} pool")

pairs = block(q, pool, K=meta["K"], max_df_abs=meta["max_df_abs"], channels=meta.get("channels", False))
s1_ids, c_ids = q.entity_id.to_numpy(), pool.entity_id.to_numpy()
tick(f"blocked: {len(pairs)} pairs")

F = pair_features(q, pool, pairs, meta["vocab"])[meta["s1_features"]]
p1 = np.mean([m.predict(F.to_numpy(dtype=np.float32)) for m in m1], axis=0)
F2 = pd.concat([F, context_features(pairs, p1)], axis=1)[meta["s2_features"]]
prob = np.mean([m.predict(F2.to_numpy(dtype=np.float32)) for m in m2], axis=0)
tick("scored")

matches = {k: set(v) for k, v in assign(pairs, prob, meta["thr"], s1_ids, c_ids).items()}
truthQ = {k: truth[k] for k in s1_ids}
f = f05_macro(truthQ, matches)
tick(f"F0.5 on this sample = {f:.4f} (indicative only - partial overlap with model's own training data)")

cand = {}
for s, c in zip(pairs.s1_idx.to_numpy(), pairs.c_idx.to_numpy()):
    cand.setdefault(s1_ids[s], set()).add(c_ids[c])

info = pool.set_index("entity_id")[["business_name", "business_address", "country", "amb_street"]]
qi = q.set_index("entity_id")[["business_name", "business_address", "country"]]
show = lambda i, d: f"{i} | {d.loc[i].business_name!r} | {d.loc[i].business_address!r}"

fps = [(k, c) for k, v in matches.items() for c in v if c not in truthQ[k]]
fns_in_cand = [(k, c) for k, v in truthQ.items() for c in v if c not in matches.get(k, set()) and c in cand.get(k, set())]
fns_no_cand = [(k, c) for k, v in truthQ.items() for c in v if c not in matches.get(k, set()) and c not in cand.get(k, set())]
print(f"\n{len(fps)} false positives, {len(fns_in_cand)} false negatives (model scored too low), "
      f"{len(fns_no_cand)} false negatives (never retrieved by blocking)")

rng2 = np.random.default_rng(1)
print("\n=== FALSE POSITIVES (model said match, ground truth says no) ===")
for k, c in [fps[i] for i in rng2.choice(len(fps), min(35, len(fps)), replace=False)]:
    print(f" S1: {show(k, qi)}\n WRONG: {show(c, info)}  amb_street={info.loc[c].amb_street:.0f}\n")

print("\n=== FALSE NEGATIVES the model SAW but scored too low ===")
for k, c in [fns_in_cand[i] for i in rng2.choice(len(fns_in_cand), min(35, len(fns_in_cand)), replace=False)]:
    print(f" S1: {show(k, qi)}\n MISSED: {show(c, info)}  amb_street={info.loc[c].amb_street:.0f}\n")

print("\n=== FALSE NEGATIVES blocking never retrieved ===")
for k, c in [fns_no_cand[i] for i in rng2.choice(len(fns_no_cand), min(20, len(fns_no_cand)), replace=False)]:
    print(f" S1: {show(k, qi)}\n NOT-RETRIEVED: {show(c, info)}\n")

tick("done")
