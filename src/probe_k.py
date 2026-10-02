"""Cheap probe: does raising K on the MAIN blocking channel alone (no extra channels) meaningfully raise recall?

Single top-60 retrieval with TODAY's block()/make_doc() (current text.py + blocking.py, not the stale exp_blocking.py
numbers from before the skeleton/bigram tokens were added) over the full 25% world (551k S1, 2.58M pool - already
cached), so every smaller K's recall is read off the same run's blk_rank column for free. Reports recall AND the
oracle F0.5 ceiling at K = 30 (current), 40, 45, 60, so the decision is based on the metric that matters, not just recall.
"""
import time
import numpy as np
import pandas as pd

from data import CACHE
from blocking import block
from metric import f05_macro

T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

s1, pool, truth = pd.read_pickle(CACHE / "norm_f0.25.pkl")
tick(f"world: {len(s1)} S1, {len(pool)} pool (cached, no rebuild)")

pairs = block(s1, pool, K=60, n_jobs=10)          # single channel, current make_doc(); K=60 covers every K we test
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()
pairs["y"] = [int(c_ids[c] in truth[s1_ids[s]]) for s, c in zip(pairs.s1_idx, pairs.c_idx)]
n_true = sum(len(v) for v in truth.values())
tick(f"blocked: {len(pairs)} pairs")

rows = []
for K in (30, 40, 45, 60):
    sub = pairs[pairs.blk_rank < K]
    recall = sub.y.sum() / n_true
    oracle = {}
    for s, c, y in zip(sub.s1_idx.to_numpy(), sub.c_idx.to_numpy(), sub.y.to_numpy()):
        if y:
            oracle.setdefault(s1_ids[s], set()).add(c_ids[c])
    ceiling = f05_macro(truth, oracle)
    avg_cand = len(sub) / len(s1)
    rows.append((K, avg_cand, recall, ceiling))
    tick(f"K={K:<3} avg_cand/S1={avg_cand:5.1f}  recall={recall:.4f}  oracle F0.5 ceiling={ceiling:.4f}")

R = pd.DataFrame(rows, columns=["K", "avg_cand_per_S1", "recall", "oracle_ceiling"])
print("\n" + R.round(4).to_string(index=False))
base = R[R.K == 30].iloc[0]
print("\nvs current (K=30):")
for _, r in R[R.K > 30].iterrows():
    print(f"  K={int(r.K):<3} candidates +{(r.avg_cand_per_S1 / base.avg_cand_per_S1 - 1) * 100:4.0f}%  "
          f"recall {r.recall - base.recall:+.4f}  oracle ceiling {r.oracle_ceiling - base.oracle_ceiling:+.4f}")
tick("done")
