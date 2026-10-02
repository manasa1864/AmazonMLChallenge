"""Build a labeled SOURCE sample of stage-1 pair features at TEST SCALE (all train S1 vs the full pool), using exactly the
feature set / vocabulary of a saved model.  Used by france_selftrain.py to anchor self-training so the adapted model
cannot drift towards its own guesses.

    python src/build_source.py --model models/g --fracS 0.06 --out cache/source_stage1.npz
Only training data is used.  Peak memory ~ the training run's blocking stage (close other heavy programs).
"""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from data import DATA, read_tsv, parse_gt
from text import normalise_frame
from blocking import block
from features import loop_features, group_features, sim_avg_only, add_ambiguity

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/g")
ap.add_argument("--fracS", type=float, default=0.06, help="share of train S1 entities kept as the source sample")
ap.add_argument("--out", default="cache/source_stage1.npz")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
meta = json.loads(Path(args.model + ".json").read_text())

s1 = read_tsv(DATA / "train/train_source1.tsv")
pool = pd.concat([read_tsv(DATA / f"train/train_source{i}.tsv") for i in (2, 3)], ignore_index=True)
tick(f"train world: {len(s1)} S1, {len(pool)} pool")
normalise_frame(s1); normalise_frame(pool); add_ambiguity(s1); add_ambiguity(pool)
for d in (s1, pool):
    d.drop(columns=["business_name", "business_address"], inplace=True)
gc.collect()
tick("normalised")

pairs = block(s1, pool, K=meta["K"], max_df_abs=meta["max_df_abs"], channels=meta.get("channels", False))
pairs = pairs.astype({"s1_idx": np.int32, "c_idx": np.int32, "blk_rank": np.int16})
pairs = pairs.sort_values(["s1_idx", "blk_rank"], kind="stable").reset_index(drop=True)
s1_ids = s1.entity_id.to_numpy()
tick(f"blocking: {len(pairs)} pairs")

truth = parse_gt(read_tsv(DATA / "train/train_ground_truth.tsv"))
c2s = {c: s for s, cs in truth.items() for c in cs}
del truth
s1pos = pd.Series(np.arange(len(s1), dtype=np.int64), index=s1.entity_id)
owner = np.nan_to_num(pool.entity_id.map(c2s).map(s1pos).to_numpy(dtype=np.float64), nan=-1).astype(np.int32)
del c2s, s1pos
y = (owner[pairs.c_idx.to_numpy()] == pairs.s1_idx.to_numpy()).astype(np.int8)
del owner
gc.collect()

sim = sim_avg_only(s1, pool, pairs)
G = group_features(pairs, sim, s1, pool)          # rival / ambiguity features at full test scale (all S1 are queries)
del sim
tick("global comparison features")

rng = np.random.default_rng(args.seed)
keep_s1 = rng.random(len(s1)) < args.fracS
iS = np.flatnonzero(keep_s1[pairs.s1_idx.to_numpy()])
L = loop_features(s1, pool, pairs.iloc[iS], meta["vocab"])
F = pd.concat([L, G.iloc[iS]], axis=1)[meta["s1_features"]]
Path(args.out).parent.mkdir(exist_ok=True)
np.savez_compressed(args.out, X=F.to_numpy(dtype=np.float32), y=y[iS], cols=np.array(meta["s1_features"]))
tick(f"saved {len(iS)} source pairs ({keep_s1.sum()} S1 entities, {y[iS].mean():.3f} positive) -> {args.out}")
