"""Re-threshold ONLY France from the saved test probabilities (no recomputation).

    python src/make_france_variants.py --thr_france 0.85 --out output_v3_fr085

US / India rows keep the model's threshold (0.66); only pairs whose S1 entity is in France use --thr_france.
Everything else (candidates, probabilities, assignment rule) is identical to output_v3, so a leaderboard difference
between variants can only come from France.  Writes matching_results.tsv only (candidate_pairs.tsv is unchanged: use
the one in output_v3/).  Needs cache/test_probs_full.npy + cache/test_pairs_full.parquet from `run_test.py`.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data import DATA
from decision import assign

ap = argparse.ArgumentParser()
ap.add_argument("--thr_france", type=float, required=True,
                help="threshold applied to --country (default France); 1.1 leaves that country's rows empty")
ap.add_argument("--country", default="France", help="country whose threshold is changed (default France)")
ap.add_argument("--model", default="models/g")
ap.add_argument("--out", required=True)
args = ap.parse_args()

base_thr = json.loads(Path(args.model + ".json").read_text())["thr"]
rd = lambda p, cols: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=cols)
s1 = rd(DATA / "test/test_source1.tsv", ["entity_id", "country"])
pool = pd.concat([rd(DATA / f"test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)], ignore_index=True)
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()

prob = np.load("cache/test_probs_full.npy")
pairs = pd.read_parquet("cache/test_pairs_full.parquet")
assert len(pairs) == len(prob), "cache mismatch"
is_fr = (s1.country.to_numpy() == args.country)[pairs.s1_idx.to_numpy()]
thr = np.where(is_fr, args.thr_france, base_thr).astype(np.float32)
matches = assign(pairs, prob, thr, s1_ids, c_ids)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in s1_ids:
        f.write(f"{sid}\t{','.join(matches.get(sid, []))}\n")

fr = s1.country.to_numpy() == args.country
n_fr = np.array([len(matches.get(k, [])) for k in s1_ids[fr]])
print(f"{out}: {args.country} thr {args.thr_france:.2f} (others {base_thr:.2f}); {args.country} avg matches/S1 "
      f"{n_fr.mean():.3f}, empty {np.mean(n_fr == 0):.3f}")
