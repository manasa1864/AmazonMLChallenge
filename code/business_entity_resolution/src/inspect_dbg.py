"""Sanity-check a (debug) test run: per-country match counts and France examples."""
import sys
import numpy as np
import pandas as pd
from data import DATA, read_tsv

out = sys.argv[1] if len(sys.argv) > 1 else "output_dbg"
s1 = read_tsv(DATA / "test/test_source1.tsv")
m = pd.read_csv(f"{out}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
c = pd.read_csv(f"{out}/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
df = s1.merge(m, left_on="entity_id", right_on="source1_entity_id").merge(c, on="source1_entity_id")
df["n_match"] = df.matched_entity_ids.map(lambda x: len([i for i in x.split(",") if i]))
df["n_cand"] = df.candidate_entity_ids.map(lambda x: len([i for i in x.split(",") if i]))
print(df.groupby("country").agg(n=("n_match", "size"), avg_match=("n_match", "mean"),
                                empty=("n_match", lambda s: (s == 0).mean()), avg_cand=("n_cand", "mean")))
print(df.groupby("country").n_match.value_counts(normalize=True).unstack().round(3).iloc[:, :10])

pool = pd.concat([read_tsv(DATA / f"test/test_source{i}.tsv") for i in (2, 3)]).set_index("entity_id")
ids = set(df[df.country == "France"].matched_entity_ids.str.split(",").explode()) - {""}
rng = np.random.default_rng(0)
fr = df[(df.country == "France")].sample(8, random_state=1)
for _, r in fr.iterrows():
    print("-" * 100)
    print("S1 :", r.entity_id, "|", r.business_name, "|", r.business_address)
    for i in [x for x in r.matched_entity_ids.split(",") if x][:6]:
        p = pool.loc[i]
        print("  M :", i, "|", p.business_name, "|", p.business_address, "|", p.country)
