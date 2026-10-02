"""Quick EDA on the training data: match structure, singleton rate, sample matched pairs."""
import pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"
rd = lambda p: pd.read_csv(D / p, sep="\t", dtype=str, keep_default_na=False, quoting=3)

s1, s2, s3 = (rd(f"train/train_source{i}.tsv") for i in (1, 2, 3))
gt = rd("train/train_ground_truth.tsv")
print("rows", len(s1), len(s2), len(s3), len(gt))
print("country counts S1:\n", s1.country.value_counts().to_string())
print("country counts S2:\n", s2.country.value_counts().to_string())
print("country counts S3:\n", s3.country.value_counts().to_string())

gt["ids"] = gt.matched_entity_ids.map(lambda x: [i for i in x.split(",") if i])
gt["n"] = gt.ids.map(len)
gt["n2"] = gt.ids.map(lambda l: sum(i.startswith("S2-") for i in l))
gt["n3"] = gt.ids.map(lambda l: sum(i.startswith("S3-") for i in l))
print("\nsingleton rate:", (gt.n == 0).mean())
print("matches per S1 (n distribution):\n", gt.n.value_counts().sort_index().head(15).to_string())
print("mean n2, n3:", gt.n2.mean(), gt.n3.mean())

flat = gt.explode("ids").dropna(subset=["ids"])
flat = flat[flat.ids.astype(bool)]
print("\nmatched S2/S3 ids:", len(flat), "unique:", flat.ids.nunique(), "(dup => a record matches >1 S1)")
allids = set(s2.entity_id) | set(s3.entity_id)
print("gt ids not in S2/S3 files:", (~flat.ids.isin(allids)).sum())
print("S2 records with a match:", s2.entity_id.isin(flat.ids).mean(), " S3:", s3.entity_id.isin(flat.ids).mean())
print("S1 ids duplicated:", s1.entity_id.duplicated().sum(), " gt ids in s1:", gt.source1_entity_id.isin(s1.entity_id).mean())

by_id = pd.concat([s1, s2, s3]).set_index("entity_id")
print("\n=== sample matched groups")
for _, r in gt[gt.n >= 3].sample(8, random_state=1).iterrows():
    print("-" * 90)
    for i in [r.source1_entity_id] + r.ids:
        x = by_id.loc[i]
        print(f"{i:<14} | {x.business_name} | {x.business_address} | {x.country}")

print("\n=== sample singletons")
for _, r in gt[gt.n == 0].sample(5, random_state=1).iterrows():
    x = by_id.loc[r.source1_entity_id]
    print(f"{r.source1_entity_id:<14} | {x.business_name} | {x.business_address} | {x.country}")

print("\nempty address rate  S1/S2/S3:", [(d.business_address == "").mean().round(4) for d in (s1, s2, s3)])
print("empty name rate     S1/S2/S3:", [(d.business_name == "").mean().round(4) for d in (s1, s2, s3)])
