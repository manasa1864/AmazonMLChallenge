"""Label-free comparison of TRAIN vs TEST input distributions per country and source (train/test shift check)."""
import re
import numpy as np
import pandas as pd
from data import DATA, read_tsv

NONLAT = re.compile(r"[^\x00-˿Ḁ-ỿ -⁯]")
rows = []
for split in ("train", "test"):
    for src in (1, 2, 3):
        d = read_tsv(DATA / f"{split}/{split}_source{src}.tsv")
        for c, g in d.groupby("country"):
            g = g.sample(min(150_000, len(g)), random_state=0)
            rows.append((split, f"S{src}", c, len(d[d.country == c]),
                         g.business_name.map(lambda x: bool(NONLAT.search(x))).mean(),
                         (g.business_address == "").mean(),
                         g.business_name.str.split().str.len().mean(),
                         g.business_address.str.split().str.len().mean(),
                         g.business_address.str.contains(r"\b\d{6}\b").mean(),
                         g.business_name.str.contains(r"[^\w\s.,&'()\-]").mean()))
R = pd.DataFrame(rows, columns=["split", "src", "country", "n_records", "name_nonlatin", "addr_empty", "name_tokens", "addr_tokens",
                                "addr_has_6digit", "name_has_symbols"])
pd.set_option("display.width", 220)
print(R.round(3).sort_values(["country", "src", "split"]).to_string(index=False))
# pool records per S1 record (competition density), by country
for c in ("India", "US"):
    tr = sum(R[(R.split == "train") & (R.country == c) & (R.src != "S1")].n_records) / R[(R.split == "train") & (R.country == c) & (R.src == "S1")].n_records.iloc[0]
    te = sum(R[(R.split == "test") & (R.country == c) & (R.src != "S1")].n_records) / R[(R.split == "test") & (R.country == c) & (R.src == "S1")].n_records.iloc[0]
    print(f"S2+S3 records per S1 record, {c}: train {tr:.2f}  test {te:.2f}")
