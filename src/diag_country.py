"""Label-free diagnostics of the submitted test predictions, per country:
 - how distinctive are addresses (how many S1 entities share the same street+city once house numbers are removed)?
 - among ACCEPTED pairs, how similar are the names / house numbers?"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from data import DATA, read_tsv
from text import norm_name, norm_addr

s1 = read_tsv(DATA / "test/test_source1.tsv")
pool = pd.concat([read_tsv(DATA / f"test/test_source{i}.tsv") for i in (2, 3)], ignore_index=True)
p = np.load("cache/test_probs_full.npy")
pr = pd.read_parquet("cache/test_pairs_full.parquet")
pr["p"] = p
pr["country"] = s1.country.to_numpy()[pr.s1_idx.to_numpy()]

# ---- 1) address distinctiveness: S1 records sharing street+city (house numbers removed) ----------------------------
_num = re.compile(r"\b\d+\w*\b")


def street_key(a):
    return _num.sub("", norm_addr(a)).replace("  ", " ").strip()


rows = []
for c in s1.country.unique():
    sub = s1[s1.country == c].sample(min(60000, (s1.country == c).sum()), random_state=0)
    keys = sub.business_address.map(street_key)
    full = sub.business_address.map(norm_addr)
    vc_street = keys.value_counts()
    vc_full = full.value_counts()
    rows.append((c, len(sub), (keys.map(vc_street) > 1).mean(), keys.map(vc_street).mean(), (full.map(vc_full) > 1).mean()))
print("Address distinctiveness within a 60k S1 sample per country:")
print(pd.DataFrame(rows, columns=["country", "n", "share_S1_sharing_street+city", "avg_S1_per_street+city", "share_sharing_full_addr"]).round(3).to_string(index=False))

# ---- 2) accepted pairs: name / number agreement -----------------------------------------------------------------------
thr = 0.69
acc = pr[pr.p >= thr]
out = []
for c in ["US", "India", "France"]:
    a = acc[acc.country == c].sample(80000, random_state=1)
    A = s1.iloc[a.s1_idx.to_numpy()]
    B = pool.iloc[a.c_idx.to_numpy()]
    nm = np.array([fuzz.token_set_ratio(norm_name(x)[1], norm_name(y)[1]) for x, y in zip(A.business_name, B.business_name)])
    na = [(_num.findall(norm_addr(x)) or [""])[0] for x in A.business_address]
    nb = [(_num.findall(norm_addr(y)) or [""])[0] for y in B.business_address]
    same_num = np.array([x != "" and x == y for x, y in zip(na, nb)])
    no_addr = np.array([y == "" for y in B.business_address])
    out.append((c, (nm < 50).mean(), (nm < 30).mean(), same_num.mean(), ((nm < 50) & ~same_num & ~no_addr).mean(), no_addr.mean(),
                a.p.mean()))
print("\nAccepted pairs (80k sampled per country):")
print(pd.DataFrame(out, columns=["country", "name_sim<50", "name_sim<30", "same_house_no", "name<50 & diff/absent no.", "cand_addr_empty", "mean_p"]).round(3).to_string(index=False))
