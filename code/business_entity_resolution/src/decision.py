"""Turn pair probabilities into final matches."""
import numpy as np
import pandas as pd


def assign(pairs: pd.DataFrame, prob, thr, s1_ids, c_ids) -> dict:
    """Keep pairs with prob >= thr, then give each S2/S3 record only to its best-scoring S1 entity
    (each S2/S3 record belongs to at most one S1 entity). `thr` is a single value or one per pair (e.g. a
    stricter threshold for S1 entities whose address is shared by many other businesses). Returns
    {s1_id: [ids sorted by prob desc]}."""
    thr = np.broadcast_to(np.asarray(thr, dtype=np.float32), len(pairs))
    d = pd.DataFrame({"s": pairs.s1_idx.to_numpy(), "c": pairs.c_idx.to_numpy(), "p": np.asarray(prob)})
    d = d[d.p.to_numpy() >= thr].sort_values("p", ascending=False).drop_duplicates("c")
    out = {}
    for s, c in zip(d.s.to_numpy(), d.c.to_numpy()):
        out.setdefault(s1_ids[s], []).append(c_ids[c])
    return out


def ambiguity_threshold(amb_street_of_s1_per_pair, t_low: float, t_high: float, cut: float = 0.0) -> np.ndarray:
    """Per-pair threshold: t_high for pairs whose S1 entity's address is shared with other S1 records
    (amb_street > cut, e.g. most of France), t_low otherwise. `amb_street_of_s1_per_pair` is already
    indexed to one value per pair (the S1 record's ambiguity, not the candidate's)."""
    return np.where(np.asarray(amb_street_of_s1_per_pair) > cut, t_high, t_low).astype(np.float32)
