"""Pair features for the matcher.

Stage-1 features: string similarities, phonetic-skeleton similarity, extra-token / extra-number counts,
decoy-word flags, blocking-score context and competition between S1 rivals for the same S2/S3 record.
Stage-2 (context) features: built from stage-1 out-of-fold probabilities.
"""
import re
from collections import Counter
import numpy as np
import pandas as pd
from par import par_map
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

_NUM = re.compile(r"\d+")
_ZIP = re.compile(r"\b\d{5,6}\b")

BASE_COLS = ["nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw", "nm_nospace", "nm_jac", "nm_full_tset",
             "nm_lenratio", "nm_first_jw", "ad_tset", "ad_tsort", "ad_jac", "ad_partial", "ad_empty",
             "num_first_eq", "num_first_ratio", "num_jac", "zip_match", "c_nonlatin", "c_src",
             "sk_ratio", "sk_tset", "sk_jac", "ex_c", "ex_s", "ex_c_frac", "ex_s_frac", "numx_c", "numx_s"]


def _jac(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_vocab(s1: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame, top: int = 60, max_pairs: int = 400_000):
    """Most frequent name tokens that a candidate has and its S1 record lacks ('holdings', 'enterprises', ...).
    Frequency only (no labels used); the model decides which of them signal a decoy."""
    sub = pairs.sample(min(max_pairs, len(pairs)), random_state=0)
    A, B = s1.name_core.to_numpy(), pool.name_core.to_numpy()
    cnt = Counter()
    for i, j in zip(sub.s1_idx.to_numpy(), sub.c_idx.to_numpy()):
        cnt.update(set(B[j].split()) - set(A[i].split()))
    return [t for t, _ in cnt.most_common(top)]


def _loop(A_core, A_full, A_addr, A_sk, B_core, B_full, B_addr, B_sk, B_nl, B_src, vocab):
    """Per-pair similarity features for aligned per-pair arrays (one entry per candidate pair)."""
    vidx = {t: k for k, t in enumerate(vocab)}
    nb_base = len(BASE_COLS)
    out = np.zeros((len(A_core), nb_base + len(vocab)), dtype=np.float32)
    for k in range(len(A_core)):
        a, b = A_core[k], B_core[k]
        aa, ba = A_addr[k], B_addr[k]
        ta, tb = set(a.split()), set(b.split())
        sa, sb = A_sk[k], B_sk[k]
        na, nb = _NUM.findall(aa), _NUM.findall(ba)
        za, zb = _ZIP.findall(aa), _ZIP.findall(ba)
        ex_c, ex_s = tb - ta, ta - tb
        out[k, :nb_base] = (
            fuzz.ratio(a, b),
            fuzz.token_sort_ratio(a, b),
            fuzz.token_set_ratio(a, b),
            fuzz.partial_ratio(a, b),
            JaroWinkler.similarity(a, b) * 100,
            fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")),
            _jac(ta, tb),
            fuzz.token_set_ratio(A_full[k], B_full[k]),
            min(len(a), len(b)) / max(len(a), len(b), 1),
            (JaroWinkler.similarity(a.split()[0], b.split()[0]) * 100) if (ta and tb) else 0.0,
            fuzz.token_set_ratio(aa, ba) if ba else -1,
            fuzz.token_sort_ratio(aa, ba) if ba else -1,
            _jac(set(aa.split()), set(ba.split())) if ba else -1,
            fuzz.partial_ratio(aa, ba) if ba else -1,
            0 if ba else 1,
            (1.0 if (na and nb and na[0] == nb[0]) else 0.0) if ba else -1,
            (fuzz.ratio(na[0], nb[0]) if (na and nb) else -1) if ba else -1,
            _jac(set(na), set(nb)) if (na and nb) else -1,
            (1.0 if set(za) & set(zb) else (-1.0 if (za and zb) else 0.0)),
            B_nl[k],
            B_src[k],
            fuzz.ratio(sa, sb),
            fuzz.token_set_ratio(sa, sb),
            _jac(set(sa.split()), set(sb.split())),
            len(ex_c), len(ex_s),
            len(ex_c) / max(len(tb), 1), len(ex_s) / max(len(ta), 1),
            len(set(nb) - set(na)) if ba else -1, len(set(na) - set(nb)) if ba else -1,
        )
        for t in ex_c:
            v = vidx.get(t)
            if v is not None:
                out[k, nb_base + v] = 1.0
    return out


def _sim_only(A_core, A_addr, B_core, B_addr):
    """Just the two similarities that make up sim_avg (cheap pass used to build competition features globally)."""
    nm = np.fromiter((fuzz.token_set_ratio(a, b) for a, b in zip(A_core, B_core)), dtype=np.float32, count=len(A_core))
    ad = np.fromiter(((fuzz.token_set_ratio(a, b) if b else 0.0) for a, b in zip(A_addr, B_addr)),
                     dtype=np.float32, count=len(A_core))
    return (nm + ad) / 2


def _arrays(s1, pool):
    cols = ("name_core", "name_full", "addr", "name_skel", "nonlatin")
    return ({c: s1[c].to_numpy() for c in cols}, {c: pool[c].to_numpy() for c in cols},
            pool.entity_id.str[:2].map({"S2": 2, "S3": 3}).to_numpy())


def sim_avg_only(s1, pool, pairs, n_jobs: int = 20, chunk: int = 500_000) -> np.ndarray:
    A, B, _ = _arrays(s1, pool)
    si, ci = pairs.s1_idx.to_numpy(), pairs.c_idx.to_numpy()
    parts = par_map(_sim_only, [(A["name_core"][si[lo:lo + chunk]], A["addr"][si[lo:lo + chunk]],
                                 B["name_core"][ci[lo:lo + chunk]], B["addr"][ci[lo:lo + chunk]])
                                for lo in range(0, len(pairs), chunk)], n_jobs=n_jobs)
    return np.concatenate(parts)


def loop_features(s1, pool, pairs, vocab, n_jobs: int = 20, chunk: int = 200_000) -> pd.DataFrame:
    """Per-pair string features (independent of other pairs) + sim_avg."""
    si, ci = pairs.s1_idx.to_numpy(), pairs.c_idx.to_numpy()
    A, B, B_src = _arrays(s1, pool)

    def task(lo, hi):
        s, c = si[lo:hi], ci[lo:hi]
        return (A["name_core"][s], A["name_full"][s], A["addr"][s], A["name_skel"][s],
                B["name_core"][c], B["name_full"][c], B["addr"][c], B["name_skel"][c],
                B["nonlatin"][c], B_src[c], vocab)

    bounds = [(lo, min(lo + chunk, len(pairs))) for lo in range(0, len(pairs), chunk)]
    parts = par_map(_loop, [task(lo, hi) for lo, hi in bounds], n_jobs=n_jobs)
    out = np.concatenate(parts) if parts else np.zeros((0, len(BASE_COLS) + len(vocab)), dtype=np.float32)
    f = pd.DataFrame(out, columns=BASE_COLS + [f"xw_{t}" for t in vocab], index=pairs.index)
    f["s1_nonlatin"] = A["nonlatin"][si]
    f["sim_avg"] = (f.nm_tset + f.ad_tset.clip(lower=0)) / 2
    return f


def add_ambiguity(df: pd.DataFrame) -> pd.DataFrame:
    """Label-free 'how distinctive is this record's address' columns, computed within one file (S1 file or S2+S3 pool)
    and one country: number of OTHER records sharing the exact normalised address / the street+city (house numbers
    removed), per million records of that country. Density (not raw count) so it is comparable between a sampled
    world and the full data, and between countries. -1 when the address is empty.
    Why: in the US / India an address match is almost an identifier; in France many businesses share a street."""
    ccode = pd.factorize(df["country"])[0].astype(np.int64)
    n_country = np.bincount(ccode)[ccode].astype(np.float64)
    for col, key in (("amb_full", "addr"), ("amb_street", "addr_street")):
        k = pd.factorize(df[key])[0].astype(np.int64)
        dense = pd.factorize(ccode * (int(k.max()) + 1) + k)[0]            # id of the (country, address) pair
        cnt = np.bincount(dense)[dense].astype(np.float64)
        val = (cnt - 1) / n_country * 1e6
        val[(df[key] == "").to_numpy()] = -1
        df[col] = val.astype(np.float32)
    return df


def add_name_ambiguity(df: pd.DataFrame) -> pd.DataFrame:
    """Label-free 'how common/generic is this record's name' column (same recipe as add_ambiguity, applied to
    name_core instead of address): records per million of that country sharing the exact normalised name.
    Error scan finding: false positives cluster heavily where the CANDIDATE has an empty address (no way to
    disambiguate) and a generic name the model over-trusts - this gives it the missing signal."""
    n_country = df.groupby("country").country.transform("size").to_numpy(dtype=np.float64)
    cnt = df.groupby(["country", "name_core"]).country.transform("size").to_numpy(dtype=np.float64)
    df["name_amb"] = ((cnt - 1) / n_country * 1e6).astype(np.float32)
    return df


def group_features(pairs: pd.DataFrame, sim_avg, s1: pd.DataFrame = None, pool: pd.DataFrame = None) -> pd.DataFrame:
    """Features that compare a pair with the other pairs of the same S1 entity / the same S2-S3 record.
    Small arrays only, so it can be computed globally even when the per-pair features are built in chunks."""
    ci = pairs.c_idx.to_numpy()
    f = pd.DataFrame(index=pairs.index)
    f["blk_score"] = pairs.blk_score.to_numpy()
    f["blk_rank"] = pairs.blk_rank.to_numpy()
    for col in ("r_main", "r_name", "r_addr", "r_chr"):     # which blocking channels retrieved this pair, and at what rank
        if col in pairs.columns:
            f[col] = pairs[col].to_numpy()
    g = pairs.groupby("s1_idx").blk_score
    f["blk_top1"] = g.transform("max").to_numpy()
    f["blk_ratio"] = f.blk_score / f.blk_top1.clip(lower=1e-6)
    f["n_cands"] = g.transform("size").to_numpy()
    # competition: an S2/S3 record belongs to at most one S1 entity, so compare S1 rivals for this record
    for v, key in ((f.blk_score.to_numpy(), "rv_blk"), (np.asarray(sim_avg), "rv_sim")):
        gm = pd.DataFrame({"c": ci, "v": v}).groupby("c").v
        f[key + "_rank"] = gm.rank(ascending=False, method="min").to_numpy()
        f[key + "_gap"] = v - gm.transform("max").to_numpy()   # 0 if this S1 is the best rival
        f[key + "_n"] = gm.transform("size").to_numpy()
    if s1 is not None and "amb_street" in s1.columns:        # address ambiguity of the S1 record and of the candidate
        si = pairs.s1_idx.to_numpy()
        f["s1_amb_full"] = s1.amb_full.to_numpy()[si]
        f["s1_amb_street"] = s1.amb_street.to_numpy()[si]
        f["c_amb_full"] = pool.amb_full.to_numpy()[ci]
        f["c_amb_street"] = pool.amb_street.to_numpy()[ci]
    if s1 is not None and "name_amb" in s1.columns:
        f["s1_name_amb"] = s1.name_amb.to_numpy()[si]
        f["c_name_amb"] = pool.name_amb.to_numpy()[ci]
        f["c_name_amb_x_addr_empty"] = f["c_name_amb"] * (pool.addr.to_numpy()[ci] == "")
    return f.astype(np.float32)


def pair_features(s1: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame, vocab, n_jobs: int = 20) -> pd.DataFrame:
    L = loop_features(s1, pool, pairs, vocab, n_jobs=n_jobs)
    return pd.concat([L, group_features(pairs, L.sim_avg.to_numpy(), s1, pool)], axis=1)


def _sim2(A_core, A_addr, B_core, B_addr):
    nm = np.fromiter((fuzz.token_set_ratio(a, b) for a, b in zip(A_core, B_core)), dtype=np.float32, count=len(A_core))
    ad = np.fromiter(((fuzz.token_set_ratio(a, b) if (a and b) else -1.0) for a, b in zip(A_addr, B_addr)),
                     dtype=np.float32, count=len(A_core))
    return np.stack([nm, ad], axis=1)


def consensus_features(pool: pd.DataFrame, pairs: pd.DataFrame, p1, n_jobs: int = 20, chunk: int = 500_000) -> pd.DataFrame:
    """S2<->S3 (candidate<->candidate) agreement: similarity of each candidate to the entity's best other candidate
    (by stage-1 probability). True matches of one S1 record resemble each other; decoys usually resemble only the S1."""
    s, c, p = pairs.s1_idx.to_numpy(), pairs.c_idx.to_numpy(), np.asarray(p1, dtype=np.float64)
    order = np.lexsort((-p, s))
    so, co, po = s[order], c[order], p[order]
    first = np.r_[True, so[1:] != so[:-1]]
    gstart = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    pos = np.arange(len(order)) - gstart[gid]
    gsize = np.diff(np.r_[gstart, len(order)])
    has2 = gsize >= 2
    t1c, t1p = co[gstart], po[gstart]
    t2i = np.where(has2, gstart + 1, gstart)
    t2c, t2p = co[t2i], np.where(has2, po[t2i], -1.0)
    is_top = pos == 0
    ref_c = np.where(is_top, t2c[gid], t1c[gid])
    ref_p = np.where(is_top, t2p[gid], t1p[gid])
    has_ref = ~(is_top & ~has2[gid])
    core, addr = pool.name_core.to_numpy(), pool.addr.to_numpy()
    parts = par_map(_sim2, [(core[co[lo:lo + chunk]], addr[co[lo:lo + chunk]], core[ref_c[lo:lo + chunk]],
                             addr[ref_c[lo:lo + chunk]]) for lo in range(0, len(order), chunk)], n_jobs=n_jobs)
    sims = np.concatenate(parts)
    sims[~has_ref] = -1
    res = np.empty((len(order), 3), dtype=np.float32)
    res[order] = np.column_stack([sims, np.where(has_ref, ref_p, -1.0)])
    return pd.DataFrame(res, columns=["ref_sim_nm", "ref_sim_ad", "ref_p"], index=pairs.index)


CTX_COLS = ["p1","p1_over_smax", "s_max", "s_sum", "s_n50", "s_rank", "c_max", "c_gap", "c_rank", "c_sum_other", "c_n"]


def context_features(pairs: pd.DataFrame, p1) -> pd.DataFrame:
    """Stage-2 features from stage-1 probabilities: how does this candidate compare with the other candidates
    of the same S1 entity (a confident best match means singletons are unlikely), and with the other S1 entities
    that also want this S2/S3 record."""
    d = pd.DataFrame({"s": pairs.s1_idx.to_numpy(), "c": pairs.c_idx.to_numpy(), "p": np.asarray(p1)})
    gs, gc = d.groupby("s").p, d.groupby("c").p
    o = pd.DataFrame(index=pairs.index)
    o["p1"] = d.p.to_numpy()
    o["s_max"] = gs.transform("max").to_numpy()
    o["p1_over_smax"] = o.p1 / o.s_max.clip(lower=1e-6)
    o["s_sum"] = gs.transform("sum").to_numpy()
    o["s_n50"] = (d.p > 0.5).groupby(d.s).transform("sum").to_numpy()
    o["s_rank"] = gs.rank(ascending=False, method="first").to_numpy()
    o["c_max"] = gc.transform("max").to_numpy()
    o["c_gap"] = o.p1 - o.c_max
    o["c_rank"] = gc.rank(ascending=False, method="min").to_numpy()
    o["c_sum_other"] = gc.transform("sum").to_numpy() - o.p1
    o["c_n"] = gc.transform("size").to_numpy()
    return o[CTX_COLS].astype(np.float32)
