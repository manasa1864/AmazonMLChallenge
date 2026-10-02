"""Candidate generation: per-country TF-IDF retrieval (top-30 cosine within the same country label).

SUBMITTED CONFIGURATION: the 'main' channel only (channels=False). The optional union with three extra channels
(channels=True) raised pair recall 97.4% -> 98.1% but lowered validation F0.5 0.9785 -> 0.9761 (more decoys among
the 44.5 candidates/S1 hurt singletons and precision), so it is off by default.

Channels (all cosine top-K within the same country label):
  main  name words + name skeleton + address words + address bigrams   (K = 30)
  name  name words only                                                 (K = 10)
  addr  address words + address bigrams only                            (K = 10)
  chr   character 3-grams of the space-free name (typo robust)          (K = 10)
The union is what the matcher scores (= candidate_pairs.tsv). Every pair carries its blocking score (cosine of the
main channel, computed exactly for pairs found only by another channel) and the rank it had in each channel.
Query chunks are scored in parallel.
"""
import gc
import time
import numpy as np
import pandas as pd
from par import par_map
from scipy.sparse import diags
from sklearn.feature_extraction.text import TfidfVectorizer

# Phonetic-skeleton tokens in the main channel: 'all' | 'nonlatin' | 'none' (blocking experiments: 'all' was best).
K_MODE = "all"
RANK_COLS = ["r_main", "r_name", "r_addr", "r_chr"]   # rank+1 within the channel, 0 = not retrieved by that channel


def make_doc(df: pd.DataFrame) -> list:
    """Main channel. Prefixes keep vocabularies apart: n_ name word, k_ name skeleton, a_ address word,
    b_ adjacent address word pair (captures house numbers like '88 5', 'bangalore north')."""
    out = []
    for n, k, a, nl, eid in zip(df["name_core"].tolist(), df["name_skel"].tolist(), df["addr"].tolist(),
                                df["nonlatin"].tolist(), df["entity_id"].tolist()):
        at = a.split()
        t = ["n_" + x for x in n.split()] + ["a_" + x for x in at]
        t += ["b_" + at[i] + "_" + at[i + 1] for i in range(len(at) - 1)]
        if K_MODE == "all" or (K_MODE == "nonlatin" and (nl or eid.startswith("S1"))):
            t += ["k_" + x for x in k.split() if len(x) >= 3]
        out.append(" ".join(t))
    return out


def doc_name(df):
    return [" ".join("n_" + t for t in n.split()) for n in df.name_core.tolist()]


def doc_addr(df):
    o = []
    for a in df.addr.tolist():
        t = a.split()
        o.append(" ".join(["a_" + x for x in t] + ["b_" + t[i] + "_" + t[i + 1] for i in range(len(t) - 1)]))
    return o


def doc_chr(df):
    o = []
    for n in df.name_core.tolist():
        s = n.replace(" ", "")
        o.append(" ".join("c_" + s[i:i + 3] for i in range(max(len(s) - 2, 1))))
    return o


# name, doc function, K, max_df_abs
EXTRA_CHANNELS = [("r_name", doc_name, 10, 12000), ("r_addr", doc_addr, 10, 12000), ("r_chr", doc_chr, 10, 6000)]


def _topk_chunk(Qc, PT, K):
    """Top-K pool columns per query row for one chunk. Returns (row, col, score, rank) arrays."""
    M = (Qc @ PT).tocsr()
    ip, ix, dt = M.indptr, M.indices, M.data
    R, C, S, N = [], [], [], []
    for r in range(M.shape[0]):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        cols, val = ix[a:b], dt[a:b]
        if len(val) > K:
            sel = np.argpartition(-val, K)[:K]
            cols, val = cols[sel], val[sel]
        o = np.argsort(-val)
        n = len(o)
        R.append(np.full(n, r)); C.append(cols[o]); S.append(val[o]); N.append(np.arange(n))
    if not R:
        e = np.zeros(0)
        return e.astype(int), e.astype(int), e.astype(np.float32), e.astype(int)
    return np.concatenate(R), np.concatenate(C), np.concatenate(S), np.concatenate(N)


def _retrieve(q_docs, p_docs, K, cap, max_df, chunk, n_jobs, weights=None):
    """TF-IDF fit on the pool docs, top-K per query. Returns (rows, cols, scores, ranks). Builds and releases its
    own TF-IDF matrices (never returns them) so peak memory stays at one channel's matrices at a time - important
    at full scale, where several channels are retrieved back to back for the same country."""
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
                          max_df=max(50, min(int(max_df * len(p_docs)), cap)), dtype=np.float32)
    P = vec.fit_transform(p_docs)
    Q = vec.transform(q_docs)
    if weights:   # per-prefix weights on the query side, e.g. {"n_": 0.7}
        w = np.ones(len(vec.vocabulary_), dtype=np.float32)
        for t, ix in vec.vocabulary_.items():
            w[ix] = weights.get(t[:2], 1.0)
        Q = Q @ diags(w)
    PT = P.T.tocsr()
    del P
    starts = list(range(0, Q.shape[0], chunk))
    res = par_map(_topk_chunk, [(Q[s:s + chunk], PT, K) for s in starts], n_jobs=n_jobs)
    r = np.concatenate([s + x[0] for s, x in zip(starts, res)])
    c = np.concatenate([x[1] for x in res])
    v = np.concatenate([x[2] for x in res])
    n = np.concatenate([x[3] for x in res])
    del PT, Q, res, vec
    gc.collect()
    return r, c, v, n


def block(s1: pd.DataFrame, pool: pd.DataFrame, K: int = 30, max_df: float = 0.005, max_df_abs: int = 12000,
          chunk: int = 4000, doc_fn=None, weights=None, n_jobs: int = 20, channels: bool = False) -> pd.DataFrame:
    """Returns pairs [s1_idx, c_idx, blk_score, blk_rank, r_main, r_name, r_addr, r_chr] (positional indices into
    s1 / pool). Records are only compared within the same country label (open set: works for unseen countries).
    Tokens appearing in more than min(max_df * pool_size, cap) pool records are ignored, which keeps the sparse
    product tractable at full scale. Passing doc_fn selects the legacy single-channel mode (used by experiments)."""
    t0 = time.time()
    single = doc_fn is not None or not channels
    doc_fn = doc_fn or make_doc
    out = []
    for country, q_idx in s1.groupby("country").indices.items():
        p_idx = np.flatnonzero((pool.country == country).to_numpy())
        if len(p_idx) == 0:
            continue
        qdf, pdf = s1.iloc[q_idx], pool.iloc[p_idx]
        if single:      # legacy / submitted configuration: no need to keep the big TF-IDF matrices afterwards
            r, c, v, n = _retrieve(doc_fn(qdf), doc_fn(pdf), K, max_df_abs, max_df, chunk, n_jobs, weights)
            print(f"  {country}: main channel {len(q_idx)} queries x {len(p_idx)} pool [{time.time() - t0:.0f}s]", flush=True)
            out.append(pd.DataFrame({"s1_idx": q_idx[r], "c_idx": p_idx[c], "blk_score": v, "blk_rank": n}))
            gc.collect()
            continue
        r, c, v, n = _retrieve(doc_fn(qdf), doc_fn(pdf), K, max_df_abs, max_df, chunk, n_jobs, weights)
        print(f"  {country}: main channel {len(q_idx)} queries x {len(p_idx)} pool [{time.time() - t0:.0f}s]", flush=True)
        npool = np.int64(len(p_idx))
        keys = [r.astype(np.int64) * npool + c]           # union key = local query * pool size + local pool row
        ranks = [(n + 1).astype(np.int16)]
        scores = [v]
        for name, fn, k_ch, cap in EXTRA_CHANNELS:
            r2, c2, v2, n2 = _retrieve(fn(qdf), fn(pdf), k_ch, cap, max_df, chunk, n_jobs)
            keys.append(r2.astype(np.int64) * npool + c2)
            ranks.append((n2 + 1).astype(np.int16))
            scores.append(v2)
            print(f"  {country}: {name[2:]} channel done [{time.time() - t0:.0f}s]", flush=True)
        ukeys = np.unique(np.concatenate(keys))
        rk = {}
        for name, k_arr, r_arr in zip(["r_main"] + [e[0] for e in EXTRA_CHANNELS], keys, ranks):
            col = np.zeros(len(ukeys), dtype=np.int16)
            col[np.searchsorted(ukeys, k_arr)] = r_arr
            rk[name] = col
        us, uc = ukeys // npool, ukeys % npool
        # blk_score: the main channel's cosine where available; 0 (no main-channel signal) for pairs found only by
        # another channel - cheap and safe (no extra retrieval / memory), the model has rank + text-similarity
        # features to tell these apart anyway.
        score = np.zeros(len(ukeys), dtype=np.float32)
        score[np.searchsorted(ukeys, keys[0])] = scores[0]
        df = pd.DataFrame({"s1_idx": q_idx[us], "c_idx": p_idx[uc], "blk_score": score, **rk})
        df["blk_rank"] = df.groupby("s1_idx").blk_score.rank(ascending=False, method="first").to_numpy() - 1
        out.append(df)
        print(f"  {country}: union {len(df)} pairs ({len(df) / len(q_idx):.1f}/S1) [{time.time() - t0:.0f}s]", flush=True)
        del keys, ranks, scores, rk
        gc.collect()
    return pd.concat(out, ignore_index=True)
