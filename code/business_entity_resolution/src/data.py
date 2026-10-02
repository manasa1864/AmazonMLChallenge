"""Data loading + a coherent down-sampled 'mini world' of the training set for fast iteration."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"       # expects dataset/train/*.tsv, dataset/test/*.tsv (see README.md)
CACHE = ROOT / "cache"


def read_tsv(path) -> pd.DataFrame:
    # quoting=3 (QUOTE_NONE): names/addresses may contain stray quotes; tabs are the only delimiter.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def parse_gt(gt: pd.DataFrame) -> dict:
    return {a: {i for i in b.split(",") if i} for a, b in zip(gt.source1_entity_id, gt.matched_entity_ids)}


def load_train_sample(frac: float, seed: int = 0):
    """Sample `frac` of S1 entities and keep the world coherent:
    pool = records of the sampled entities + `frac` of the orphan S2/S3 records (matched to no S1).
    Returns (s1, pool, truth) where pool has S2 and S3 records, truth={s1_id:set(matched ids)}."""
    CACHE.mkdir(exist_ok=True)
    cp = CACHE / f"train_f{frac}_s{seed}.pkl"
    if cp.exists():
        return pd.read_pickle(cp)
    rng = np.random.default_rng(seed)
    s1 = read_tsv(DATA / "train/train_source1.tsv")
    gt = parse_gt(read_tsv(DATA / "train/train_ground_truth.tsv"))
    keep = s1.entity_id[rng.random(len(s1)) < frac]
    s1 = s1[s1.entity_id.isin(set(keep))].reset_index(drop=True)
    truth = {k: gt[k] for k in s1.entity_id}
    matched_sample = set().union(*truth.values())
    matched_all = set().union(*gt.values())
    parts = []
    for i in (2, 3):
        d = read_tsv(DATA / f"train/train_source{i}.tsv")
        in_sample = d.entity_id.isin(matched_sample)
        orphan = ~d.entity_id.isin(matched_all) & (rng.random(len(d)) < frac)
        parts.append(d[in_sample | orphan])
    pool = pd.concat(parts, ignore_index=True)
    out = (s1, pool, truth)
    pd.to_pickle(out, cp)
    return out
