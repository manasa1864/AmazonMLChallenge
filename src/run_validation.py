"""Validate the pipeline on a coherent down-sampled slice of the TRAINING data, and (optionally) save the models.

    python src/run_validation.py --frac 0.05 [--save_model models/m]

Steps: sample world -> normalise -> blocking (top-K TF-IDF over name/skeleton/address tokens) -> pair features
-> stage-1 LightGBM (2-fold out-of-fold by S1 entity) -> stage-2 LightGBM on context features of the stage-1
probabilities (2-fold OOF) -> one-S1-per-record assignment -> threshold sweep on the official macro F0.5.
With --save_model both stages' fold models are saved; at test time their predictions are averaged.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from data import load_train_sample
from text import normalise_frame
from blocking import block
from features import pair_features, build_vocab, context_features, add_ambiguity
from metric import f05_macro
from decision import assign

LGB_PARAMS = dict(n_estimators=350, learning_rate=0.07, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, n_jobs=-1, verbose=-1)

ap = argparse.ArgumentParser()
ap.add_argument("--frac", type=float, default=0.03)
ap.add_argument("--K", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_df_abs", type=int, default=12000)
ap.add_argument("--channels", action="store_true", help="union with name/address/char-3gram blocking channels (worse: see blocking.py)")
ap.add_argument("--save_model", default=None, help="prefix for saving fold models + metadata")
args = ap.parse_args()
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

s1, pool, truth = load_train_sample(args.frac, args.seed)
tick(f"sample: {len(s1)} S1, {len(pool)} pool records (S2={pool.entity_id.str.startswith('S2').sum()}, "
     f"S3={pool.entity_id.str.startswith('S3').sum()}); singletons={np.mean([not v for v in truth.values()]):.3f}")
normalise_frame(s1)
normalise_frame(pool)
add_ambiguity(s1)
add_ambiguity(pool)
tick("normalised")

# ---- blocking ---------------------------------------------------------------------------
pairs = block(s1, pool, K=args.K, max_df_abs=args.max_df_abs, channels=args.channels)
s1_ids, c_ids = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()
pairs["y"] = [int(c_ids[c] in truth[s1_ids[s]]) for s, c in zip(pairs.s1_idx, pairs.c_idx)]
n_true = sum(len(v) for v in truth.values())
tick(f"blocking: {len(pairs)} candidate pairs ({len(pairs) / len(s1):.1f}/S1); "
     f"pair recall = {pairs.y.sum() / n_true:.4f}")
cand = {}
for s, c in zip(pairs.s1_idx, pairs.c_idx):
    cand.setdefault(s1_ids[s], set()).add(c_ids[c])
oracle = {k: (v & cand.get(k, set())) for k, v in truth.items()}
print(f"          oracle F0.5 ceiling given candidates = {f05_macro(truth, oracle):.4f}")

# ---- features ---------------------------------------------------------------------------
vocab = build_vocab(s1, pool, pairs)
F = pair_features(s1, pool, pairs, vocab)
tick(f"stage-1 features: {F.shape[1]} columns")


def decide(prob, thr):
    return {k: set(v) for k, v in assign(pairs, prob, thr, s1_ids, c_ids).items()}


def sweep(name, prob, grid):
    best = (-1, None)
    for t in grid:
        f = f05_macro(truth, decide(prob, t))
        if f > best[0]:
            best = (f, t)
    print(f"  {name:<34} best F0.5 = {best[0]:.4f} @ thr {best[1]:.2f}", flush=True)
    return best


def oof_lgb(X, y, fold, tag):
    """2-fold out-of-fold predictions; returns (oof, [models], summed importances)."""
    oof, models, imp = np.zeros(len(y)), [], 0
    for k in (0, 1):
        tr, te = fold != k, fold == k
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        models.append(m)
        imp = imp + m.feature_importances_
    tick(f"{tag}: OOF done")
    return oof, models, imp


print("\nBaseline v0 (TF-IDF cosine only):")
sweep("cosine", F.blk_score.to_numpy(), np.arange(0.2, 0.95, 0.05))

fold = pairs.s1_idx.to_numpy() % 2
y = pairs.y.to_numpy()
grid = np.arange(0.2, 0.97, 0.05)

print("\nStage 1 (LightGBM OOF):")
p1, models1, imp1 = oof_lgb(F.to_numpy(), y, fold, "stage-1")
sweep("stage-1", p1, grid)

C = context_features(pairs, p1)
F2 = pd.concat([F, C], axis=1)
print("\nStage 2 (LightGBM on stage-1 context features, OOF):")
p2, models2, imp2 = oof_lgb(F2.to_numpy(), y, fold, "stage-2")
f_best, t_best = sweep("stage-2", p2, grid)
fine = np.arange(max(t_best - 0.05, 0.05), min(t_best + 0.051, 0.99), 0.01)
f_best, t_best = sweep("stage-2 (fine thr)", p2, fine)

if args.save_model:
    pre = args.save_model
    Path(pre).parent.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models1):
        m.booster_.save_model(f"{pre}_s1_fold{i}.txt")
    for i, m in enumerate(models2):
        m.booster_.save_model(f"{pre}_s2_fold{i}.txt")
    Path(pre + ".json").write_text(json.dumps(
        {"thr": float(t_best), "val_f05": float(f_best), "vocab": vocab, "frac": args.frac, "K": args.K,
         "max_df_abs": args.max_df_abs, "channels": args.channels, "s1_features": list(F.columns), "s2_features": list(F2.columns)}))
    tick(f"models saved with prefix {pre} (thr={t_best:.3f})")

pred = decide(p2, t_best)
sing = [k for k, v in truth.items() if not v]
print(f"\nAt thr {t_best:.2f}: singletons handled = {np.mean([not pred.get(k) for k in sing]):.3f}, "
      f"avg #pred/S1 = {np.mean([len(pred.get(k, ())) for k in truth]):.2f} (true {np.mean([len(v) for v in truth.values()]):.2f})")
tp = sum(len(pred.get(k, set()) & v) for k, v in truth.items())
npred = sum(len(v) for v in pred.values())
print(f"pair precision = {tp / max(npred, 1):.4f}  pair recall = {tp / n_true:.4f}")

print("\nTop stage-2 features:\n" + pd.Series(imp2, index=F2.columns).sort_values(ascending=False).head(15).to_string())

# ---- error analysis ---------------------------------------------------------------------
info = pool.set_index("entity_id")[["business_name", "business_address"]]
s1i = s1.set_index("entity_id")[["business_name", "business_address"]]
show = lambda i, d: f"{i} | {d.loc[i].business_name} | {d.loc[i].business_address}"
rng = np.random.default_rng(1)
fps = [(k, c) for k, v in pred.items() for c in v if c not in truth[k]]
print(f"\n=== {len(fps)} false-positive pairs; examples")
for k, c in [fps[i] for i in rng.choice(len(fps), min(8, len(fps)), replace=False)]:
    print(" S1 :", show(k, s1i), "\n FP :", show(c, info), "\n")
fns = [(k, c) for k, v in truth.items() for c in v if c not in pred.get(k, set())]
print(f"=== {len(fns)} missed true pairs; examples")
for k, c in [fns[i] for i in rng.choice(len(fns), min(8, len(fns)), replace=False)]:
    print(" S1 :", show(k, s1i), "\n FN :", show(c, info), ("" if c in cand.get(k, set()) else "   (NOT in candidates)"), "\n")
tick("done")
