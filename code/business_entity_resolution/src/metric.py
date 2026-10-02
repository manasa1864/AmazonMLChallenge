"""Official metric: per-S1-entity F0.5, macro-averaged (singletons: 1.0 if empty pred, else 0.0)."""


def f05_one(truth: set, pred: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(truth & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def f05_macro(truth: dict, pred: dict) -> float:
    """truth/pred: {s1_id: set(ids)}; every s1 in truth is scored (missing pred = empty)."""
    return sum(f05_one(t, pred.get(k, set())) for k, t in truth.items()) / max(len(truth), 1)
