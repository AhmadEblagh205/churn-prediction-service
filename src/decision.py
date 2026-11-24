"""Decision utilities shared by training, evaluation and the dashboard.

The classifier estimates churn probability.  The business action is a
separate decision problem: a false negative (missed churner) and a false
positive (unnecessary retention offer) have explicit costs.  Keeping the
probability model and the action rule separate makes the assumptions
visible and lets the threshold change without retraining the model.
"""
from __future__ import annotations

import numpy as np


def theoretical_threshold(fn_cost: float, fp_cost: float) -> float:
    """Bayes threshold for a calibrated probability under constant costs.

    Act when the expected cost of *not* acting, p * C_FN, exceeds the
    expected cost of acting, (1-p) * C_FP.  The break-even probability is
    therefore C_FP / (C_FN + C_FP).
    """
    if fn_cost <= 0 or fp_cost < 0:
        raise ValueError("fn_cost must be > 0 and fp_cost must be >= 0")
    return float(fp_cost / (fn_cost + fp_cost))


def expected_cost(y, pred, fn_cost: float, fp_cost: float) -> float:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return float((fn * fn_cost + fp * fp_cost) / max(len(y), 1))


def threshold_curve(y, p, fn_cost: float, fp_cost: float,
                    thresholds: np.ndarray | None = None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ts = np.asarray(thresholds if thresholds is not None
                    else np.linspace(0.02, 0.98, 193), dtype=float)
    pred = p[:, None] >= ts[None, :]
    fn = ((~pred) & (y[:, None] == 1)).sum(axis=0)
    fp = (pred & (y[:, None] == 0)).sum(axis=0)
    tp = (pred & (y[:, None] == 1)).sum(axis=0)
    costs = (fn * fn_cost + fp * fp_cost) / max(len(y), 1)
    recalls = tp / np.maximum(tp + fn, 1)
    return ts, costs.astype(float), recalls.astype(float)


def empirical_threshold(y, p, fn_cost: float, fp_cost: float,
                        thresholds: np.ndarray | None = None) -> tuple[float, float]:
    ts, costs, _ = threshold_curve(y, p, fn_cost, fp_cost, thresholds)
    i = int(np.argmin(costs))
    return float(ts[i]), float(costs[i])


def bootstrap_threshold_interval(y, p, fn_cost: float, fp_cost: float,
                                 n_boot: int = 1000,
                                 seed: int = 0) -> dict:
    """Non-parametric bootstrap of the empirical cost-minimising threshold.

    This quantifies how unstable a point-optimal threshold can be on a
    finite validation sample.  It is a diagnostic, not a confidence
    interval for the true business cost ratio.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.02, 0.98, 193)
    chosen = np.empty(n_boot, dtype=float)
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        chosen[b], _ = empirical_threshold(
            y[idx], p[idx], fn_cost, fp_cost, thresholds=ts)
    q = np.quantile(chosen, [0.025, 0.5, 0.975])
    return {
        "n_boot": int(n_boot),
        "p2_5": round(float(q[0]), 3),
        "median": round(float(q[1]), 3),
        "p97_5": round(float(q[2]), 3),
    }


def classification_metrics(y, p, threshold: float,
                           fn_cost: float, fp_cost: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "flag_rate": float(pred.mean()),
        "cost_per_customer": expected_cost(y, pred, fn_cost, fp_cost),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "offers": tp + fp, "caught": tp, "missed": fn,
        "spared": tn, "n": int(len(y)),
    }
