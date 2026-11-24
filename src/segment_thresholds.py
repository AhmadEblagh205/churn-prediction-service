"""Audit whether segment-specific decision thresholds are worth deploying.

A lower point estimate of cost is not enough to justify operational
complexity.  Thresholds are fitted on the validation split, evaluated on
the held-out test split, and the *difference* in test cost is bootstrapped.
If the 95% interval includes zero, the recommendation is to retain the
single global policy and treat the segment result as exploratory.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from decision import empirical_threshold, expected_cost
from validate import CATEGORICAL, NUMERIC, TARGET, load_raw, validate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "churn-2.0.0.joblib"


def fit_segment_thresholds(p_val, y_val, seg_val,
                           fn_cost: float, fp_cost: float) -> dict:
    out = {}
    seg_arr = np.asarray(seg_val)
    y_arr = np.asarray(y_val, dtype=int)
    p_arr = np.asarray(p_val, dtype=float)
    for seg in sorted(np.unique(seg_arr)):
        m = seg_arr == seg
        t, _ = empirical_threshold(y_arr[m], p_arr[m], fn_cost, fp_cost)
        out[str(seg)] = float(t)
    return out


def bootstrap_cost_difference(y, pred_global, pred_segmented,
                              fn_cost: float, fp_cost: float,
                              n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap (global cost - segmented cost); positive favours segments."""
    y = np.asarray(y, dtype=int)
    pg = np.asarray(pred_global, dtype=int)
    ps = np.asarray(pred_segmented, dtype=int)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = (expected_cost(y[idx], pg[idx], fn_cost, fp_cost)
                   - expected_cost(y[idx], ps[idx], fn_cost, fp_cost))
    lo, med, hi = np.quantile(vals, [0.025, 0.5, 0.975])
    return {
        "n_boot": int(n_boot),
        "p2_5": round(float(lo), 4),
        "median": round(float(med), 4),
        "p97_5": round(float(hi), 4),
        "probability_segmented_lower_cost": round(float((vals > 0).mean()), 3),
    }


def main():
    artifact = joblib.load(DEFAULT_MODEL)
    fn_cost = float(artifact["costs"]["fn"])
    fp_cost = float(artifact["costs"]["fp"])
    df = validate(load_raw(ROOT / "data" / "raw" / "telco_churn.csv")).df
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=0)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0)

    p_val = artifact["pipeline"].predict_proba(X_val)[:, 1]
    p_te = artifact["pipeline"].predict_proba(X_te)[:, 1]
    global_t = float(artifact["threshold"])
    seg_ts = fit_segment_thresholds(p_val, y_val.to_numpy(),
                                    X_val.Contract.to_numpy(),
                                    fn_cost, fp_cost)

    seg_te = X_te.Contract.to_numpy()
    pred_global = (p_te >= global_t).astype(int)
    pred_seg = np.array([p >= seg_ts[s]
                         for p, s in zip(p_te, seg_te)]).astype(int)

    rows = {}
    yte = y_te.to_numpy()
    for seg in sorted(seg_ts):
        m = seg_te == seg
        rows[seg] = {
            "threshold": round(seg_ts[seg], 3),
            "n_test": int(m.sum()),
            "churners_test": int(yte[m].sum()),
            "recall_global": round(float(recall_score(
                yte[m], pred_global[m], zero_division=0)), 3),
            "recall_segmented": round(float(recall_score(
                yte[m], pred_seg[m], zero_division=0)), 3),
        }

    cost_global = expected_cost(yte, pred_global, fn_cost, fp_cost)
    cost_segmented = expected_cost(yte, pred_seg, fn_cost, fp_cost)
    boot = bootstrap_cost_difference(yte, pred_global, pred_seg,
                                     fn_cost, fp_cost)
    robust_improvement = boot["p2_5"] > 0
    recommendation = ("deploy_segment_thresholds" if robust_improvement
                      else "retain_global_threshold")
    summary = {
        "global_threshold": round(global_t, 4),
        "segment_thresholds": rows,
        "expected_cost_global": round(cost_global, 4),
        "expected_cost_segmented": round(cost_segmented, 4),
        "observed_cost_saving_per_customer": round(cost_global - cost_segmented, 4),
        "bootstrap_cost_saving_95pct": boot,
        "recommendation": recommendation,
        "interpretation": (
            "The segmented policy has a slightly lower point estimate of cost, "
            "but the bootstrap interval includes zero, so the evidence is not "
            "strong enough to justify segment-specific deployment complexity."
            if not robust_improvement else
            "The segmented policy lowers cost with a bootstrap interval above zero."
        ),
    }

    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    segs = list(rows)
    for i, seg in enumerate(segs):
        r = rows[seg]
        ax.hlines(i, r["recall_global"], r["recall_segmented"], lw=2,
                  color="#0b3d91")
        ax.plot(r["recall_global"], i, "s", color="#c1440e",
                label="global threshold" if i == 0 else None)
        ax.plot(r["recall_segmented"], i, "o", color="#0b3d91",
                label="segment threshold" if i == 0 else None)
        ax.annotate(f"n={r['n_test']}, churners={r['churners_test']}",
                    (0.02, i), fontsize=8, va="bottom")
    ax.set_yticks(range(len(segs)), segs)
    ax.set(xlim=(0, 1.05), xlabel="churner recall (held-out test)",
           title="Segment thresholds: exploratory, with sample sizes shown")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "segment_thresholds.png",
                dpi=150)
    plt.close(fig)

    (ROOT / "reports" / "segment_thresholds.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
