"""Train and package the churn decision service.

The workflow deliberately separates three questions:
1. Which model discriminates churners from non-churners well enough?
2. Are its probabilities calibrated enough to support expected-cost decisions?
3. Given explicit false-negative/false-positive costs, what action rule follows?

Model choice uses cross-validated ROC AUC on the training split.  Logistic
regression is preferred when it is statistically/operationally comparable
with the best candidate, because the deployed service can then provide exact
linear log-odds reason codes.  Calibration is fitted on a held-aside
validation split; the final test split is used only for reporting.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split

from decision import (bootstrap_threshold_interval, classification_metrics,
                      empirical_threshold, theoretical_threshold,
                      threshold_curve)
from features import make_pipeline
from validate import (CATEGORICAL, NUMERIC, TARGET, allowed_values, load_raw,
                      validate)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
VERSION = "2.0.0"
SELECTION_TOLERANCE = 0.005

CANDIDATES = {
    "majority": DummyClassifier(strategy="most_frequent"),
    "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "hist_gbm": HistGradientBoostingClassifier(max_depth=3,
                                               learning_rate=0.08,
                                               max_iter=300, random_state=0),
}


def reliability_bins(y, p, n=10):
    edges = np.quantile(p, np.linspace(0, 1, n + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n - 1)
    xs, ys = [], []
    for b in range(n):
        m = idx == b
        if m.sum() >= 20:
            xs.append(float(p[m].mean()))
            ys.append(float(np.mean(y[m])))
    return xs, ys


def select_champion(cv_summary: dict, tolerance: float = SELECTION_TOLERANCE) -> tuple[str, str]:
    """Prefer logistic if it is within tolerance of the best non-baseline model.

    This is an explicit simplicity/interpretability rule, not a claim that a
    0.001 AUC difference establishes superiority.
    """
    non_baseline = {k: v for k, v in cv_summary.items() if k != "majority"}
    best_name = max(non_baseline, key=lambda k: non_baseline[k]["mean"])
    best_mean = non_baseline[best_name]["mean"]
    logistic_mean = non_baseline["logistic"]["mean"]
    if best_name == "logistic":
        runner = max((k for k in non_baseline if k != "logistic"),
                     key=lambda k: non_baseline[k]["mean"])
        gap = logistic_mean - non_baseline[runner]["mean"]
        if gap <= tolerance:
            reason = (f"logistic has the highest mean ROC-AUC, but the {gap:.4f} "
                      f"gap to {runner} is within the {tolerance:.3f} "
                      "comparability tolerance; no meaningful win is claimed, "
                      "and logistic is retained for exact linear reason codes")
        else:
            reason = f"logistic has the highest mean ROC-AUC by {gap:.4f}"
        return "logistic", reason
    if best_mean - logistic_mean <= tolerance:
        reason = (f"logistic is within {tolerance:.3f} ROC-AUC of {best_name} "
                  "and is retained for simpler, exact linear reason codes")
        return "logistic", reason
    return best_name, f"{best_name} exceeds logistic by more than {tolerance:.3f} ROC-AUC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn-cost", type=float, default=5.0)
    ap.add_argument("--fp-cost", type=float, default=1.0)
    ap.add_argument("--data", default=ROOT / "data" / "raw" /
                    "telco_churn.csv")
    ap.add_argument("--threshold-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    clean = validate(load_raw(args.data))
    df = clean.df
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=0)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0)

    # Model selection: report fold-level uncertainty, not a fake winner from
    # a one-thousandth AUC difference.
    cv_summary = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(make_pipeline(model), X_tr, y_tr,
                                 cv=5, scoring="roc_auc")
        cv_summary[name] = {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std(ddof=1)), 4),
            "folds": [round(float(s), 4) for s in scores],
        }
    best_name, selection_reason = select_champion(cv_summary)

    FIGURES.mkdir(parents=True, exist_ok=True)
    figm, axm = plt.subplots(figsize=(7.0, 4.0))
    names = ["majority", "logistic", "hist_gbm"]
    labels = ["Majority", "Logistic", "HistGradientBoosting"]
    for i, name in enumerate(names):
        folds = np.array(cv_summary[name]["folds"], dtype=float)
        axm.scatter(np.full_like(folds, i, dtype=float), folds, s=28, alpha=0.75)
        axm.errorbar(i, cv_summary[name]["mean"],
                     yerr=cv_summary[name]["std"], fmt="o", capsize=4,
                     color="black", zorder=4)
    axm.set_xticks(range(len(labels)), labels)
    axm.set(ylabel="5-fold ROC AUC",
            title="Model selection: fold variation matters more than a tiny mean gap")
    axm.grid(alpha=0.25, axis="y")
    figm.tight_layout(); figm.savefig(FIGURES / "model_selection.png", dpi=150)
    plt.close(figm)

    pipe = make_pipeline(CANDIDATES[best_name]).fit(X_tr, y_tr)

    # Calibration on validation only.
    raw_val = pipe.predict_proba(X_val)[:, 1]
    calibrator = CalibratedClassifierCV(FrozenEstimator(pipe),
                                        method="isotonic")
    calibrator.fit(X_val, y_val)
    cal_val = calibrator.predict_proba(X_val)[:, 1]

    # Final held-out test: compare raw and calibrated probabilities here so
    # the calibration claim is genuinely out-of-sample.
    raw_te = pipe.predict_proba(X_te)[:, 1]
    p_te = calibrator.predict_proba(X_te)[:, 1]
    test_brier_raw = float(brier_score_loss(y_te, raw_te))
    test_brier_cal = float(brier_score_loss(y_te, p_te))

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for p, lab, style in [(raw_te, "raw", "o-"),
                          (p_te, "isotonic", "s-")]:
        xs, ys = reliability_bins(y_te.to_numpy(), p)
        ax.plot(xs, ys, style, lw=1.6,
                label=f"{lab}, Brier {brier_score_loss(y_te, p):.3f}")
    ax.set(xlabel="Predicted churn probability",
           ylabel="Observed churn rate", title="Reliability (held-out test)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=150)
    plt.close(fig)

    # Decision rule.  For calibrated probabilities and constant costs, the
    # theoretical Bayes threshold is transparent and stable.  I still show
    # the empirical validation optimum as a diagnostic and bootstrap it to
    # make threshold uncertainty visible.
    theory_t = theoretical_threshold(args.fn_cost, args.fp_cost)
    empirical_t, empirical_cost = empirical_threshold(
        y_val.to_numpy(), cal_val, args.fn_cost, args.fp_cost)
    threshold_boot = bootstrap_threshold_interval(
        y_val.to_numpy(), cal_val, args.fn_cost, args.fp_cost,
        n_boot=args.threshold_bootstrap, seed=0)

    # Isotonic probabilities can have plateaus.  With this dataset, the
    # theoretical and empirical thresholds produce identical validation
    # actions even though their numeric values differ.
    theory_val_pred = cal_val >= theory_t
    empirical_val_pred = cal_val >= empirical_t
    same_actions_val = bool(np.array_equal(theory_val_pred, empirical_val_pred))

    # Deploy the cost-derived threshold, not the sample-specific point optimum.
    threshold = float(theory_t)
    ts, costs, recalls = threshold_curve(y_val.to_numpy(), cal_val,
                                         args.fn_cost, args.fp_cost)
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.6))
    ax2.plot(ts, costs, lw=1.8, color="#0b3d91",
             label="validation expected cost")
    ax2.axvline(theory_t, color="black", ls="--", lw=1.4,
                label=f"cost-derived {theory_t:.3f}")
    ax2.axvline(empirical_t, color="crimson", ls=":", lw=1.6,
                label=f"empirical point optimum {empirical_t:.2f}")
    ax2b = ax2.twinx()
    ax2b.plot(ts, recalls, lw=1.1, color="seagreen", alpha=0.7)
    ax2b.set_ylabel("Churner recall", color="seagreen")
    ax2b.tick_params(axis="y", labelcolor="seagreen")
    ax2.set(xlabel="Decision threshold", ylabel="Expected cost per customer",
            title=f"Decision rule under {args.fn_cost:.0f}:{args.fp_cost:.0f} costs")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(FIGURES / "threshold_cost.png", dpi=150)
    plt.close(fig2)

    # Bootstrap threshold stability visual.
    # Recompute small sample of choices for a compact reproducible histogram.
    rng = np.random.default_rng(0)
    boot_choices = []
    n = len(y_val)
    for _ in range(min(args.threshold_bootstrap, 500)):
        idx = rng.integers(0, n, n)
        t, _ = empirical_threshold(y_val.to_numpy()[idx], cal_val[idx],
                                   args.fn_cost, args.fp_cost)
        boot_choices.append(t)
    figb, axb = plt.subplots(figsize=(7.2, 3.7))
    axb.hist(boot_choices, bins=np.arange(0.015, 0.305, 0.01),
             color="#0b3d91", alpha=0.8)
    axb.axvline(theory_t, color="black", ls="--", lw=1.5,
                label=f"cost-derived {theory_t:.3f}")
    axb.axvline(empirical_t, color="crimson", ls=":", lw=1.6,
                label=f"point optimum {empirical_t:.2f}")
    axb.set(xlabel="Bootstrap empirical threshold",
            ylabel="Bootstrap samples",
            title="Threshold stability on the validation sample")
    axb.legend(fontsize=8); axb.grid(alpha=0.2, axis="y")
    figb.tight_layout(); figb.savefig(FIGURES / "threshold_stability.png", dpi=150)
    plt.close(figb)

    # Held-out test metrics at the deployed (cost-derived) threshold.
    pred_te = (p_te >= threshold).astype(int)
    decision_metrics = classification_metrics(
        y_te.to_numpy(), p_te, threshold, args.fn_cost, args.fp_cost)
    test_report = {
        "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        "brier_raw": round(test_brier_raw, 4),
        "brier_calibrated": round(test_brier_cal, 4),
        "churner_recall": round(float(recall_score(y_te, pred_te)), 3),
        "churner_precision": round(float(precision_score(y_te, pred_te)), 3),
        "flag_rate": round(float(pred_te.mean()), 3),
        "expected_cost_per_customer": round(decision_metrics["cost_per_customer"], 4),
    }

    # Segment audit: report sample sizes and positives so rare-segment claims
    # cannot look more certain than they are.
    seg_rows = {}
    for seg, g in X_te.assign(y=y_te.to_numpy(), pred=pred_te).groupby("Contract"):
        y_seg = g.y.to_numpy()
        pred_seg = g.pred.to_numpy()
        seg_rows[seg] = {
            "recall": round(float(recall_score(y_seg, pred_seg,
                                               zero_division=0)), 3),
            "churn_rate": round(float(y_seg.mean()), 4),
            "n": int(len(g)),
            "churners": int(y_seg.sum()),
        }
    fig3, ax3 = plt.subplots(figsize=(7.2, 2.9))
    segs = list(sorted(seg_rows))
    vals = [seg_rows[s]["recall"] for s in segs]
    ax3.hlines(range(len(segs)), 0, vals, lw=2, color="#0b3d91")
    ax3.plot(vals, range(len(segs)), "o", color="#0b3d91")
    for i, s in enumerate(segs):
        r = seg_rows[s]
        ax3.annotate(f"{r['recall']:.2f} (n={r['n']}, churners={r['churners']})",
                     (r["recall"], i), fontsize=8, xytext=(6, -3),
                     textcoords="offset points")
    ax3.set_yticks(range(len(segs)), segs, fontsize=9)
    ax3.set(xlim=(0, 1.05), xlabel="Churner recall",
            title="Segment audit: recall and sample size (held-out test)")
    ax3.grid(alpha=0.3, axis="x")
    fig3.tight_layout(); fig3.savefig(FIGURES / "segments.png", dpi=150)
    plt.close(fig3)

    artifact = {
        "version": VERSION,
        "model_name": best_name,
        "pipeline": calibrator,
        "threshold": threshold,
        "threshold_policy": "cost-derived Bayes threshold on calibrated probability",
        "costs": {"fn": args.fn_cost, "fp": args.fp_cost},
        "allowed_values": allowed_values(X_tr),
        "reference": reference_stats(X_tr),
        "trained": str(date.today()),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    for old in (ROOT / "models").glob("churn-*.joblib"):
        old.unlink()
    path = ROOT / "models" / f"churn-{VERSION}.joblib"
    joblib.dump(artifact, path)

    summary = {
        "cleaning_log": clean.log,
        "split_sizes": {"train": len(X_tr), "calibration_threshold": len(X_val),
                        "test": len(X_te)},
        "cv_roc_auc": cv_summary,
        "selected_model": best_name,
        "selection_reason": selection_reason,
        "calibration": {
            "validation_brier_raw": round(float(brier_score_loss(y_val, raw_val)), 4),
            "validation_brier_calibrated": round(float(brier_score_loss(y_val, cal_val)), 4),
            "heldout_test_brier_raw": round(test_brier_raw, 4),
            "heldout_test_brier_calibrated": round(test_brier_cal, 4),
        },
        "decision": {
            "fn_cost": args.fn_cost, "fp_cost": args.fp_cost,
            "theoretical_threshold": round(theory_t, 4),
            "deployed_threshold": round(threshold, 4),
            "empirical_validation_threshold": round(empirical_t, 3),
            "empirical_validation_cost": round(empirical_cost, 4),
            "bootstrap_empirical_threshold_95pct": threshold_boot,
            "theoretical_and_empirical_same_validation_actions": same_actions_val,
        },
        "test": test_report,
        "segment_audit": seg_rows,
        "artifact": path.name,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "training_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    card = f"""# Model card: churn-{VERSION}

**Purpose.** Estimate churn probability and support retention decisions under an
explicit false-negative:false-positive cost ratio ({args.fn_cost:.0f}:{args.fp_cost:.0f}).

**Data.** IBM Telco customer-churn public sample, {len(df)} rows after validation,
churn rate {df[TARGET].mean():.1%}. Cleaning interventions: {"; ".join(clean.log)}.

**Model selection.** Logistic regression and histogram gradient boosting have
similar cross-validated discrimination on the training split. The deployed
champion is **{best_name}** (mean ROC AUC {cv_summary[best_name]['mean']:.3f} ±
{cv_summary[best_name]['std']:.3f}); selection rule: {selection_reason}.

**Calibration.** Isotonic calibration is fitted only on the validation split.
On the untouched test split, Brier score improves from **{test_brier_raw:.3f} to
{test_brier_cal:.3f}**.

**Decision rule.** With calibrated probabilities and costs {args.fn_cost:.0f}:1,
the transparent Bayes threshold is **{theory_t:.3f}**. The validation sample's
point-optimal threshold is {empirical_t:.2f}, but its bootstrap 95% range is
{threshold_boot['p2_5']:.2f}–{threshold_boot['p97_5']:.2f}; the point optimum is
therefore treated as a diagnostic rather than a universal business constant.

**Held-out test.** ROC AUC {test_report['roc_auc']:.3f}; churner recall
{test_report['churner_recall']:.2f} at precision {test_report['churner_precision']:.2f};
flag rate {test_report['flag_rate']:.0%}; expected cost per customer
{test_report['expected_cost_per_customer']:.3f} under the stated ratio.

**Explanations.** Reason codes are exact additive contributions to the underlying
logistic **log-odds**. They do not claim to add up to the calibrated probability;
isotonic calibration is a monotone remapping applied after the logistic score.

**Known limitations.** This is a static public sample, not a longitudinal
production cohort. Segment estimates can be noisy (the held-out two-year group
contains only {seg_rows['Two year']['churners']} churners). Cost ratios are
illustrative and assumed constant per customer. Drift and retraining modules are
offline governance demonstrations rather than claims of a live production loop.
"""
    (ROOT / "MODEL_CARD.md").write_text(card)
    print(f"Artifact -> {path}")


def reference_stats(train_df):
    """Reference distributions carried with the artifact for PSI monitoring."""
    ref = {"numeric": {}, "categorical": {}}
    for col in NUMERIC:
        edges = np.quantile(train_df[col], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        unique_edges = np.unique(edges)
        counts, _ = np.histogram(train_df[col], bins=unique_edges)
        ref["numeric"][col] = {"edges": unique_edges.tolist(),
                               "freqs": (counts / counts.sum()).tolist()}
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        vc = train_df[col].value_counts(normalize=True)
        ref["categorical"][col] = vc.to_dict()
    return ref


if __name__ == "__main__":
    main()
