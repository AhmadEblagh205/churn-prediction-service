"""Offline retraining-governance demonstration.

Two guardrails are shown separately:
1. a PSI rule decides when a shifted batch is worth investigating;
2. a challenger is promoted only on a *disjoint* promotion holdout and only
   when decision cost improves without material regression in discrimination
   or calibration.

The IBM Telco sample is a static snapshot, so this module does **not** claim a
live or temporal retraining loop.  It simulates historical/future partitions
with fixed random splits solely to demonstrate defensible promotion logic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from decision import classification_metrics, theoretical_threshold
from features import make_pipeline
from monitor import ALERT, drift_report
from validate import CATEGORICAL, NUMERIC, TARGET, load_raw, validate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "churn-2.0.0.joblib"
KEY_FEATURES = {"tenure", "MonthlyCharges", "Contract"}


def should_retrain(report: dict, alert: float = ALERT,
                   min_alerts: int = 2) -> tuple[bool, list[str]]:
    """Investigate when multiple features drift or a key feature breaches."""
    breaches = [k for k, v in report.items() if v > alert]
    fire = len(breaches) >= min_alerts or bool(set(breaches) & KEY_FEATURES)
    return fire, breaches


def train_calibrated(df: pd.DataFrame, seed: int = 1):
    """Fit logistic model and isotonic calibrator on disjoint inner splits."""
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    pipe = make_pipeline(LogisticRegression(max_iter=2000,
                                            class_weight="balanced"))
    pipe.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipe),
                                        method="isotonic")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def evaluate_model(model, X_hold, y_hold, threshold: float,
                   fn_cost: float, fp_cost: float) -> dict:
    p = model.predict_proba(X_hold)[:, 1]
    dec = classification_metrics(y_hold, p, threshold, fn_cost, fp_cost)
    return {
        "roc_auc": float(roc_auc_score(y_hold, p)),
        "brier": float(brier_score_loss(y_hold, p)),
        "expected_cost": float(dec["cost_per_customer"]),
        "recall": float(dec["recall"]),
        "flag_rate": float(dec["flag_rate"]),
    }


def promotion_gate(champion, challenger, X_hold, y_hold,
                   fn_cost: float = 5.0, fp_cost: float = 1.0,
                   min_cost_improvement: float = 0.005,
                   auc_tolerance: float = 0.002,
                   brier_tolerance: float = 0.002) -> dict:
    """Decision-cost-first promotion gate with calibration/AUC guardrails."""
    threshold = theoretical_threshold(fn_cost, fp_cost)
    champ = evaluate_model(champion, X_hold, y_hold, threshold,
                           fn_cost, fp_cost)
    chall = evaluate_model(challenger, X_hold, y_hold, threshold,
                           fn_cost, fp_cost)
    cost_improvement = champ["expected_cost"] - chall["expected_cost"]
    auc_delta = chall["roc_auc"] - champ["roc_auc"]
    brier_delta = chall["brier"] - champ["brier"]  # negative is better
    promote = (cost_improvement >= min_cost_improvement
               and auc_delta >= -auc_tolerance
               and brier_delta <= brier_tolerance)
    return {
        "threshold": round(threshold, 4),
        "champion": {k: round(v, 4) for k, v in champ.items()},
        "challenger": {k: round(v, 4) for k, v in chall.items()},
        "deltas": {
            "cost_improvement": round(float(cost_improvement), 4),
            "auc": round(float(auc_delta), 4),
            "brier": round(float(brier_delta), 4),
        },
        "guardrails": {
            "min_cost_improvement": min_cost_improvement,
            "auc_tolerance": auc_tolerance,
            "brier_tolerance": brier_tolerance,
        },
        "promote": bool(promote),
    }


def governance_demo(df: pd.DataFrame) -> dict:
    """Simulate old history -> new adaptation batch -> promotion holdout."""
    history, future = train_test_split(
        df, test_size=0.30, stratify=df[TARGET], random_state=21)
    adaptation, promotion_holdout = train_test_split(
        future, test_size=0.50, stratify=future[TARGET], random_state=22)

    champion = train_calibrated(history, seed=23)
    challenger_train = pd.concat([history, adaptation], ignore_index=True)
    challenger = train_calibrated(challenger_train, seed=24)

    X_hold = promotion_holdout[NUMERIC + CATEGORICAL]
    y_hold = promotion_holdout[TARGET]
    verdict = promotion_gate(champion, challenger, X_hold, y_hold)
    verdict["partition_sizes"] = {
        "history": int(len(history)),
        "adaptation_batch": int(len(adaptation)),
        "promotion_holdout": int(len(promotion_holdout)),
    }
    verdict["scope"] = (
        "Offline random-split governance simulation on a static public sample; "
        "not a temporal or live retraining claim."
    )
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    artifact = joblib.load(args.model)
    df = validate(load_raw(ROOT / "data" / "raw" / "telco_churn.csv")).df

    # Drift-trigger demonstration on feature distributions.
    clean = df.sample(1500, random_state=7)
    shifted = clean.copy()
    shifted["tenure"] = (shifted["tenure"] * 0.5).astype(int)
    shifted["MonthlyCharges"] *= 1.3

    demo = {}
    for name, batch in [("clean_batch", clean), ("shifted_batch", shifted)]:
        fire, breaches = should_retrain(drift_report(batch, artifact))
        demo[name] = {"investigate_retraining": fire, "breaches": breaches}

    demo["promotion_simulation"] = governance_demo(df)
    demo["note"] = (
        "PSI can trigger investigation, but it does not automatically replace "
        "the model. Promotion requires lower expected decision cost on a "
        "disjoint holdout while AUC and Brier score remain within explicit "
        "guardrails."
    )
    out = ROOT / "reports" / "retrain_decision.json"
    out.write_text(json.dumps(demo, indent=2))
    print(json.dumps(demo, indent=2))


if __name__ == "__main__":
    main()
