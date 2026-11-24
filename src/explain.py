"""Exact linear reason codes for the pre-calibration logistic model.

The deployed probability is produced by an isotonic calibration layer.
Feature contributions therefore explain the *underlying logistic logit*,
not an additive decomposition of the calibrated probability.  Isotonic
calibration is monotone, so the ordering of cases is preserved even though
there is no coefficient-style additive decomposition after calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _linear_parts(calibrated_pipeline):
    pipe = calibrated_pipeline.calibrated_classifiers_[0].estimator.estimator
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    return prep, model


def feature_contributions(calibrated_pipeline,
                          row: pd.DataFrame) -> list[tuple[str, float]]:
    """Log-odds contribution per original feature for one customer row."""
    prep, model = _linear_parts(calibrated_pipeline)
    z = prep.transform(row)[0]
    contrib = model.coef_[0] * z
    names = prep.get_feature_names_out()
    per_feature: dict[str, float] = {}
    for name, c in zip(names, contrib):
        base = name.split("__", 1)[1].split("_", 1)[0] \
            if name.startswith("cat__") else name.split("__", 1)[1]
        per_feature[base] = per_feature.get(base, 0.0) + float(c)
    ranked = sorted(per_feature.items(), key=lambda kv: -abs(kv[1]))
    return [(k, round(v, 4)) for k, v in ranked]


def explain_prediction(calibrated_pipeline, row: pd.DataFrame,
                       top: int = 3) -> dict:
    prep, model = _linear_parts(calibrated_pipeline)
    ranked = feature_contributions(calibrated_pipeline, row)
    raising = [{"feature": k, "log_odds_contribution": v}
               for k, v in ranked if v > 0][:top]
    lowering = [{"feature": k, "log_odds_contribution": v}
                for k, v in ranked if v < 0][:top]
    raw_logit = float(model.decision_function(prep.transform(row))[0])
    raw_probability = float(1.0 / (1.0 + np.exp(-raw_logit)))
    calibrated_probability = float(
        calibrated_pipeline.predict_proba(row)[0, 1])
    return {
        "raw_logit": raw_logit,
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "raising_factors": raising,
        "lowering_factors": lowering,
        "basis": ("Exact additive contributions to the underlying logistic "
                  "log-odds; isotonic calibration is a monotone remapping "
                  "of the final probability."),
    }


def top_factors(calibrated_pipeline, row: pd.DataFrame,
                top: int = 3) -> list[dict]:
    """Backward-compatible alias for positive risk drivers."""
    out = explain_prediction(calibrated_pipeline, row, top=top)["raising_factors"]
    return out or [{"feature": "none", "log_odds_contribution": 0.0}]
