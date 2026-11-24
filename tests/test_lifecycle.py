import json

import joblib
import numpy as np
import pytest

from validate import CATEGORICAL, NUMERIC, load_raw, validate

MODEL = "models/churn-2.0.0.joblib"


def test_bundled_local_sample_matches_pinned_ibm_checksum():
    import hashlib
    from download_data import EXPECTED_SHA256
    digest = hashlib.sha256(open("data/raw/telco_churn.csv", "rb").read()).hexdigest()
    assert digest == EXPECTED_SHA256


def test_validation_repairs_only_tenure_zero_totalcharges(tmp_path):
    base = load_raw("data/raw/telco_churn.csv").head(30).copy()
    base.loc[base.index[0], "TotalCharges"] = " "
    base.loc[base.index[0], "tenure"] = 0
    res = validate(base)
    assert res.df["TotalCharges"].dtype.kind == "f"
    assert any("repaired 1" in line for line in res.log)
    assert res.df["TotalCharges"].isna().sum() == 0

    bad = base.copy()
    bad.loc[bad.index[1], "TotalCharges"] = "not-a-number"
    bad.loc[bad.index[1], "tenure"] = 12
    with pytest.raises(ValueError, match="refusing silent repair"):
        validate(bad)


def test_validation_supports_inference_without_target():
    df = load_raw("data/raw/telco_churn.csv").head(10).drop(columns=["Churn"])
    out = validate(df, require_target=False).df
    assert "Churn" not in out.columns
    assert len(out) == 10


def test_validation_rejects_duplicate_customer_ids():
    df = load_raw("data/raw/telco_churn.csv").head(5).copy()
    dup = df.iloc[[0]].copy()
    df = __import__("pandas").concat([df, dup], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate customer ids"):
        validate(df)


def test_validation_rejects_missing_columns():
    df = load_raw("data/raw/telco_churn.csv").drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        validate(df)


def test_pipeline_learns_on_a_subsample():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from features import make_pipeline
    df = validate(load_raw("data/raw/telco_churn.csv")).df.sample(
        3000, random_state=0)
    X, y = df[NUMERIC + CATEGORICAL], df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y,
                                              random_state=0)
    pipe = make_pipeline(LogisticRegression(max_iter=2000,
                                            class_weight="balanced"))
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    assert auc > 0.78


def test_training_summary_pins_heldout_calibration_claim():
    summary = json.load(open("reports/training_summary.json"))
    cal = summary["calibration"]
    assert cal["heldout_test_brier_calibrated"] < cal["heldout_test_brier_raw"]
    assert cal["heldout_test_brier_raw"] == pytest.approx(0.1696, abs=1e-4)
    assert cal["heldout_test_brier_calibrated"] == pytest.approx(0.1363, abs=1e-4)
    assert summary["decision"]["deployed_threshold"] == pytest.approx(1 / 6, abs=1e-4)
    assert summary["decision"]["theoretical_and_empirical_same_validation_actions"] is True


def test_monitor_flags_shift_and_not_noise():
    from monitor import drift_report
    artifact = joblib.load(MODEL)
    df = validate(load_raw("data/raw/telco_churn.csv")).df
    clean = df.sample(1500, random_state=3)
    clean_psi = drift_report(clean, artifact)
    assert max(clean_psi.values()) < 0.1
    shifted = clean.copy()
    shifted["MonthlyCharges"] *= 1.5
    assert drift_report(shifted, artifact)["MonthlyCharges"] > 0.2


def test_dashboard_drift_drilldown_uses_the_same_shifted_batch():
    from dashboard_core import (drift_windows, latest_drift_batch,
                                reference_vs_current)
    artifact = joblib.load(MODEL)
    df = validate(load_raw("data/raw/telco_churn.csv")).df
    X = df[NUMERIC + CATEGORICAL].head(1200)
    hot = drift_windows(X, artifact, shift=True)
    last = hot[hot.window == hot.window.max()].set_index("feature").PSI
    shifted_batch = latest_drift_batch(X, shift=True)
    # Drill-down batch is genuinely shifted, not the original final chunk.
    unshifted_batch = latest_drift_batch(X, shift=False)
    assert shifted_batch.MonthlyCharges.mean() > unshifted_batch.MonthlyCharges.mean()
    frame = reference_vs_current("MonthlyCharges", shifted_batch, artifact)
    assert not frame.empty
    assert last["MonthlyCharges"] > 0.2


def test_retrain_trigger_and_multimetric_promotion_gate():
    from monitor import drift_report
    from retrain import governance_demo, should_retrain
    artifact = joblib.load(MODEL)
    df = validate(load_raw("data/raw/telco_churn.csv")).df
    clean = df.sample(1200, random_state=11)
    fire, _ = should_retrain(drift_report(clean, artifact))
    assert not fire
    shifted = clean.copy()
    shifted["tenure"] = (shifted["tenure"] * 0.4).astype(int)
    shifted["MonthlyCharges"] *= 1.4
    fire, breaches = should_retrain(drift_report(shifted, artifact))
    assert fire and "tenure" in breaches

    verdict = governance_demo(df)
    assert verdict["scope"].startswith("Offline random-split governance")
    assert verdict["partition_sizes"]["promotion_holdout"] > 1000
    assert {"roc_auc", "brier", "expected_cost"} <= set(verdict["champion"])
    assert {"roc_auc", "brier", "expected_cost"} <= set(verdict["challenger"])
    assert isinstance(verdict["promote"], bool)


def test_segment_thresholds_are_not_overclaimed():
    report = json.load(open("reports/segment_thresholds.json"))
    assert report["expected_cost_segmented"] < report["expected_cost_global"]
    ci = report["bootstrap_cost_saving_95pct"]
    assert ci["p2_5"] < 0 < ci["p97_5"]
    assert report["recommendation"] == "retain_global_threshold"
    assert report["segment_thresholds"]["Two year"]["churners_test"] == 7


def test_batch_cli_scores_feature_only_crm_file():
    from batch_score import score_frame
    artifact = joblib.load(MODEL)
    raw = load_raw("data/raw/telco_churn.csv").head(25).drop(columns=["Churn"])
    df = validate(raw, require_target=False).df
    scored = score_frame(df, artifact)
    assert {"churn_probability", "decision", "top_risk_driver",
            "top_protective_factor"} <= set(scored.columns)
    assert scored.churn_probability.between(0, 1).all()


def test_batch_scoring_rejects_unknown_categories():
    from batch_score import score_frame
    artifact = joblib.load(MODEL)
    raw = load_raw("data/raw/telco_churn.csv").head(5).drop(columns=["Churn"])
    df = validate(raw, require_target=False).df
    df.loc[df.index[0], "Contract"] = "Decade-long"
    with pytest.raises(ValueError, match="Unknown categorical values"):
        score_frame(df, artifact)
