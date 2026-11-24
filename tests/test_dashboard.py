from pathlib import Path

import numpy as np
import pytest

st = pytest.importorskip("streamlit")

APP = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"


def by_key(elements, key):
    matches = [e for e in elements if e.key == key]
    assert matches, f"no element with key {key}"
    return matches[0]


def test_threshold_follows_the_cost_ratio():
    from dashboard import metrics_at, threshold_for_ratio
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(int)
    p = np.clip(0.3 * y + rng.normal(0.2, 0.15, 2000), 0.01, 0.99)
    t_low = threshold_for_ratio(p, y, fn_cost=2.0)
    t_high = threshold_for_ratio(p, y, fn_cost=9.0)
    assert t_high < t_low          # dearer misses push the threshold down
    m_low = metrics_at(p, y, t_low, 2.0)
    m_high = metrics_at(p, y, t_high, 9.0)
    assert m_high["recall"] > m_low["recall"]
    assert m_high["offers"] >= m_low["offers"]


def test_core_segments_calibration_and_drift():
    from dashboard_core import (calibration_frame, drift_windows,
                                load_assets, per_segment_thresholds)
    a = load_assets()
    X_val, y_val, p_val = a["val"]
    table = per_segment_thresholds(p_val, y_val, X_val.Contract, 5.0)
    assert set(table.segment) == {"Month-to-month", "One year", "Two year"}
    two = table[table.segment == "Two year"].iloc[0]
    assert two.threshold > a["artifact"]["threshold"]   # the verdict, live
    X_te, y_te, p_te = a["test"]
    cal = calibration_frame(p_te, y_te)
    assert (cal.predicted - cal.observed).abs().max() < 0.15
    calm = drift_windows(X_te, a["artifact"], shift=False)
    hot = drift_windows(X_te, a["artifact"], shift=True)
    lastw = hot.window.max()
    assert calm[calm.window == lastw].PSI.max() < 0.2
    assert hot[hot.window == lastw].PSI.max() > 0.5


def test_dashboard_boots_and_reacts():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    ratio = by_key(at.slider, "ratio")
    assert ratio.value == 5.0
    threshold_before = float(at.metric[6].value)   # Decision tab, col 0
    by_key(at.slider, "ratio").set_value(9.0)
    at.run()
    assert not at.exception
    assert float(at.metric[6].value) <= threshold_before


def test_dashboard_inspector_and_drift_react():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    tenure = by_key(at.number_input, "tenure_input")
    tenure.set_value(1)
    at.run()
    assert not at.exception
    prob_at_1 = float(at.metric[11].value.rstrip("%"))
    by_key(at.number_input, "tenure_input").set_value(70)
    at.run()
    assert not at.exception
    prob_at_70 = float(at.metric[11].value.rstrip("%"))
    assert prob_at_70 < prob_at_1        # long tenure, lower churn risk
    by_key(at.checkbox, "simulate").set_value(True)
    at.run()
    assert not at.exception
    assert any("Drift alert" in str(e.value) for e in at.error)
