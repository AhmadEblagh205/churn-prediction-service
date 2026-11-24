import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import serve
from explain import _linear_parts, feature_contributions

client = TestClient(serve.app)
EXAMPLE = json.load(open("examples/customer.json"))


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["version"] == "2.0.0"


def test_predict_returns_probability_decision_and_explanation_scope():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert 0.0 <= body["raw_logistic_probability"] <= 1.0
    assert body["decision"] in {"target retention offer", "no action"}
    assert "log-odds" in body["explanation_basis"]
    assert "protective_factors" in body


def test_predict_rejects_unknown_category():
    bad = dict(EXAMPLE, Contract="Decade-long")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_invalid_types():
    bad = dict(EXAMPLE, tenure=-3)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_reason_codes_are_exact_for_raw_logit():
    r = client.post("/predict", json=EXAMPLE)
    body = r.json()
    assert all(f["log_odds_contribution"] >= 0 for f in body["top_factors"])
    assert all(f["log_odds_contribution"] <= 0 for f in body["protective_factors"])
    row = pd.DataFrame([EXAMPLE])
    prep, model = _linear_parts(serve.ARTIFACT["pipeline"])
    logit = float(model.decision_function(prep.transform(row))[0])
    total = sum(v for _, v in feature_contributions(serve.ARTIFACT["pipeline"], row))
    assert np.isclose(total + float(model.intercept_[0]), logit, atol=1e-3)


def test_batch_endpoint_scores_and_summarises():
    batch = [EXAMPLE, dict(EXAMPLE, tenure=1, Contract="Month-to-month")]
    r = client.post("/predict_batch", json=batch)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2 and len(body["results"]) == 2
    assert 0 <= body["flag_rate"] <= 1
    assert all("top_factors" in row for row in body["results"])
