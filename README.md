# Churn Prediction as a Decision Service

[![CI](https://github.com/AhmadEblagh205/churn-prediction-service/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/churn-prediction-service/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this on the public IBM Telco customer-churn sample. It treats churn as a decision problem: if a missed churner and an unnecessary retention offer have different costs, how should a calibrated probability become an action? The code covers validation, model selection, calibration, a cost-based threshold, an API, batch scoring, drift monitoring and an offline retraining simulation.

## What it does

- validates the data and keeps a cleaning log;
- splits into train, calibration/threshold and held-out test sets, then compares models by cross-validation;
- calibrates probabilities with isotonic regression and sets the threshold from the costs;
- serves predictions through FastAPI, with logistic log-odds reason codes, and scores CSV files that have no target column;
- monitors drift with PSI on a synthetically shifted sample, and runs an offline champion/challenger promotion check.

A Streamlit dashboard sits on top of the same computations (`pip install -r requirements-dashboard.txt`, then `streamlit run src/dashboard.py`).

## Results

Five-fold cross-validated ROC AUC on the training split:

| Model | Mean ROC AUC | Fold SD |
|---|---:|---:|
| Majority baseline | 0.500 | 0.000 |
| Logistic regression | 0.842 | 0.016 |
| Histogram gradient boosting | 0.841 | 0.012 |

The two models are effectively tied. I kept logistic regression because it gives exact log-odds reason codes.

On the held-out test split, isotonic calibration lowers the Brier score from 0.170 to 0.136. With a false-negative cost of 5 and a false-positive cost of 1, the threshold is 1/6, about 0.167. The validation optimum is 0.09, with a bootstrap 95% range of 0.09 to 0.18, and it gives the same validation actions in this run, so I deploy 1/6.

At that threshold the test split gives ROC AUC 0.843, recall 0.922, precision 0.410, flag rate 0.598 and expected cost 0.456 per customer. These figures describe this dataset and an illustrative cost ratio.

For the two-year-contract segment, the test group has 331 customers and only 7 churners. A segment threshold lowers expected cost from 0.4556 to 0.4514, but the bootstrap interval for the saving is -0.0263 to 0.0312, which includes zero. I keep the global threshold.

Figures are in reports/figures/. The reasoning behind the threshold, the segment check and the retraining gate is in docs/DECISION_GOVERNANCE.md.

## How to run

```bash
pip install -r requirements-dev.txt
python src/download_data.py                   # fetch + checksum-verify public sample
python src/train.py
python src/segment_thresholds.py
python src/monitor.py
python src/retrain.py
pytest -q
```

Run the API and score a file without a target column:

```bash
uvicorn serve:app --app-dir src --port 8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @examples/customer.json
python src/batch_score.py examples/batch_customers.csv --out reports/scored.csv
```

Or use Docker:

```bash
docker build -t churn-service .
docker run -p 8000:8000 churn-service
```

## Repository layout

```text
src/        validate, features, decision, train, explain, serve, batch_score,
            monitor, segment_thresholds, retrain, dashboard_core, dashboard
reports/    training_summary.json, segment_thresholds.json, drift_report.json,
            retrain_decision.json, figures/
models/     churn-2.0.0.joblib
docs/       DECISION_GOVERNANCE.md, METHODOLOGY.md
```

## Notes

The data is a public static sample, not a production cohort, and the costs are illustrative constants. The drift shift is synthetic and the PSI cut-offs are heuristics. The retraining workflow is an offline random-split simulation. Validation rules, the explanation method, the API response and the dashboard tabs are described in docs/METHODOLOGY.md. Data provenance and the upstream Apache-2.0 licence are in data/README.md.

## Licence

MIT for the code. The dataset is the public IBM Telco customer-churn sample downloaded by `src/download_data.py`.
