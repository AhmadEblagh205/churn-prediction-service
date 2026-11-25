# Methodology

## Notes moved from the README

### Data validation

Dataset provenance, checksum and the upstream Apache-2.0 licence are recorded in data/README.md.

The IBM sample stores TotalCharges as text and contains blank values for tenure-zero customers. The validator converts TotalCharges explicitly, repairs a blank only when tenure is zero, and raises an error for malformed or missing TotalCharges on non-zero-tenure rows instead of turning it into zero. It also checks required columns and duplicate customer ids, rejects negative numeric values, normalises SeniorCitizen to Yes/No, and can validate training data with Churn or scoring data without the target. A real scoring file cannot be expected to contain the outcome being predicted.

### Feature pipeline

Numerical scaling and categorical one-hot encoding live inside a scikit-learn ColumnTransformer/Pipeline, so the same transformation is used in training, testing and serving. Unknown categories are ignored by the encoder at the model layer, while the API performs a stricter schema check and returns a 422 for unsupported categories.

### Explanations

Because the champion is logistic regression, each transformed feature contributes exactly coefficient x transformed value to the uncalibrated logistic log-odds. The service aggregates one-hot columns back to the original input fields and returns the top risk-raising and top protective factors. The explanation is not an additive decomposition of the calibrated probability, because isotonic calibration is a monotone remapping applied after the logistic score.

### Drift monitoring

The model artifact carries reference distributions from the training data. monitor.py computes the population stability index (PSI) for selected numeric and categorical features. It compares a clean sample with a synthetically shifted sample (shorter tenure, higher charges and a contract shift). The PSI thresholds are demonstration heuristics, not production standards. The Streamlit dashboard uses the same shifted latest batch for the drift trend and the drill-down distributions.

### Retraining

retrain.py separates two decisions: investigate retraining when the drift rule fires, and promote a challenger only on a disjoint promotion holdout. The promotion gate is cost-first and also checks AUC and Brier-score guardrails. In the current run the challenger passes because it reduces expected cost on the promotion holdout within the AUC and Brier tolerances. This is an offline random-split simulation on a static public sample. See DECISION_GOVERNANCE.md for the reasoning.

### API and dashboard

The /predict response contains the calibrated churn probability, the raw logistic probability, the deployed threshold and policy, the decision, risk drivers, protective factors and the model version. Batch API scoring supports up to 500 customers. The dashboard has five tabs: Overview, Decision (change the miss cost and see the threshold respond), Inspect (customer what-if with log-odds drivers), Drift and Batch (upload a target-free file and download scores). Its numerical logic is in dashboard_core.py. The example API and dashboard have no authentication.
