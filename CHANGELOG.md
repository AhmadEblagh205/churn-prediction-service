# Changelog

## 2.0.0, decision governance and defensibility
- Reframed the public project around prediction, calibration and decision-making rather than a single tuned threshold.
- The deployed 5:1 decision threshold is now the transparent cost-derived 1/6 rule; the empirical validation optimum (0.09) is retained as a diagnostic and bootstrapped for stability.
- Calibration improvement is now reported on the untouched test split (Brier 0.1696 -> 0.1363), not on the calibrator fitting split.
- Model selection reports fold-level ROC-AUC variation and treats logistic vs gradient boosting as comparable rather than claiming a meaningful 0.0017 win.
- Segment thresholds now carry bootstrap uncertainty and default to retaining the global policy because the observed cost saving is not statistically robust.
- Batch scoring no longer requires the target column. Malformed non-zero-tenure TotalCharges values now fail loudly instead of being silently repaired.
- Drift drill-down uses the same synthetically shifted latest batch as the trend. Wording now makes clear the data are successive simulated batches, not a real time series.
- Retraining is now an explicitly offline governance simulation with a disjoint promotion holdout and cost/AUC/Brier guardrails.
- API explanations distinguish exact raw logistic log-odds contributions from the calibrated probability and expose protective as well as risk-raising factors.
- CI now installs and executes the optional Streamlit dashboard test suite.

## 1.6.0, the surface, professional
The dashboard grown up: a themed five tab app over a UI free computation core. Overview is the live model card (calibration curve hugging the diagonal, class score separation, the validator's cleaning log). Decision adds segment filters, deltas against the deployed threshold, an offers sent, caught and missed strip, and the per segment threshold table with the two year verdict visible live. Inspect is per customer what if analysis, move tenure, contract or charges and watch the probability and exact reason codes follow. Drift tracks worst PSI over successive batches with a reference versus current drill down. Batch ships a sample file, shows the validator log and repair count, and returns the scored CSV. Every widget carries a key and the AppTest suite drives the app like a user: the threshold falls when misses get dearer, long tenure lowers the inspected probability, the simulated shift trips the alert.

## 1.5.0, the stakeholder surface
A stakeholder dashboard (Streamlit, optional extra like shap): the Decision tab turns the cost ratio into a slider and lets the threshold, recall and expected cost follow it live, Drift runs the PSI monitor with a simulated shift toggle so the alert can be watched firing, and Batch scores an uploaded CSV in the browser. Tested headlessly through streamlit's AppTest: the app boots, the threshold falls when misses get dearer, and the simulated shift trips the alert.

## 1.4.0, update round 9
Batch functionality: a predict_batch endpoint (up to 500 customers, per row validation and reasons plus a batch summary) and a CLI that scores a raw schema CSV into a CRM ready file with probability, decision and top reason columns.

## 1.3.0, update round 8
Added an exploratory per-segment threshold analysis. This early version focused on the lower point estimate of expected cost; v2.0.0 adds bootstrap uncertainty and reverses the deployment recommendation because the observed saving is not robust.

## 1.2.0, update round 7
The API now explains itself: exact per prediction reason codes for the linear champion, one hot contributions summed back to original features, returned as top factors on every response, with an additivity test proving the attributions sum to the logit.

## 1.1.0, update round 6
Added the first retraining-trigger and champion/challenger prototype. Its evaluation partition was later judged insufficiently clean for a defensible promotion claim; v2.0.0 replaces it with a disjoint offline promotion-holdout simulation and multi-metric guardrails.

## 1.0.0
Initial release: the full lifecycle on the public IBM Telco sample, validation with a cleaning log, pipeline features, CV model selection, isotonic calibration, a cost optimal threshold, segment error analysis, a versioned artifact with an auto generated model card, a FastAPI service with pydantic validation, a Dockerfile, and a PSI drift monitor with a shifted batch demonstration.
