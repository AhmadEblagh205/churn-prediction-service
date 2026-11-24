# Model card: churn-2.0.0

**Purpose.** Estimate churn probability and support retention decisions under an
explicit false-negative:false-positive cost ratio (5:1).

**Data.** IBM Telco customer-churn public sample, 7043 rows after validation,
churn rate 26.5%. Cleaning interventions: TotalCharges: coerced text to numeric; repaired 11 tenure-zero blank values with 0.0; SeniorCitizen: recoded 0/1 to No/Yes for consistency; target encoded; churn rate 0.265 on 7043 customers.

**Model selection.** Logistic regression and histogram gradient boosting have
similar cross-validated discrimination on the training split. The deployed
champion is **logistic** (mean ROC AUC 0.842 ±
0.015); selection rule: logistic has the highest mean ROC-AUC, but the 0.0017 gap to hist_gbm is within the 0.005 comparability tolerance; no meaningful win is claimed, and logistic is retained for exact linear reason codes.

**Calibration.** Isotonic calibration is fitted only on the validation split.
On the untouched test split, Brier score improves from **0.170 to
0.136**.

**Decision rule.** With calibrated probabilities and costs 5:1,
the transparent Bayes threshold is **0.167**. The validation sample's
point-optimal threshold is 0.09, but its bootstrap 95% range is
0.09–0.18; the point optimum is
therefore treated as a diagnostic rather than a universal business constant.

**Held-out test.** ROC AUC 0.843; churner recall
0.92 at precision 0.41;
flag rate 60%; expected cost per customer
0.456 under the stated ratio.

**Explanations.** Reason codes are exact additive contributions to the underlying
logistic **log-odds**. They do not claim to add up to the calibrated probability;
isotonic calibration is a monotone remapping applied after the logistic score.

**Known limitations.** This is a static public sample, not a longitudinal
production cohort. Segment estimates can be noisy (the held-out two-year group
contains only 7 churners). Cost ratios are
illustrative and assumed constant per customer. Drift and retraining modules are
offline governance demonstrations rather than claims of a live production loop.
