# Decision and model-governance notes

This document records the reasoning behind the project's decision layer. It is intentionally separate from the predictive model.

## 1. Prediction is not the decision

The model estimates a calibrated churn probability `p`. The retention action depends on two assumed costs:

- `C_FN`: cost of missing a customer who churns;
- `C_FP`: cost of sending an unnecessary retention offer.

Act when the expected cost of not acting exceeds the expected cost of acting:

`p * C_FN > (1 - p) * C_FP`

which gives the break-even threshold:

`p > C_FP / (C_FN + C_FP)`.

For the illustrative 5:1 ratio, this is `1/6 ≈ 0.167`.

## 2. Why the empirical validation optimum is 0.09

The calibrated validation probabilities are produced by isotonic regression, which is piecewise constant. Multiple numeric thresholds can therefore produce exactly the same decisions.

The finite validation sample has a point-optimal threshold of 0.09, while the cost-derived 1/6 threshold produces the same validation classifications in this run. A non-parametric bootstrap of the empirical optimum gives a 95% range of 0.09–0.18.

The project therefore deploys the transparent cost-derived threshold and reports the empirical optimum as a diagnostic rather than treating 0.09 as a universal truth.

## 3. Calibration claim

The calibrator is fitted on the validation split. The claim that calibration improves probability quality is evaluated on the untouched test split:

- raw logistic Brier: 0.1696;
- calibrated Brier: 0.1363.

This avoids using the calibrator's own fitting sample as the headline evidence for improvement.

## 4. Segment-policy gate

Segment-specific thresholds are fitted on validation data and evaluated on the held-out test data.

Point estimate:

- global expected cost: 0.4556 per customer;
- segmented expected cost: 0.4514;
- observed saving: 0.0043.

Bootstrap 95% interval for `global cost - segmented cost`:

- -0.0263 to 0.0312.

Because the interval crosses zero, the evidence does not justify extra segment-specific policy complexity. The recommendation remains the global threshold.

The two-year test segment contains only seven churners, so its recall estimate is particularly uncertain.

## 5. Retraining trigger versus promotion

Drift is a trigger for investigation, not permission to deploy a new model.

The offline governance simulation creates disjoint random partitions representing historical data, a later adaptation batch and a promotion holdout. Both models are evaluated on the same promotion holdout, which neither uses for fitting/calibration.

The challenger is promoted only when:

- expected decision cost improves by at least the configured margin;
- ROC AUC does not regress beyond a small tolerance;
- Brier score does not regress beyond a small tolerance.

The current simulation passes the gate. This is evidence that the code path behaves coherently under one reproducible offline scenario, **not** evidence of live retraining performance.

## 6. Explanation boundary

Logistic feature contributions are exact additive terms in the underlying linear logit. The isotonic calibration layer remaps the logistic score to a calibrated probability and does not preserve coefficient-style probability additivity.

Therefore the API labels its reason codes as log-odds contributions and separately reports the calibrated probability.
