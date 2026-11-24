"""Batch scoring from the command line.

Input is a CRM-style feature file: customer id plus model features.  The
future ``Churn`` outcome is intentionally optional because inference data
cannot be expected to contain the target being predicted.

Usage:
    python src/batch_score.py examples/batch_customers.csv --out scored.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from explain import explain_prediction
from validate import CATEGORICAL, NUMERIC, load_raw, validate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "churn-2.0.0.joblib"


def validate_scoring_categories(df: pd.DataFrame, artifact) -> None:
    """Reject categories unseen by the deployed artifact.

    The underlying encoder can technically ignore unknown categories, but a
    batch job should fail loudly rather than score a customer whose category
    would silently become an all-zero dummy block.
    """
    problems = {}
    for col, allowed in artifact["allowed_values"].items():
        seen = set(df[col].astype(str).unique())
        unknown = sorted(seen - set(map(str, allowed)))
        if unknown:
            problems[col] = unknown
    if problems:
        raise ValueError(f"Unknown categorical values for deployed model: {problems}")


def score_frame(df: pd.DataFrame, artifact) -> pd.DataFrame:
    validate_scoring_categories(df, artifact)
    X = df[NUMERIC + CATEGORICAL]
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["decision"] = ["target retention offer" if p >= artifact["threshold"]
                       else "no action" for p in proba]
    reasons = []
    protective = []
    for i in range(len(X)):
        ex = explain_prediction(artifact["pipeline"], X.iloc[[i]], top=1)
        reasons.append(ex["raising_factors"][0]["feature"]
                       if ex["raising_factors"] else "none")
        protective.append(ex["lowering_factors"][0]["feature"]
                          if ex["lowering_factors"] else "none")
    out["top_risk_driver"] = reasons
    out["top_protective_factor"] = protective
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=ROOT / "reports" / "scored.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    artifact = joblib.load(args.model)
    result = validate(load_raw(args.csv), require_target=False)
    df = result.df
    if args.limit:
        df = df.head(args.limit)
    scored = score_frame(df, artifact)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out, index=False)
    flagged = (scored.decision != "no action").mean()
    print(f"Scored {len(scored)} customers -> {args.out} "
          f"(flag rate {flagged:.0%})")
    if result.log:
        print("Validation: " + "; ".join(result.log))


if __name__ == "__main__":
    main()
