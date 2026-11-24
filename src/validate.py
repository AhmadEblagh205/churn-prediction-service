"""Schema validation and explicit cleaning for the IBM Telco sample.

The dataset stores ``TotalCharges`` as text and contains blank values for
new customers with zero tenure.  Only that schema-consistent case is
repaired automatically.  Any other non-numeric ``TotalCharges`` value is
rejected rather than silently converted to zero.

Training/evaluation data require the target column.  Inference batches do
not: a CRM export cannot be expected to contain the future churn outcome
I am trying to predict.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents",
               "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod"]
FEATURE_COLUMNS = NUMERIC + CATEGORICAL


@dataclass
class CleanResult:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _required_columns(require_target: bool) -> set[str]:
    required = {ID_COL, *FEATURE_COLUMNS}
    if require_target:
        required.add(TARGET)
    return required


def validate(raw: pd.DataFrame, *, require_target: bool = True) -> CleanResult:
    log: list[str] = []
    df = raw.copy()

    missing = _required_columns(require_target) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dupes = int(df[ID_COL].duplicated().sum())
    if dupes:
        ids = df.loc[df[ID_COL].duplicated(keep=False), ID_COL] \
            .astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"duplicate customer ids detected ({dupes} duplicate rows); "
            f"example ids: {ids}"
        )

    # Known schema trap: blank TotalCharges occurs for tenure-zero customers.
    raw_tc = df["TotalCharges"]
    tc = pd.to_numeric(raw_tc, errors="coerce")
    invalid = tc.isna()
    if invalid.any():
        repairable = invalid & (df["tenure"] == 0)
        unrepaired = invalid & ~repairable
        if unrepaired.any():
            ids = df.loc[unrepaired, ID_COL].astype(str).head(5).tolist()
            raise ValueError(
                "TotalCharges contains non-numeric/missing values for "
                "non-zero-tenure customers; refusing silent repair. "
                f"Example customer ids: {ids}"
            )
        n = int(repairable.sum())
        tc.loc[repairable] = 0.0
        log.append(
            f"TotalCharges: coerced text to numeric; repaired {n} "
            "tenure-zero blank values with 0.0"
        )
    else:
        # Record the conversion because the source column is text in the
        # public sample even when a particular batch has no blanks.
        log.append("TotalCharges: coerced text to numeric")
    df["TotalCharges"] = tc.astype(float)

    # Fail rather than silently dropping impossible feature values.
    for col in NUMERIC:
        bad = df[col] < 0
        if bad.any():
            ids = df.loc[bad, ID_COL].astype(str).head(5).tolist()
            raise ValueError(
                f"{col} contains negative values; example customer ids: {ids}"
            )

    senior = df["SeniorCitizen"]
    # Raw training file uses 0/1; inference files may already use No/Yes.
    mapped = senior.map({0: "No", 1: "Yes", "0": "No", "1": "Yes",
                         "No": "No", "Yes": "Yes"})
    if mapped.isna().any():
        bad = sorted(senior[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"SeniorCitizen has unsupported values: {bad}")
    if not senior.astype(str).isin(["No", "Yes"]).all():
        log.append("SeniorCitizen: recoded 0/1 to No/Yes for consistency")
    df["SeniorCitizen"] = mapped

    if TARGET in df.columns:
        target = df[TARGET]
        allowed = set(target.dropna().unique().tolist())
        if allowed <= {0, 1}:
            df[TARGET] = target.astype(int)
        elif allowed <= {"Yes", "No"}:
            df[TARGET] = (target == "Yes").astype(int)
        else:
            raise ValueError(f"{TARGET} has unsupported values: {sorted(map(str, allowed))}")
        log.append(f"target encoded; churn rate {df[TARGET].mean():.3f} "
                   f"on {len(df)} customers")
    elif require_target:
        # Defensive: normally caught by missing-column check above.
        raise ValueError(f"Missing required columns: ['{TARGET}']")

    return CleanResult(df=df, log=log)


def allowed_values(df: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}
