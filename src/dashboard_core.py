"""Pure computation behind the dashboard, kept free of streamlit so the
numbers are unit testable on their own and the UI file stays thin. Every
figure the dashboard shows is produced here; the UI only arranges it.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from decision import classification_metrics, empirical_threshold, theoretical_threshold
from explain import top_factors
from monitor import drift_report
from validate import CATEGORICAL, NUMERIC, TARGET, load_raw, validate

ROOT = Path(__file__).resolve().parents[1]
SEGMENT_COL = "Contract"


def load_assets():
    """Artifact plus the same validation and test splits the training
    run held out, with probabilities precomputed."""
    from sklearn.model_selection import train_test_split
    artifact = joblib.load(ROOT / "models" / "churn-2.0.0.joblib")
    result = validate(load_raw(ROOT / "data" / "raw" / "telco_churn.csv"))
    df = result.df
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=0)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0)
    p_val = artifact["pipeline"].predict_proba(X_val)[:, 1]
    p_te = artifact["pipeline"].predict_proba(X_te)[:, 1]
    ids = df.loc[X_te.index, "customerID"] if "customerID" in df else None
    return {"artifact": artifact, "clean_log": result.log,
            "val": (X_val, y_val.to_numpy(), p_val),
            "test": (X_te, y_te.to_numpy(), p_te), "test_ids": ids}


def segment_mask(X: pd.DataFrame, segments: list[str]) -> np.ndarray:
    if not segments:
        return np.ones(len(X), dtype=bool)
    return X[SEGMENT_COL].isin(segments).to_numpy()


def threshold_for_ratio(p_val, y_val, fn_cost: float) -> float:
    # The decision tab follows the same transparent cost-derived policy as
    # the deployed artifact. p_val/y_val remain in the signature because the
    # UI also passes them to the exploratory segment audit below.
    _ = p_val, y_val
    return theoretical_threshold(fn_cost, 1.0)


def metrics_at(p, y, t, fn_cost: float) -> dict:
    m = classification_metrics(y, p, t, fn_cost, 1.0)
    return {
        "recall": m["recall"], "precision": m["precision"],
        "flag_rate": m["flag_rate"],
        "cost_per_customer": m["cost_per_customer"],
        "offers": m["offers"], "caught": m["caught"],
        "missed": m["missed"], "spared": m["spared"], "n": m["n"],
    }


def cost_curve(p, y, fn_cost: float, n: int = 45) -> pd.DataFrame:
    ts = np.linspace(0.02, 0.9, n)
    return pd.DataFrame({
        "threshold": ts,
        "expected cost": [metrics_at(p, y, t, fn_cost)["cost_per_customer"]
                          for t in ts],
    })


def per_segment_thresholds(p_val, y_val, seg_val: pd.Series,
                           fn_cost: float) -> pd.DataFrame:
    rows = []
    for seg in sorted(seg_val.unique()):
        m = (seg_val == seg).to_numpy()
        t, _ = empirical_threshold(y_val[m], p_val[m], fn_cost, 1.0)
        met = metrics_at(p_val[m], y_val[m], t, fn_cost)
        rows.append({"segment": seg, "threshold": round(t, 2),
                     "recall": round(met["recall"], 2),
                     "flag rate": round(met["flag_rate"], 2)})
    return pd.DataFrame(rows)


def calibration_frame(p, y, bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append({"predicted": float(p[m].mean()),
                         "observed": float(y[m].mean()),
                         "count": int(m.sum())})
    return pd.DataFrame(rows)


def score_distribution(p, y) -> pd.DataFrame:
    edges = np.linspace(0, 1, 21)
    mids = (edges[:-1] + edges[1:]) / 2
    rows = []
    for label, m in (("stayed", y == 0), ("churned", y == 1)):
        hist, _ = np.histogram(p[m], bins=edges, density=True)
        rows += [{"score": float(s), "density": float(h), "class": label}
                 for s, h in zip(mids, hist)]
    return pd.DataFrame(rows)


def apply_demo_shift(batch: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented synthetic drift scenario to one batch."""
    out = batch.copy()
    out["tenure"] = (out["tenure"] * 0.5).astype(int)
    out["MonthlyCharges"] = out["MonthlyCharges"] * 1.3
    return out


def latest_drift_batch(X: pd.DataFrame, n_windows: int = 6,
                       shift: bool = False) -> pd.DataFrame:
    chunks = np.array_split(np.arange(len(X)), n_windows)
    batch = X.iloc[chunks[-1]].copy()
    return apply_demo_shift(batch) if shift else batch


def drift_windows(X: pd.DataFrame, artifact: dict, n_windows: int = 6,
                  shift: bool = False) -> pd.DataFrame:
    """PSI per feature over consecutive windows of the frame, the last
    window optionally shifted to demonstrate the alert."""
    rows = []
    chunks = np.array_split(np.arange(len(X)), n_windows)
    for w, idx in enumerate(chunks, start=1):
        batch = X.iloc[idx].copy()
        if shift and w == n_windows:
            batch = apply_demo_shift(batch)
        for feat, val in drift_report(batch, artifact).items():
            rows.append({"window": w, "feature": feat, "PSI": val})
    return pd.DataFrame(rows)


def reference_vs_current(feature: str, X: pd.DataFrame,
                         artifact: dict) -> pd.DataFrame:
    """Long frame of reference proportions against the current window's,
    on the reference's own bins, for the drill down chart."""
    ref = artifact["reference"]
    rows = []
    if feature in ref["numeric"]:
        r = ref["numeric"][feature]
        edges = np.array(r["edges"], dtype=float)
        counts, _ = np.histogram(X[feature].to_numpy(),
                                 bins=np.nan_to_num(edges, neginf=-1e12,
                                                    posinf=1e12))
        cur = counts / max(counts.sum(), 1)
        for i, (rf, cf) in enumerate(zip(r["freqs"], cur)):
            label = f"bin {i + 1}"
            rows.append({"bin": label, "share": rf, "which": "reference"})
            rows.append({"bin": label, "share": float(cf),
                         "which": "current"})
    else:
        r = ref["categorical"][feature]
        cur = X[feature].value_counts(normalize=True)
        for cat in r:
            rows.append({"bin": str(cat), "share": r[cat],
                         "which": "reference"})
            rows.append({"bin": str(cat),
                         "share": float(cur.get(cat, 0.0)),
                         "which": "current"})
    return pd.DataFrame(rows)


def explain_row(artifact: dict, row: pd.DataFrame) -> tuple[float, list]:
    proba = float(artifact["pipeline"].predict_proba(row)[0, 1])
    return proba, top_factors(artifact["pipeline"], row)


def latency_ms(artifact: dict, row: pd.DataFrame, n: int = 50) -> float:
    start = time.perf_counter()
    for _ in range(n):
        artifact["pipeline"].predict_proba(row)
    return (time.perf_counter() - start) / n * 1000
