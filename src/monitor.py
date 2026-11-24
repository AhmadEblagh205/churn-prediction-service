"""Stage 10: the part most portfolios skip. A deployed model degrades
silently as its input distribution moves, so the service ships with a
monitor: population stability index per feature between the training
reference (stored inside the artifact) and any incoming batch, with the
conventional reading that under 0.1 is stable, 0.1 to 0.2 is worth
watching, and above 0.2 is an alert.

Usage:
    python src/monitor.py            # scores a clean and a shifted batch
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from validate import load_raw, validate

ROOT = Path(__file__).resolve().parents[1]
ALERT = 0.2


def psi_numeric(ref: dict, values: pd.Series) -> float:
    edges = np.array(ref["edges"])
    expected = np.array(ref["freqs"])
    counts, _ = np.histogram(values, bins=edges)
    actual = counts / max(counts.sum(), 1)
    e = np.clip(expected, 1e-4, None)
    a = np.clip(actual, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(ref: dict, values: pd.Series) -> float:
    actual = values.value_counts(normalize=True)
    cats = set(ref) | set(actual.index)
    e = np.array([max(ref.get(c, 0.0), 1e-4) for c in cats])
    a = np.array([max(actual.get(c, 0.0), 1e-4) for c in cats])
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(batch: pd.DataFrame, artifact: dict) -> dict:
    ref = artifact["reference"]
    out = {}
    for col, r in ref["numeric"].items():
        out[col] = round(psi_numeric(r, batch[col]), 4)
    for col, r in ref["categorical"].items():
        out[col] = round(psi_categorical(r, batch[col]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ROOT / "models" / "churn-2.0.0.joblib")
    ap.add_argument("--data", default=ROOT / "data" / "raw" /
                    "telco_churn.csv")
    args = ap.parse_args()
    artifact = joblib.load(args.model)
    df = validate(load_raw(args.data)).df

    clean = df.sample(1500, random_state=1)
    shifted = clean.copy()
    shifted["tenure"] = (shifted["tenure"] * 0.5).astype(int)   # younger book
    shifted["MonthlyCharges"] *= 1.3                            # repriced
    shifted.loc[shifted.sample(frac=0.5, random_state=2).index,
                "Contract"] = "Month-to-month"

    reports = {"clean_batch": drift_report(clean, artifact),
               "shifted_batch": drift_report(shifted, artifact)}
    alerts = [k for k, v in reports["shifted_batch"].items() if v > ALERT]
    reports["alerts_on_shifted"] = alerts

    feats = list(reports["clean_batch"])
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(feats) + 1.6))
    y = range(len(feats))
    ax.plot([reports["clean_batch"][f] for f in feats], y, "o",
            color="#0b3d91", label="clean batch")
    ax.plot([reports["shifted_batch"][f] for f in feats], y, "s",
            color="#c1440e", label="shifted batch")
    ax.axvline(ALERT, color="crimson", ls="--", lw=1.2,
               label=f"alert at {ALERT}")
    ax.set_yticks(y, feats, fontsize=9)
    ax.set(xlabel="Population stability index",
           title="Drift monitor: the shifted batch trips the alert")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "psi_drift.png", dpi=150)
    plt.close(fig)

    (ROOT / "reports" / "drift_report.json").write_text(
        json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
