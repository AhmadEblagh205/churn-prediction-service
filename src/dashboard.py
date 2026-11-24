"""Stage 13, the stakeholder surface, grown up. A themed five tab
Streamlit app over the deployed artifact: Overview is the live model
card (calibration, score separation, the validator's own cleaning log),
Decision turns the cost ratio into a slider with every metric following
on the untouched test split, Inspect is per customer what if analysis
with exact reason codes, Drift watches PSI over successive batches with a
demonstrable alert, and Batch scores a CSV in the browser. All numbers
come from dashboard_core, which never imports streamlit, so the UI is a
thin arrangement over unit tested computation, and every widget carries
a key so the headless AppTest suite drives the app like a user.

Run:  streamlit run src/dashboard.py     (pip install streamlit first)
"""
import altair as alt
import pandas as pd
import streamlit as st

import dashboard_core as core
from dashboard_core import metrics_at, threshold_for_ratio  # noqa: F401  (re-export for tests)

BLUE, RUST = "#0b3d91", "#c1440e"


@st.cache_resource(show_spinner="Loading artifact and splits...")
def assets():
    return core.load_assets()


def header(a):
    art = a["artifact"]
    st.title("Churn retention service")
    st.caption("Live view over the deployed artifact; the model is "
               "frozen, the decisions are adjustable.")
    X_te, y_te, p_te = a["test"]
    lat = core.latency_ms(art, X_te.iloc[[0]])
    c = st.columns(6)
    c[0].metric("Model", art["model_name"])
    c[1].metric("Version", art["version"])
    c[2].metric("Trained", str(art["trained"]))
    c[3].metric("Deployed threshold", f"{art['threshold']:.2f}")
    c[4].metric("Churn base rate", f"{y_te.mean():.0%}")
    c[5].metric("Latency, 1 row", f"{lat:.0f} ms")


def overview_tab(a):
    X_te, y_te, p_te = a["test"]
    left, right = st.columns(2)
    with left:
        st.subheader("Calibration on the test split")
        cal = core.calibration_frame(p_te, y_te)
        line = alt.Chart(cal).mark_line(point=True, color=BLUE).encode(
            x=alt.X("predicted", title="predicted churn probability"),
            y=alt.Y("observed", title="observed churn rate"),
            tooltip=["predicted", "observed", "count"])
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})) \
            .mark_line(strokeDash=[4, 4], color="grey").encode(x="x", y="y")
        st.altair_chart(diag + line, use_container_width=True)
        st.caption("Isotonic calibration at work: the curve hugs the "
                   "diagonal, so the probabilities mean what they say.")
    with right:
        st.subheader("Score separation")
        dist = core.score_distribution(p_te, y_te)
        chart = alt.Chart(dist).mark_line(interpolate="step").encode(
            x=alt.X("score", title="predicted probability"),
            y=alt.Y("density", title="density"),
            color=alt.Color("class",
                            scale=alt.Scale(range=[RUST, BLUE])))
        st.altair_chart(chart, use_container_width=True)
        st.caption("Churners pile up on the right, stayers on the left; "
                   "the overlap is why thresholds are a business choice.")
    with st.expander("What the validator did to the raw file"):
        for line in a["clean_log"]:
            st.write("- " + str(line))


def decision_tab(a, segs):
    art = a["artifact"]
    X_val, y_val, p_val = a["val"]
    X_te, y_te, p_te = a["test"]
    mv, mt = core.segment_mask(X_val, segs), core.segment_mask(X_te, segs)
    fn_cost = st.slider("Cost of a missed churner (an unneeded offer "
                        "costs 1)", 1.0, 10.0, float(art["costs"]["fn"]),
                        0.5, key="ratio")
    t = core.threshold_for_ratio(p_val[mv], y_val[mv], fn_cost)
    m = core.metrics_at(p_te[mt], y_te[mt], t, fn_cost)
    base = core.metrics_at(p_te[mt], y_te[mt], art["threshold"],
                           fn_cost)
    c = st.columns(5)
    c[0].metric("Threshold", f"{t:.2f}",
                delta=f"{t - art['threshold']:+.2f} vs deployed")
    c[1].metric("Churner recall", f"{m['recall']:.0%}",
                delta=f"{(m['recall'] - base['recall']):+.0%}")
    c[2].metric("Precision", f"{m['precision']:.0%}")
    c[3].metric("Flag rate", f"{m['flag_rate']:.0%}")
    c[4].metric("Cost per customer", f"{m['cost_per_customer']:.3f}",
                delta=f"{m['cost_per_customer'] - base['cost_per_customer']:+.3f}",
                delta_color="inverse")
    st.caption(f"On these {m['n']:,} test customers: {m['offers']:,} "
               f"offers sent, {m['caught']:,} churners caught, "
               f"{m['missed']:,} missed.")
    curve = core.cost_curve(p_te[mt], y_te[mt], fn_cost)
    line = alt.Chart(curve).mark_line(color=BLUE).encode(
        x=alt.X("threshold"), y=alt.Y("expected cost"))
    rule = alt.Chart(pd.DataFrame({"t": [t]})).mark_rule(
        color=RUST).encode(x="t")
    dep = alt.Chart(pd.DataFrame({"t": [art["threshold"]]})).mark_rule(
        strokeDash=[4, 4], color="grey").encode(x="t")
    st.altair_chart(line + rule + dep, use_container_width=True)
    st.subheader("What this ratio implies per segment")
    st.dataframe(core.per_segment_thresholds(p_val, y_val,
                                             X_val[core.SEGMENT_COL],
                                             fn_cost),
                 hide_index=True, use_container_width=True)
    st.caption("Segment thresholds are exploratory. Small segments can make "
               "point-optimal thresholds unstable; the offline governance "
               "report bootstraps the cost difference before recommending "
               "any segment-specific policy.")


def inspect_tab(a):
    art = a["artifact"]
    X_te, y_te, p_te = a["test"]
    ids = a["test_ids"]
    labels = [f"{ids.iloc[i]}" if ids is not None else f"row {i}"
              for i in range(min(len(X_te), 200))]
    pick = st.selectbox("Customer", labels, key="inspect_customer")
    i = labels.index(pick)
    row = X_te.iloc[[i]].copy()
    st.caption("Change the levers and watch the probability and the "
               "reasons move; everything else stays as this customer "
               "has it.")
    c1, c2, c3 = st.columns(3)
    tenure = c1.number_input("Tenure, months", 0, 72,
                             int(row.tenure.iloc[0]), key="tenure_input")
    contract = c2.selectbox("Contract",
                            sorted(X_te.Contract.unique()),
                            index=sorted(X_te.Contract.unique())
                            .index(row.Contract.iloc[0]),
                            key="contract_input")
    monthly = c3.number_input("Monthly charges", 0.0, 200.0,
                              float(row.MonthlyCharges.iloc[0]),
                              key="monthly_input")
    row.loc[:, "tenure"] = tenure
    row.loc[:, "Contract"] = contract
    row.loc[:, "MonthlyCharges"] = monthly
    proba, factors = core.explain_row(art, row)
    decision = "target retention offer" if proba >= art["threshold"] \
        else "no action"
    c = st.columns(3)
    c[0].metric("Churn probability", f"{proba:.0%}")
    c[1].metric("Deployed threshold", f"{art['threshold']:.2f}")
    c[2].metric("Decision", decision)
    st.subheader("Why")
    for f in factors:
        st.write(f"- **{f['feature']}** pushes the log odds up by "
                 f"{f['log_odds_contribution']:.2f}")


def drift_tab(a):
    art = a["artifact"]
    X_te, _, _ = a["test"]
    shift = st.checkbox("Simulate a shifted latest batch (tenure squeezed, "
                        "charges inflated)", key="simulate")
    dw = core.drift_windows(X_te, art, shift=shift)
    worst = dw.groupby("window").PSI.max().reset_index()
    last = dw[dw.window == dw.window.max()]
    alerts = last[last.PSI >= 0.2].feature.tolist()
    if alerts:
        st.error(f"Drift alert in the latest window: {', '.join(alerts)}")
    else:
        st.success("Latest window clean: no feature over the 0.2 line.")
    trend = alt.Chart(worst).mark_line(point=True, color=BLUE).encode(
        x=alt.X("window:O", title="window"),
        y=alt.Y("PSI", title="worst feature PSI"))
    ruleline = alt.Chart(pd.DataFrame({"y": [0.2]})).mark_rule(
        strokeDash=[4, 4], color=RUST).encode(y="y")
    st.altair_chart(trend + ruleline, use_container_width=True)
    st.subheader("Drill into a feature, latest window")
    feat = st.selectbox("Feature", sorted(last.feature.unique()),
                        key="drift_feature")
    latest_batch = core.latest_drift_batch(X_te, shift=shift)
    frame = core.reference_vs_current(feat, latest_batch, art)
    bars = alt.Chart(frame).mark_bar().encode(
        x=alt.X("bin:N", title=feat), y=alt.Y("share"),
        color=alt.Color("which", scale=alt.Scale(range=[BLUE, RUST])),
        xOffset="which")
    st.altair_chart(bars, use_container_width=True)


def batch_tab(a):
    from batch_score import score_frame
    from validate import load_raw, validate
    art = a["artifact"]
    sample = load_raw(core.ROOT / "data" / "raw" / "telco_churn.csv") \
        .head(25).drop(columns=["Churn"])
    st.download_button("Download a 25 row sample in the raw schema",
                       sample.to_csv(index=False),
                       file_name="sample_customers.csv", key="sample_dl")
    up = st.file_uploader("Raw schema CSV", type="csv", key="uploader")
    if up is not None:
        result = validate(pd.read_csv(up), require_target=False)
        scored = score_frame(result.df, art)
        c = st.columns(3)
        c[0].metric("Rows scored", f"{len(scored):,}")
        c[1].metric("Flag rate",
                    f"{(scored.decision != 'no action').mean():.0%}")
        c[2].metric("Repairs made", f"{len(result.log)}")
        with st.expander("Validator log"):
            for line in result.log:
                st.write("- " + str(line))
        st.dataframe(scored.head(50), use_container_width=True)
        st.download_button("Download scored CSV",
                           scored.to_csv(index=False),
                           file_name="scored.csv", key="scored_dl")


def main():
    st.set_page_config(page_title="Churn service", layout="wide",
                       page_icon=":chart_with_downwards_trend:")
    a = assets()
    with st.sidebar:
        st.header("Filters")
        segs = st.multiselect("Contract segments (empty = all)",
                              sorted(a["test"][0][core.SEGMENT_COL]
                                     .unique()), key="segments")
        st.divider()
        st.caption("Model card")
        st.write(f"{a['artifact']['model_name']} "
                 f"v{a['artifact']['version']}, isotonic calibrated, "
                 f"threshold fitted at "
                 f"{a['artifact']['costs']['fn']:.0f} to 1 costs.")
    header(a)
    tabs = st.tabs(["Overview", "Decision", "Inspect", "Drift", "Batch"])
    with tabs[0]:
        overview_tab(a)
    with tabs[1]:
        decision_tab(a, segs)
    with tabs[2]:
        inspect_tab(a)
    with tabs[3]:
        drift_tab(a)
    with tabs[4]:
        batch_tab(a)


if __name__ == "__main__":
    main()
