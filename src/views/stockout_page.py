"""Stock-Out Risk: rule-based risk estimation and (clearly separated) the simulated-label classifier experiment."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import app_charts as charts
from src import app_data as ad
from src import app_support as sup
from src import dashboard_metrics as dm
from src import reorder_optimizer as ro
from src import stockout_prediction as sp
from src.preprocess import GROUP_KEYS
from src.ui_components import Kpi, callout, kpi_row, page_header, section, show_table
from src.ui_theme import RISK_TONE
from src.utils import DataFileError, DataValidationError

SIMULATED_ASSUMPTIONS = [
    "The dataset has NO record of real stock-outs, so a label was invented: on day t the stock on hand is compared with the recorded sales of days t to t+L-1 (L = lead time).",
    "Label = 1 when those recorded sales add up to MORE than today's stock. It assumes no delivery arrives during the window.",
    "It uses recorded sales as demand. Real demand can be higher than recorded sales when stock runs out, so the label is only a proxy.",
    "In this synthetic file each day's sales look like a random share of that day's stock, so the label mostly reflects how the file was generated, not retail behaviour.",
    "Features use only history up to the day before the decision plus the stock on hand; train, validation and test are consecutive periods and label windows that cross a boundary are purged.",
    "The operating threshold is chosen on validation data to reach a target recall; scores are NOT calibrated probabilities (see the calibration check below).",
]


def render(ctx: ad.Ctx) -> None:
    page_header("Stock-Out Risk", "Two separate approaches. Do not mix them up: one is a formula, the other is an experiment on a hypothetical label.",
                ["Rule-based = formula estimate", "Classifier = simulated-label experiment", "Synthetic dataset"])
    section_choice = st.radio("Approach", ["Rule-based risk assessment", "Simulated-label classifier (experiment)"], horizontal=True, key="so_section")
    if section_choice.startswith("Rule"):
        _rule_based(ctx)
    else:
        _classifier(ctx)


# ================================================================ rule-based
def _rule_based(ctx: ad.Ctx) -> None:
    callout("info", "**Rule-based estimate.** A formula, not a trained model and not an observed outcome. Change the settings under *Inventory settings* in the sidebar. "
                    "Stock is read at the snapshot date; demand is estimated from the days before it (the sidebar date range does not apply here).")
    st.latex(r"\text{Lead-time demand} = d \times L \qquad \text{Safety stock} = z\,\sigma\sqrt{L} \qquad \text{Reorder point} = d \times L + z\,\sigma\sqrt{L}")
    st.caption("d = average daily demand, sigma = standard deviation of daily demand (both from the days before the snapshot), L = lead time, z = normal quantile of the service level.")
    recs, params = ad.filtered_recommendations(ctx)
    if recs is None:
        return
    k = ro.summarize(recs)
    kpi_row([Kpi(lvl, str(k["risk_counts"][lvl]), {"Low": "Service target met", "Medium": "Above 1 - service level, below 20%", "High": "20% to below 50%", "Critical": "50% or more"}[lvl], RISK_TONE[lvl]) for lvl in ("Low", "Medium", "High", "Critical")]
            + [Kpi("Mean stock-out probability", sup.fmt_pct(k["mean_stockout_probability"]), "Chance demand in the lead time exceeds the stock", "blue")])
    if k["share_reorder"] > 0.9:
        callout("warn", f"**{k['share_reorder']:.0%} of the items are below their reorder point.** Stock on hand covers only about {k['median_days_of_cover']:.1f} days of demand in this dataset, "
                        "so a long lead time makes every item look urgent. This reflects the synthetic data; try a shorter lead time in the sidebar.")
    with st.expander("How the risk levels are defined"):
        st.text(ro.RISK_RULES_TEXT)
    c1, c2 = st.columns(2)
    c1.altair_chart(charts.risk_donut(recs), use_container_width=True)
    c2.altair_chart(charts.position_vs_rop_chart(recs), use_container_width=True)
    levels = st.multiselect("Show risk levels", [*sp.RISK_LEVELS, "Unknown"], default=[*sp.RISK_LEVELS, "Unknown"], key="so_levels")
    shown = recs[recs["risk_level"].astype(str).isin(levels)]
    cols = [*GROUP_KEYS, "category", "current_inventory", "avg_daily_demand", "lead_time_demand", "safety_stock", "reorder_point", "projected_stock_after_lead_time",
            "stockout_probability", "risk_level", "stock_status", "recommendation"]
    show_table(shown[[c for c in cols if c in shown.columns]].assign(risk_level=shown["risk_level"].astype(str)), "so_rule", file_name="stockout_risk_rule_based.csv", height=440,
               column_config={"stockout_probability": st.column_config.NumberColumn("stock-out probability", format="%.3f"),
                              "projected_stock_after_lead_time": st.column_config.NumberColumn("projected inventory", format="%.0f"),
                              **{c: st.column_config.NumberColumn(c, format="%.0f") for c in ("avg_daily_demand", "lead_time_demand", "safety_stock", "reorder_point", "current_inventory")}})
    st.caption("Projected inventory = inventory position minus expected lead-time demand. Estimates only: demand is assumed normal with independent days and a fixed lead time; "
               "real sales here are right-skewed, so the formula is somewhat optimistic (see the back-test in Model Evaluation).")


# ================================================================ simulated classifier
def _classifier(ctx: ad.Ctx) -> None:
    paths = ctx.paths
    callout("danger", "**SIMULATED EXPERIMENT.** The dataset contains no real stock-out records. The label is hypothetical: the recorded sales of the next L days exceed today's stock, "
                      "assuming no delivery. These results are not real stock-out history, the classifier does not predict actual historical stock-outs, and the rule-based assessment is a formula, not this classifier.")
    available = ad.available_classifier_lead_times(paths)
    if not available:
        st.info("No classifier trained yet. Run:  `python -m src.stockout_prediction`")
        return
    L = st.selectbox("Label window / lead time (days)", available, key="clf_L") if len(available) > 1 else available[0]
    mpath = sp.classifier_paths(L, paths.models_dir)[0]
    _, meta = ad.load_classifier_cached(L, str(paths.models_dir), sup.mtime(mpath))
    tm = meta["test_metrics"]
    kpi_row([Kpi("Model", meta["model_name"], "Best ML model on validation PR-AUC", "purple"), Kpi("Recall (simulated stock-out)", f"{tm['recall']:.2f}", "Share of simulated stock-outs caught", "purple"),
             Kpi("Precision", f"{tm['precision']:.2f}", "Share of alerts that were simulated stock-outs", "purple"), Kpi("F1-score", f"{tm['f1']:.2f}", "Harmonic mean of precision and recall", "purple"),
             Kpi("ROC-AUC", f"{tm['roc_auc']:.3f}", "0.5 = no ranking ability", "purple"), Kpi("Simulated stock-out rate (test)", sup.fmt_pct(meta["label_prevalence_test"]), "Share of rows with label = 1", "grey")])
    st.markdown(f"**Label:** {meta['label_definition']}  \n**Decision moment:** {meta['decision_moment']}  \n"
                f"**Operating threshold:** {meta['threshold']:.3f} (highest threshold that still caught at least {meta['target_recall_on_validation']:.0%} of simulated stock-outs on validation data).")
    with st.expander("Assumptions behind the simulated label and limitations", expanded=False):
        for a in SIMULATED_ASSUMPTIONS:
            st.markdown(f"- {a}")
        st.markdown(f"- Warning stored with the model: *{meta.get('warning', 'Synthetic dataset AND simulated label.')}*")

    comp = sup.read_csv_optional(paths.reports_dir / f"stockout_classifier_simulated_L{L}_comparison.csv")
    if comp is not None:
        test_cols = ["model"] + [c for c in comp.columns if c.endswith("_test") and c.split("_test")[0] in ("threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp")]
        st.markdown("**Test-period comparison with the rule-based formula (same simulated label)**")
        st.dataframe(comp[test_cols].rename(columns=lambda c: c.replace("_test", "")), use_container_width=True, hide_index=True)
    _calibration(paths, L, ["Logistic Regression", "Random Forest", "Rule-based probability"])

    section("Latest snapshot scored by the classifier (simulated)")
    try:
        scores = ad.classifier_scores_cached(L, str(paths.models_dir), sup.mtime(mpath), ctx.path, ctx.stamp, None)
    except (DataValidationError, DataFileError) as exc:
        st.warning(str(exc))
        return
    for field, col in (("stores", "store_id"), ("products", "product_id")):
        if ctx.filters[field] is not None:
            scores = scores[scores[col].astype(str).isin(ctx.filters[field])]
    if scores.empty:
        st.warning("No store-products match the sidebar filters.")
        return
    st.caption(f"Scored at the last date in the data ({scores['snapshot_date'].iloc[0]:%d %b %Y}); the sidebar snapshot date and the category/region filters do not apply to the classifier. "
               "The score is a model output for the hypothetical label, not the probability of a real stock-out.")
    show_table(scores.drop(columns=["snapshot_date"]), "so_clf", file_name="stockout_classifier_simulated_scores.csv",
               column_config={"simulated_stockout_probability": st.column_config.NumberColumn("simulated-label score", format="%.3f")})


def _calibration(paths: sup.AppPaths, L: int, models: list[str]) -> None:
    """Reliability check from the saved test-period scores: do the scores behave like probabilities of the simulated label?"""
    path = paths.reports_dir / f"stockout_test_predictions_simulated_L{L}.csv"
    if not path.exists():
        st.info("Calibration and curves need the saved test predictions. Re-run `python -m src.stockout_prediction` to create them.")
        return
    try:
        pred = ad.read_csv_cached(str(path), sup.mtime(path))
        cur = dm.roc_pr_curves(pred, models)
    except DataValidationError as exc:
        st.info(str(exc))
        return
    with st.expander("Is the score a calibrated probability? (calibration check on the test period)"):
        const = cur["prevalence"] * (1 - cur["prevalence"])
        st.altair_chart(charts.calibration_chart(cur["calibration"]), use_container_width=True)
        st.dataframe(pd.DataFrame({"model": list(cur["brier"]), "Brier score (lower is better)": list(cur["brier"].values()), "Brier of a constant prevalence guess": const}), use_container_width=True, hide_index=True)
        st.caption("If the curve is far from the dashed diagonal, the score ranks risk but is not a probability. Models trained with balanced class weights typically over-state the rate. "
                   "Calibrating (for example isotonic or Platt scaling) would need its own calibration period and still would not make the label real, so the scores are shown as scores.")
