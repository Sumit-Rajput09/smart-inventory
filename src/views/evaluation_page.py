"""Model Evaluation: forecasting models, the simulated-label classifier and inventory policy back-testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import app_charts as charts
from src import app_data as ad
from src import app_support as sup
from src import dashboard_metrics as dm
from src import demand_forecasting as dfc
from src import reorder_optimizer as ro
from src.ui_components import Kpi, callout, kpi_row, page_header, section
from src.utils import DataValidationError

ML_MODELS = ["Linear Regression", "Random Forest", "Hist Gradient Boosting"]


@st.cache_data(show_spinner="Comparing models with the baseline...")
def _baseline_comparison(pred_path: str, stamp: float, models: tuple[str, ...], baseline: str = "Series mean") -> pd.DataFrame:
    """Mean difference of absolute error vs the baseline with a 95% paired bootstrap interval (negative = better than baseline)."""
    pred = pd.read_csv(pred_path)
    rows = []
    base_err = (pred[baseline] - pred["units_sold"]).abs().to_numpy()
    for m in models:
        if m in pred.columns:
            d, lo, hi = dfc.paired_bootstrap_diff((pred[m] - pred["units_sold"]).abs().to_numpy(), base_err)
            verdict = "better than the baseline" if hi < 0 else "WORSE than the baseline" if lo > 0 else "not distinguishable from the baseline"
            rows.append({"model": m, "MAE difference vs series mean": d, "95% interval low": lo, "95% interval high": hi, "verdict": verdict})
    return pd.DataFrame(rows)


def render(ctx: ad.Ctx) -> None:
    page_header("Model Evaluation", "How well do the models and policies work, measured on data they were not fitted on?", ["Chronological validation and test", "Synthetic dataset"])
    section_name = st.radio("Evaluation", ["Demand forecasting models", "Stock-out experiment (simulated label)", "Inventory back-test"], horizontal=True, key="ev_section")
    if section_name.startswith("Demand"):
        _forecasting(ctx)
    elif section_name.startswith("Stock-out"):
        _classifier(ctx)
    else:
        _backtest(ctx)


# ================================================================ demand forecasting
def _forecasting(ctx: ad.Ctx) -> None:
    paths = ctx.paths
    horizons = ad.forecast_horizons(paths)
    if not horizons:
        st.error("No trained forecasting models found.")
        st.code("python -m src.demand_forecasting", language="powershell")
        return
    h = st.select_slider("Forecast horizon (days)", options=horizons, value=horizons[0], key="ev_horizon")
    comp_path = paths.reports_dir / f"forecast_model_comparison_h{h}.csv"
    comp = sup.read_csv_optional(comp_path)
    summary = sup.read_text(paths.reports_dir / f"forecast_summary_h{h}.txt")
    if comp is None:
        st.warning("Comparison file missing. Re-run:  `python -m src.demand_forecasting`")
        return
    _, meta = ad.load_model_cached(h, str(paths.models_dir), sup.mtime(dfc.model_paths(h, paths.models_dir)[0]))
    callout("info", "All models are scored on the **same chronological test period**. Validation scores (models fitted on the training period) choose the model; "
                    "test scores (refitted on training + validation) are the final check. Rows with *valid = False* are leakage demonstrations, not usable models.")
    valid = comp[comp["valid"].fillna(False).astype(bool)].copy()
    sel = meta["model_name"]
    base = valid[valid["model"] == "Series mean"]
    tm = meta["test_metrics"]

    # ------------------------------------------------------------ selected model + baseline
    gain = None
    if len(base):
        b = float(base["mae_test"].iloc[0])
        gain = 100 * (b - tm["mae"]) / b
    kpi_row([Kpi("Selected model", sel, meta["selection_rule"], "purple"), Kpi("Test MAE", f"{tm['mae']:.2f}", "Units per store-day", "blue"), Kpi("Test RMSE", f"{tm['rmse']:.2f}", "Penalises large errors", "blue"),
             Kpi("Test WAPE", f"{100 * tm['wape']:.1f}%", "Sum of absolute errors / sum of actual", "blue"), Kpi("Test R2", f"{tm['r2']:.3f}", "0 = no better than predicting the mean", "blue"),
             Kpi("MAE vs baseline", f"{gain:+.2f}%" if gain is not None else "n/a", "Positive = better than the series-mean baseline", "grey")])
    if ML_MODELS[2] not in set(comp["model"]):
        st.caption("This comparison was trained before Hist Gradient Boosting was added. Re-run `python -m src.demand_forecasting` to include it.")

    # ------------------------------------------------------------ comparison table and charts
    section("Model comparison", "WAPE = sum of absolute errors / sum of actual units (safe when some days sell zero). Percentage errors per row are not used because zero-demand rows would divide by zero.")
    wanted = ["model", "mae_val", "mae_test", "rmse_test", "wape_test", "r2_test", "bias_test"] + (["mae_7d_total_test", "wape_7d_total_test"] if h >= 7 else []) + ["valid"]
    table = comp[[c for c in wanted if c in comp.columns]].copy()
    if len(base):
        table.insert(3, "mae_vs_baseline_%", 100 * (float(base["mae_test"].iloc[0]) - table["mae_test"]) / float(base["mae_test"].iloc[0]))
    st.dataframe(table, use_container_width=True, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c not in ("model", "valid")})
    m = valid.rename(columns={"mae_test": "mae", "rmse_test": "rmse", "wape_test": "wape", "r2_test": "r2"})
    c1, c2 = st.columns(2)
    c1.altair_chart(charts.test_metric_bars(m, "mae"), use_container_width=True)
    c2.altair_chart(charts.test_metric_bars(m, "rmse"), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.altair_chart(charts.test_metric_bars(m, "wape"), use_container_width=True)
    c4.altair_chart(charts.test_metric_bars(m, "r2"), use_container_width=True)
    st.altair_chart(charts.model_comparison_chart(valid, "mae"), use_container_width=True)

    # ------------------------------------------------------------ baseline comparison with intervals
    pred_path = paths.reports_dir / f"forecast_test_predictions_h{h}.csv"
    pred = None
    if pred_path.exists():
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        bc = _baseline_comparison(str(pred_path), sup.mtime(pred_path), tuple(m_ for m_ in ML_MODELS if m_ in pred.columns))
        if len(bc):
            section("Baseline comparison", "Is any machine-learning model really better than predicting each series' average? Paired bootstrap on the test rows (approximate: rows of one series are not fully independent).")
            st.dataframe(bc, use_container_width=True, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in bc.columns if c not in ("model", "verdict")})
            best = bc.sort_values("MAE difference vs series mean").iloc[0]
            if best["95% interval high"] >= 0:
                callout("warn", f"**No machine-learning model is clearly better than the series-mean baseline.** The best, {best['model']}, differs by {best['MAE difference vs series mean']:+.3f} MAE units "
                                f"(95% interval {best['95% interval low']:+.3f} to {best['95% interval high']:+.3f}). The simpler model is preferred. This is a property of the synthetic data (daily sales are close to random noise), not a bug.")
            else:
                callout("ok", f"**{best['model']} is better than the baseline** on this test period (95% interval {best['95% interval low']:+.3f} to {best['95% interval high']:+.3f} MAE units).")

        # -------------------------------------------------------- actual vs predicted and residuals
        section("Actual versus predicted", f"Test period, totals per day over all store-products. Selected model: {sel}.")
        cols = [c for c in dict.fromkeys([sel, "Series mean"]) if c in pred.columns]
        daily = pred.groupby("date")[["units_sold", *cols]].sum().reset_index()
        st.altair_chart(charts.actual_vs_predicted_chart(daily, cols), use_container_width=True)
        section("Residual analysis", "Residual = predicted - actual. A good model has residuals centred on 0 with no pattern against the predicted value.")
        resid = pred[sel] - pred["units_sold"]
        kpi_row([Kpi("Mean residual (bias)", f"{resid.mean():+.2f}", "Units; 0 = unbiased", "purple"), Kpi("Residual std", f"{resid.std():.2f}", "Units", "purple"),
                 Kpi("Share over-forecast", f"{(resid > 0).mean():.0%}", "Predicted above actual", "purple"), Kpi("Residuals analysed", f"{len(resid):,}", "Test rows", "grey")])
        r1, r2 = st.columns(2)
        r1.altair_chart(charts.residual_hist(resid), use_container_width=True)
        r2.altair_chart(charts.residual_scatter(pred[sel], resid), use_container_width=True)
    else:
        st.info("Saved test predictions are missing, so the baseline comparison, actual-vs-predicted and residual charts cannot be drawn. Re-run `python -m src.demand_forecasting`.")

    # ------------------------------------------------------------ interpretation and details
    if summary and "INTERPRETATION" in summary:
        body = summary.split("INTERPRETATION (generated from the numbers above)")[1].split("LIMITATION")[0].strip()
        st.markdown("**Interpretation (generated from the numbers)**")
        st.markdown(body)
    with st.expander("Selected model information"):
        st.json({k: meta[k] for k in ("model_name", "model_family", "task", "selection_rule", "fitted_on", "periods", "rf_best_params", "created_at") if k in meta})
        st.caption("Features: " + ", ".join(meta.get("feature_columns", [])))
    fi = paths.figures_dir / f"forecast_feature_importance_h{h}.png"
    if fi.exists():
        with st.expander("Random Forest feature importance"):
            st.image(str(fi))
    if summary:
        with st.expander("Full text summary"):
            st.code(summary, language="text")
    st.caption("Feature engineering already includes lags (1, 7, 14 days), rolling means and standard deviations (7, 14, 28 days) built only from days before the forecast origin, "
               "calendar fields and price/promotion fields. Leaky columns (Demand Forecast, same-day Units Ordered and inventory) are excluded.")


# ================================================================ classifier
def _classifier(ctx: ad.Ctx) -> None:
    paths = ctx.paths
    callout("danger", "**The label is simulated.** The dataset has no real stock-out records. Treat these numbers as a demonstration of the method, not as real stock-out performance.")
    available = ad.available_classifier_lead_times(paths)
    if not available:
        st.info("Nothing to show yet. Run:  `python -m src.stockout_prediction`")
        return
    L = available[0] if len(available) == 1 else st.selectbox("Lead time (days)", available, key="ev_L")
    _, meta = ad.load_classifier_cached(L, str(paths.models_dir), sup.mtime(paths.models_dir / f"stockout_classifier_simulated_L{L}.joblib"))
    comp = sup.read_csv_optional(paths.reports_dir / f"stockout_classifier_simulated_L{L}_comparison.csv")
    text = sup.read_text(paths.reports_dir / f"stockout_classifier_simulated_L{L}_summary.txt")
    best = meta["model_name"]
    if comp is not None:
        t = comp.set_index("model")
        sel = t.loc[best]
        kpi_row([Kpi("Selected model", best, "Highest validation PR-AUC", "purple"), Kpi("ROC-AUC", f"{sel['roc_auc_test']:.3f}", "0.5 = chance", "purple"), Kpi("Precision", f"{sel['precision_test']:.2f}", "Test, at the operating threshold", "purple"),
                 Kpi("Recall", f"{sel['recall_test']:.2f}", "Test, at the operating threshold", "purple"), Kpi("F1-score", f"{sel['f1_test']:.2f}", "Test", "purple"),
                 Kpi("Operating threshold", f"{sel['threshold_test']:.3f}", "Chosen on validation data", "grey")])
        st.dataframe(comp[["model"] + [c for c in comp.columns if c.endswith("_test")]].rename(columns=lambda c: c.replace("_test", "")), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.altair_chart(charts.confusion_heatmap(int(sel["tn_test"]), int(sel["fp_test"]), int(sel["fn_test"]), int(sel["tp_test"]), f"{best} (test)"), use_container_width=True)
        rule = t.loc["Rule-based reorder-point trigger"] if "Rule-based reorder-point trigger" in t.index else None
        if rule is not None:
            c2.altair_chart(charts.confusion_heatmap(int(rule["tn_test"]), int(rule["fp_test"]), int(rule["fn_test"]), int(rule["tp_test"]), "Rule-based reorder-point trigger (test)"), use_container_width=True)
    path = paths.reports_dir / f"stockout_test_predictions_simulated_L{L}.csv"
    if path.exists():
        try:
            cur = dm.roc_pr_curves(pd.read_csv(path), ["Logistic Regression", "Random Forest", "Rule-based probability"])
            d1, d2 = st.columns(2)
            d1.altair_chart(charts.roc_chart(cur["roc"]), use_container_width=True)
            d2.altair_chart(charts.pr_chart(cur["pr"], cur["prevalence"]), use_container_width=True)
            st.caption(f"Curves use the saved test-period scores. The simulated stock-out rate (precision of a 'flag everything' rule) is {cur['prevalence']:.1%}. "
                       "A model is only useful if it keeps recall high while lifting precision above that line, and it must also beat the simple rule-based formula.")
        except DataValidationError as exc:
            st.info(str(exc))
    else:
        st.info("ROC and precision-recall curves need the saved test predictions. Re-run `python -m src.stockout_prediction` to create them.")
    fig = paths.figures_dir / f"stockout_classifier_simulated_L{L}.png"
    if fig.exists():
        with st.expander("Original evaluation figure from the training run"):
            st.image(str(fig))
    if text:
        with st.expander("Full text summary"):
            st.code(text, language="text")


# ================================================================ back-test
def _backtest(ctx: ad.Ctx) -> None:
    params, err = ad.build_params()
    if err:
        st.error(err)
        return
    callout("info", "Does the replenishment policy deliver what it promises on data it did not see? Parameters are estimated before the last 90 days, then the policy is checked inside those 90 days. "
                    "Change the lead time and service level in the sidebar.")
    try:
        bt = ad.backtest_cached(ctx.path, ctx.stamp, params)
    except (DataValidationError, ValueError) as exc:
        st.warning(str(exc))
        return
    gap = bt["achieved_service_level"] - bt["target_service_level"]
    section("Service level check", "Share of lead-time windows in which demand stayed within the reorder point (cycle service level).")
    kpi_row([Kpi("Target service level", sup.fmt_pct(bt["target_service_level"]), "Your setting", "blue"), Kpi("Achieved on unseen data", sup.fmt_pct(bt["achieved_service_level"]), f"{bt['windows']:,} windows", "blue"),
             Kpi("Difference", f"{100 * gap:+.1f} points", "Achieved minus target", "green" if abs(gap) <= 0.01 else "amber"),
             Kpi("Stock-out frequency (windows)", sup.fmt_pct(1 - bt["achieved_service_level"]), "Windows where demand exceeded the reorder point", "red"),
             Kpi("Per-series range", f"{sup.fmt_pct(bt['series_min'], 0)} - {sup.fmt_pct(bt['series_max'], 0)}", "Lowest to highest series", "grey")])
    st.caption(f"Lead time used: {bt['lead_time_days_used']} day(s) (whole days). Test period {bt['test_period'][0]} to {bt['test_period'][1]}.")
    st.info("A shortfall means the normal-demand assumption slightly overstates protection (sales are right-skewed)." if gap < -0.01 else "The formula delivers about the promised protection.")

    section("Inventory policy simulation", "The (s, S) policy is run day by day over the last 90 days for several service-level targets.")
    try:
        sim = ad.policy_backtest_cached(ctx.path, ctx.stamp, params)
    except (DataValidationError, ValueError) as exc:
        st.warning(str(exc))
        return
    ch = sim["chosen"]
    kpi_row([Kpi("Fill rate", sup.fmt_pct(ch["fill_rate"]), "Share of demand served from stock", "green"), Kpi("Stock-out frequency", sup.fmt_pct(ch["stockout_day_frequency"]), "Store-product days with unmet demand", "red"),
             Kpi("Average inventory", f"{ch['avg_on_hand_units']:,.0f} units", "Per store-product, end of day", "blue"), Kpi("Orders per series", f"{ch['orders_per_series']:.1f}", "In the 90 days", "grey"),
             Kpi("Units lost", f"{ch['units_lost']:,.0f}", f"of {ch['units_demanded']:,.0f} demanded", "amber")])
    st.caption(f"Shown for the target closest to your service level ({ch['service_level_target']:.1%}) and lead time {sim['lead_time_days_used']} day(s).")
    a, b = st.columns([3, 2])
    a.altair_chart(charts.policy_tradeoff_chart(sim["table"]), use_container_width=True)
    t = sim["table"].rename(columns={"service_level_target": "target", "fill_rate": "fill rate", "stockout_day_frequency": "stock-out freq.", "avg_on_hand_units": "avg stock", "orders_per_series": "orders/series"})
    b.dataframe(t[["target", "fill rate", "stock-out freq.", "avg stock", "orders/series"]], use_container_width=True, hide_index=True,
                column_config={"target": st.column_config.NumberColumn(format="%.2f"), "fill rate": st.column_config.NumberColumn(format="%.3f"), "stock-out freq.": st.column_config.NumberColumn(format="%.3f"),
                               "avg stock": st.column_config.NumberColumn(format="%.0f"), "orders/series": st.column_config.NumberColumn(format="%.1f")})
    with st.expander("Back-testing assumptions (read before interpreting)", expanded=True):
        for a_ in sim["assumptions"]:
            st.markdown(f"- {a_}")
        st.markdown("- **Historical back-testing on one synthetic file cannot prove real-world performance.**")
