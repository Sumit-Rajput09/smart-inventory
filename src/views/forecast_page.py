"""Demand Forecasting: controls, actual-vs-forecast charts, empirical prediction intervals, summary cards, tables and reports."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import app_charts as charts
from src import app_data as ad
from src import app_support as sup
from src import dashboard_metrics as dm
from src import demand_forecasting as dfc
from src import report_generator as rg
from src.preprocess import GROUP_KEYS
from src.ui_components import Kpi, callout, kpi_row, page_header, section, show_table
from src.utils import DataFileError, DataValidationError

COVERAGES = {"None": None, "80%": 0.80, "90%": 0.90, "95%": 0.95}


def _with_attributes(fc: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Category and region recorded on the last observed day for each store-product (they vary by row in this dataset)."""
    last = cleaned["date"].max()
    cols = [c for c in ("category", "region") if c in cleaned.columns]
    if not cols:
        return fc
    attrs = cleaned.loc[cleaned["date"] == last, [*GROUP_KEYS, *cols]].drop_duplicates(GROUP_KEYS).astype({"store_id": str, "product_id": str})
    out = fc.assign(store_id=fc["store_id"].astype(str), product_id=fc["product_id"].astype(str)).merge(attrs, on=GROUP_KEYS, how="left")
    for c in cols:
        out[c] = out[c].astype(str)
    return out


def _series_intervals(pred: pd.DataFrame, model: str, fc: pd.DataFrame, coverage: float) -> pd.DataFrame:
    """Per store-product interval bounds from that series' own held-out residuals (NaN when too few)."""
    p = pred.assign(store_id=pred["store_id"].astype(str), product_id=pred["product_id"].astype(str))
    rows = []
    for (s, pid), g in p.groupby(GROUP_KEYS):
        try:
            iv = dfc.empirical_interval(g["units_sold"] - g[model], coverage)
        except DataValidationError:
            continue
        rows.append({"store_id": s, "product_id": pid, "lo": iv["lower_offset"], "hi": iv["upper_offset"]})
    m = fc.merge(pd.DataFrame(rows), on=GROUP_KEYS, how="left") if rows else fc.assign(lo=float("nan"), hi=float("nan"))
    m[f"lower_{int(coverage * 100)}"] = (m["forecast_units"] + m["lo"]).clip(lower=0)
    m[f"upper_{int(coverage * 100)}"] = (m["forecast_units"] + m["hi"]).clip(lower=0)
    return m.drop(columns=["lo", "hi"])


def render(ctx: ad.Ctx) -> None:
    paths = ctx.paths
    page_header("Demand Forecasting", "Forecast units sold for the day (last observed date + horizon), using only information up to the last observed date.",
                ["Models are evaluated on a chronological test period", "Synthetic dataset"])
    horizons = ad.forecast_horizons(paths)
    if not horizons:
        st.error("No trained forecasting model found.")
        st.code("python -m src.demand_forecasting", language="powershell")
        return

    # ------------------------------------------------------------ controls
    st.markdown("**Forecast controls**")
    c0, c1 = st.columns([2, 1])
    h = c0.select_slider("Forecast horizon (days ahead)", options=horizons, value=horizons[0], key="fc_horizon",
                         help="Models are trained per horizon: " + ", ".join(map(str, horizons)) + " days. Train others with --horizons.")
    cov_label = c1.selectbox("Prediction interval", list(COVERAGES), index=2, key="fc_cov", help="Empirical interval from the model's errors on the held-out test period.")
    coverage = COVERAGES[cov_label]
    mpath = dfc.model_paths(h, paths.models_dir)[0]
    try:
        model, meta = ad.load_model_cached(h, str(paths.models_dir), sup.mtime(mpath))
        fc_all = ad.forecast_cached(h, str(paths.models_dir), sup.mtime(mpath), ctx.path, ctx.stamp)
    except (DataValidationError, DataFileError) as exc:
        st.error(str(exc))
        return
    fc_all = _with_attributes(fc_all, ctx.cleaned)
    try:  # sidebar filters first
        fc_base = sup.filter_recommendations(fc_all, ctx.filters["stores"], ctx.filters["products"], ctx.filters["categories"], ctx.filters["regions"])
    except DataValidationError as exc:
        st.warning(str(exc) + " Widen the sidebar filters.")
        return
    r = st.columns(3)
    stores = r[0].multiselect("Stores", sorted(fc_base["store_id"].unique()), default=sorted(fc_base["store_id"].unique()), key="fc_stores")
    products = r[1].multiselect("Products", sorted(fc_base["product_id"].unique()), default=sorted(fc_base["product_id"].unique()), key="fc_products")
    cats = r[2].multiselect("Categories (on the last observed day)", sorted(fc_base["category"].unique()), default=sorted(fc_base["category"].unique()), key="fc_cats")
    try:
        fc = sup.filter_recommendations(fc_base, stores, products, cats, None)
    except DataValidationError:
        st.warning("No store-products match these forecast controls.")
        return

    # ------------------------------------------------------------ held-out evidence
    pred_path = paths.reports_dir / f"forecast_test_predictions_h{h}.csv"
    comp_path = paths.reports_dir / f"forecast_model_comparison_h{h}.csv"
    pred = ad.read_csv_cached(str(pred_path), sup.mtime(pred_path)) if pred_path.exists() else None
    if pred is not None:
        pred["date"] = pd.to_datetime(pred["date"])
    comp = ad.read_csv_cached(str(comp_path), sup.mtime(comp_path)) if comp_path.exists() else None
    base_mae = float(comp.loc[comp["model"] == "Series mean", "mae_test"].iloc[0]) if comp is not None and "Series mean" in set(comp["model"]) else None
    tm = meta["test_metrics"]
    total = float(fc["forecast_units"].sum())
    target = fc["target_date"].iloc[0]
    interval, total_lo, total_hi = None, None, None
    if coverage is not None and pred is not None and meta["model_name"] in pred.columns:
        try:
            res = dm.selection_residuals(pred, meta["model_name"], pairs=fc[GROUP_KEYS])
            interval = dfc.empirical_interval(res, coverage)
            total_lo, total_hi = dfc.apply_interval(total, interval)
        except DataValidationError as exc:
            st.caption(f"Prediction interval not shown: {exc}")
    elif coverage is not None:
        st.caption("Prediction interval not shown: the saved test predictions are missing. Re-run `python -m src.demand_forecasting`.")

    # ------------------------------------------------------------ summary cards
    kpi_row([Kpi("Forecast model", meta["model_name"], meta["selection_rule"], "purple"), Kpi("Forecast date", f"{target:%d %b %Y}", f"{h} day(s) after {ctx.cleaned['date'].max():%d %b %Y}", "purple"),
             Kpi("Total forecast", f"{total:,.0f} units", f"{len(fc)} store-products selected", "purple"),
             Kpi(f"{cov_label} interval (total)" if interval else "Prediction interval", f"{total_lo:,.0f} to {total_hi:,.0f}" if interval else "not shown", "Empirical, from held-out test errors" if interval else "Choose a level above", "purple"),
             Kpi("Test MAE", f"{tm['mae']:.1f}", "Units per store-day" + (f"; series-mean baseline {base_mae:.1f}" if base_mae is not None else ""), "blue"),
             Kpi("Test R2", f"{tm['r2']:.3f}", "Near 0: almost no explained variation", "blue")])

    # ------------------------------------------------------------ chart
    section("Actual demand and forecast", "Blue: actual units sold (sum of the selected store-products). Purple diamond: the forecast for the target date; the bar is the prediction interval.")
    days = st.slider("History shown (days)", 30, 365, 120, step=10, key="fc_days")
    try:
        hist = dm.forecast_history(ctx.cleaned, days=days, pairs=fc[GROUP_KEYS])
        st.altair_chart(charts.forecast_vs_actual_chart(hist, target, total, total_lo, total_hi, "Units sold per day (selection total)"), use_container_width=True)
    except DataValidationError as exc:
        st.info(str(exc))
    if interval:
        cov = dm.interval_coverage(res, interval)
        st.caption(f"Interval: forecast + the {interval['lower_offset']:+,.0f} / {interval['upper_offset']:+,.0f} unit empirical quantiles of actual - predicted daily totals over {interval['n_residuals']} held-out test days "
                   f"(coverage on those days {cov:.0%}). It assumes future errors resemble these errors; it says nothing about structural change.")
    callout("info", "Forecasts are similar across items because products barely differ in this dataset and the best model is a simple average. "
                    "Price, discount and promotion of the target day are assumed equal to the latest observed values.")

    # ------------------------------------------------------------ one series
    section("One store-product in detail")
    d1, d2 = st.columns(2)
    s = d1.selectbox("Store", sorted(fc["store_id"].unique()), key="fc_store")
    p = d2.selectbox("Product", sorted(fc.loc[fc["store_id"] == s, "product_id"].unique()), key="fc_product")
    row = fc[(fc["store_id"] == s) & (fc["product_id"] == p)].iloc[0]
    one = pd.DataFrame({"store_id": [s], "product_id": [p]})
    lo1 = hi1 = None
    if coverage is not None and pred is not None and meta["model_name"] in pred.columns:
        try:
            iv1 = dfc.empirical_interval(dm.selection_residuals(pred, meta["model_name"], aggregate=False, pairs=one), coverage)
            lo1, hi1 = dfc.apply_interval(float(row["forecast_units"]), iv1)
        except DataValidationError:
            pass
    hist1 = dm.forecast_history(ctx.cleaned, days=days, pairs=one)
    st.altair_chart(charts.forecast_vs_actual_chart(hist1, row["target_date"], float(row["forecast_units"]), lo1, hi1, f"Units sold per day ({s} / {p})"), use_container_width=True)
    st.caption(f"Forecast {row['forecast_units']:.1f} units on {row['target_date']:%d %b %Y} ({row['category']}, {row['region']} on the last observed day)."
               + (f" {cov_label} interval {lo1:.0f} to {hi1:.0f}: a single store-day is very uncertain." if lo1 is not None else ""))

    # ------------------------------------------------------------ table and downloads
    section("Forecast results")
    table = fc.drop(columns=["assumption"]).assign(target_date=fc["target_date"].dt.strftime("%Y-%m-%d"))
    if coverage is not None and pred is not None and meta["model_name"] in pred.columns:
        table = _series_intervals(pred, meta["model_name"], table, coverage)
    lo_f, hi_f = float(table["forecast_units"].min()), float(table["forecast_units"].max())
    if hi_f > lo_f:
        a, b = st.slider("Show forecasts between (units)", lo_f, hi_f, (lo_f, hi_f), key="fc_range")
        table = table[table["forecast_units"].between(a, b)]
    cfg = {c: st.column_config.NumberColumn(c, format="%.1f") for c in table.columns if c.startswith(("forecast_units", "lower_", "upper_"))}
    shown = show_table(table, "forecast", file_name=f"demand_forecast_h{h}.csv", column_config=cfg)
    cA, cB = st.columns([2, 1])
    cA.altair_chart(charts.forecast_bar_chart(fc), use_container_width=True)
    info = {"horizon": h, "model": meta["model_name"], "selection_rule": meta["selection_rule"], "target_date": f"{target:%Y-%m-%d}", "last_observed": f"{ctx.cleaned['date'].max():%Y-%m-%d}",
            "selection": f"{len(stores)} store(s), {len(products)} product(s), {len(cats)} categor{'y' if len(cats) == 1 else 'ies'}", "n_series": len(fc), "total_forecast": total, "mean_forecast": total / len(fc),
            "interval": ({"coverage": interval["coverage"], "lower": total_lo, "upper": total_hi, "n_residuals": interval["n_residuals"]} if interval else None),
            "metrics": tm, "baseline_mae": base_mae}
    cB.markdown("**Forecast report**")
    cB.caption("A Markdown report with the model, date, selection, accuracy and assumptions.")
    cB.download_button("Download forecast report (.md)", rg.forecast_report_markdown(info, table if len(shown) == 0 else shown), f"forecast_report_h{h}.md", "text/markdown", key="fc_report_dl", use_container_width=True)

    # ------------------------------------------------------------ comparison
    if comp is not None:
        with st.expander("How does this model compare with the others? (test period)"):
            valid = comp[comp["valid"].fillna(False).astype(bool)] if "valid" in comp.columns else comp
            m1, m2 = st.columns(2)
            m1.altair_chart(charts.test_metric_bars(valid.rename(columns={"mae_test": "mae"}), "mae"), use_container_width=True)
            m2.altair_chart(charts.test_metric_bars(valid.rename(columns={"wape_test": "wape"}), "wape"), use_container_width=True)
            st.caption("Same test observations for every model. If a complex model does not clearly beat the series-mean baseline, the simpler model is preferred. Details on the Model Evaluation page.")
