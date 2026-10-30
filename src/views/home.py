"""Home / Executive Summary: KPIs, inventory health, trends, insights and the most urgent items."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import app_charts as charts
from src import app_data as ad
from src import app_support as sup
from src import dashboard_metrics as dm
from src import demand_forecasting as dfc
from src import reorder_optimizer as ro
from src.ui_theme import BLUE, GREY, PURPLE
from src.ui_components import Kpi, callout, insight_cards, kpi_row, page_header, section
from src.utils import DataValidationError

FREQ = {"Weekly": "W", "Daily": "D", "Monthly": "M"}


def _go_to_reorder() -> None:
    st.session_state["nav_page"] = "Reorder Recommendations"


def render(ctx: ad.Ctx) -> None:
    view, paths = ctx.view, ctx.paths
    recs, params = ad.filtered_recommendations(ctx)
    snap = recs["snapshot_date"].iloc[0] if recs is not None and len(recs) else None

    chips = [f"{len(ctx.cleaned):,} records in the file", f"{ctx.cleaned['date'].min():%d %b %Y} to {ctx.cleaned['date'].max():%d %b %Y}",
             f"{ctx.cleaned['store_id'].nunique()} stores, {ctx.cleaned['product_id'].nunique()} products", "Synthetic dataset", *ad.filter_summary(ctx)]
    page_header("Smart Inventory Analytics", "Command center for inventory health, demand forecasts, stock-out risk and replenishment. "
                "Everything shown is computed from the selected data.", chips)
    callout("warn", "**Synthetic dataset.** The Kaggle file is generated data: figures describe this file only and are not evidence of real retail behaviour or real savings.")

    # ------------------------------------------------------------ KPI cards
    k = dm.home_kpis(view, recs)
    items = [Kpi("Total products", f"{k['products']}", "In the selected rows", "blue"), Kpi("Total stores", f"{k['stores']}", "In the selected rows", "blue"),
             Kpi("Units sold", sup.fmt_int(k["units_sold"]), f"{k['days']} days selected", "blue")]
    if "inventory_units" in k:
        value = f"{sup.fmt_int(k['inventory_value'])}" if k["inventory_value"] is not None else "n/a"
        items += [Kpi("Inventory units", sup.fmt_int(k["inventory_units"]), f"Stock on hand on {snap:%d %b %Y}", "blue"),
                  Kpi("Est. inventory value", value, "Stock x dataset price; not acquisition cost", "blue"),
                  Kpi("Products to reorder", f"{k['reorder_now']} of {k['series']}", "Store-products at or below reorder point", "amber"),
                  Kpi("Critical stock-out risk", str(k["critical"]), "Estimated probability 50% or more", "red"),
                  Kpi("Excess inventory", str(k["excess"]), "Position above the excess threshold", "blue")]
    kpi_row(items)
    if recs is not None:
        st.caption(f"Sales figures follow the sidebar date range. Stock figures are a snapshot at the start of {snap:%d %b %Y} (set under Inventory settings) "
                   f"and follow the store, product, category and region filters. Lead time {params.lead_time_days:g} day(s), service level {params.service_level:.1%}.")

    # ------------------------------------------------------------ inventory health
    section("Inventory health", "Each store-product falls into exactly one group at the snapshot.")
    if recs is None:
        st.info("Inventory health needs valid inventory settings and data around the snapshot date.")
    else:
        counts = dm.health_counts(recs)
        c1, c2 = st.columns([1, 1])
        c1.altair_chart(charts.health_donut(counts), use_container_width=True)
        c2.altair_chart(charts.health_bar(counts), use_container_width=True)
        shown = counts.assign(share=counts["share"].map(lambda v: f"{v:.1%}")).rename(columns={"group": "Group", "store_products": "Store-products", "share": "Share"})
        c2.dataframe(shown, use_container_width=True, hide_index=True)
        with st.expander("How the groups are defined"):
            st.markdown(dm.HEALTH_RULES)
        if k.get("series") and k["low"] + k["critical"] > 0.9 * k["series"]:
            callout("info", f"Most store-products are below their reorder point. In this dataset stock on hand covers only a couple of days of demand, so even a short lead time "
                            f"({params.lead_time_days:g} day(s)) makes nearly everything look urgent. Try a shorter lead time in Inventory settings before reading this as a purchasing plan.")

    # ------------------------------------------------------------ trends
    section("Inventory trends", "Charts follow the sidebar filters. Drag to pan, scroll to zoom.")
    freq_label = st.radio("Aggregation", list(FREQ), horizontal=True, key="home_freq")
    try:
        tf = dm.trend_frame(view, FREQ[freq_label])
        a, b = st.columns(2)
        a.altair_chart(charts.measure_line(tf, "units_sold", "Sales trend", BLUE, "Units sold per period"), use_container_width=True)
        b.altair_chart(charts.measure_line(tf, "inventory", "Inventory level trend", GREY, "Average stock on hand (units)"), use_container_width=True)
        c, d = st.columns(2)
        c.altair_chart(charts.measure_line(tf, "demand_per_series_day", "Demand trend", PURPLE, "Units per store-product-day"), use_container_width=True)
        d.altair_chart(charts.sales_vs_inventory_chart(tf), use_container_width=True)
        st.caption("Demand = units sold (the dataset records sales, not unmet demand). Weekly and monthly views drop incomplete first/last periods.")
        st.altair_chart(charts.category_bar(dm.category_demand(view)), use_container_width=True)
    except DataValidationError as exc:
        st.info(str(exc))

    # ------------------------------------------------------------ insights
    section("Actionable insights", "Generated from the filtered data and the stock snapshot; nothing is pre-written.")
    insight_cards(dm.generate_actionable_insights(view, recs))

    # ------------------------------------------------------------ critical table
    section("Critical inventory items", "Ordered by urgency: items that must be reordered first, then higher stock-out probability, then lower projected stock after the lead time.")
    if recs is None:
        st.info("No recommendation table is available for the current filters and settings.")
    else:
        table = dm.critical_items(recs, 10)
        table["Stock-out risk"] = 100 * table["Stock-out risk"]
        st.dataframe(table, use_container_width=True, hide_index=True, height=390, column_config={
            "Stock-out risk": st.column_config.NumberColumn("Stock-out risk", format="%.1f%%"),
            "Current inventory": st.column_config.NumberColumn(format="%d"), "Reorder point": st.column_config.NumberColumn(format="%.0f"),
            "Suggested order qty": st.column_config.NumberColumn(format="%d")})
        st.button("View complete recommendation table", on_click=_go_to_reorder, key="home_view_all")

    # ------------------------------------------------------------ forecast and statistical findings
    section("Demand forecasting result")
    horizons = ad.forecast_horizons(paths)
    if not horizons:
        st.info("No forecasting model has been trained yet. Run:  `python -m src.demand_forecasting`")
    else:
        h = horizons[0]
        _, meta = ad.load_model_cached(h, str(paths.models_dir), sup.mtime(dfc.model_paths(h, paths.models_dir)[0]))
        tm = meta["test_metrics"]
        comp_path = paths.reports_dir / f"forecast_model_comparison_h{h}.csv"
        base_mae = None
        if comp_path.exists():
            comp = ad.read_csv_cached(str(comp_path), sup.mtime(comp_path))
            if "Series mean" in set(comp["model"]):
                base_mae = float(comp.loc[comp["model"] == "Series mean", "mae_test"].iloc[0])
        kpi_row([Kpi(f"Selected model ({h}-day horizon)", meta["model_name"], "Chosen on validation data with a simplicity rule", "purple"),
                 Kpi("Test MAE", f"{tm['mae']:.1f}", "Units per store-day", "purple"), Kpi("Test R2", f"{tm['r2']:.3f}", "Near 0 = almost no explained variation", "purple")]
                + ([Kpi("Series-mean baseline MAE", f"{base_mae:.1f}", "Same test period", "grey")] if base_mae is not None else []))
        st.caption("Daily demand in this dataset is close to random noise, so complex models do not beat a simple average. See Model Evaluation.")

    with st.expander("Statistical findings (generated from the selected data)"):
        try:
            for line in ad.insights_cached(ctx.path, ctx.stamp, ctx.filters):
                st.markdown(f"- {line}")
        except DataValidationError as exc:
            st.info(f"Not enough data in the current selection for the statistical summary: {exc}")
    with st.expander("Pipeline status"):
        status = pd.DataFrame(sup.artifact_status(paths))
        status["ready"] = status["ready"].map({True: "Ready", False: "Missing"})
        st.dataframe(status.rename(columns={"item": "Pipeline output", "ready": "Status", "command": "Command that creates it"}), use_container_width=True, hide_index=True)
