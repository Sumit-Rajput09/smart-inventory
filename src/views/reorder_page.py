"""Reorder Recommendations: summary cards, filters, recommendation table and lead-time sensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import app_charts as charts
from src import app_data as ad
from src import app_support as sup
from src import reorder_optimizer as ro
from src.preprocess import GROUP_KEYS
from src.ui_components import Kpi, callout, kpi_row, page_header, section, show_table
from src.utils import DataValidationError

STATUSES = ["Low stock", "Adequate stock", "Excess stock", "Insufficient history"]
RISKS = ["Low", "Medium", "High", "Critical", "Unknown"]


def sensitivity_readout(sens: pd.DataFrame) -> str:
    """One plain sentence about how the lead-time assumption changes the picture (computed from the sensitivity table)."""
    s = sens.sort_values("lead_time_days")
    first, last = s.iloc[0], s.iloc[-1]
    over = s[s["pct_low_stock"] >= 50]
    cross = (f"More than half of the items fall into Low-stock status from a lead time of {over['lead_time_days'].iloc[0]:g} day(s)." if len(over) else
             "Fewer than half of the items are in Low-stock status at every lead time tested.")
    return (f"At {first['lead_time_days']:g} day(s) {first['pct_low_stock']:.0f}% of items are low on stock and the mean reorder point is {first['mean_reorder_point']:,.0f} units; "
            f"at {last['lead_time_days']:g} days it is {last['pct_low_stock']:.0f}% and {last['mean_reorder_point']:,.0f} units, against a mean stock on hand of {last['mean_current_inventory']:,.0f}. " + cross)


def render(ctx: ad.Ctx) -> None:
    page_header("Reorder Recommendations", "Replenishment suggestions from transparent formulas: reorder when the inventory position is at or below the reorder point, then order up to the target inventory.",
                ["(s, S) policy", "Lead time is an assumption", "Synthetic dataset"])
    st.latex(r"\text{Position} = \text{On hand} + \text{On order} \qquad \text{Target} = \text{ROP} + d \times R \qquad \text{Order} = \max(0,\ \text{Target} - \text{On hand} - \text{On order})")
    on_order_bytes = None
    with st.expander("Quantities already on order (optional)"):
        st.caption("Not in the dataset: assumed 0 for every item. Upload a CSV with columns store_id, product_id, on_order to include them.")
        up = st.file_uploader("On-order CSV", type="csv", key="on_order_file")
        if up is not None:
            try:
                sup.parse_on_order_upload(up.getvalue())
                on_order_bytes = up.getvalue()
                st.success("On-order quantities loaded.")
            except DataValidationError as exc:
                st.error(str(exc))
    recs, params = ad.get_recommendations(ctx.paths, on_order_bytes)
    if recs is None:
        return
    recs = ad.restrict_recs(recs, ctx.filters)
    if recs is None:
        return

    # ------------------------------------------------------------ page filters
    st.markdown("**Filters** (store, product, category and region are set in the sidebar)")
    f1, f2, f3 = st.columns([1, 2, 2])
    only = f1.toggle("Only items to reorder", value=True, key="ro_only")
    status = f2.multiselect("Inventory status", STATUSES, default=STATUSES, key="ro_status")
    risks = f3.multiselect("Risk level", RISKS, default=RISKS, key="ro_risk")
    shown = recs[recs["stock_status"].isin(status) & recs["risk_level"].astype(str).isin(risks)]
    if only:
        shown = shown[shown["reorder_now"]]

    # ------------------------------------------------------------ summary cards (for the filtered rows)
    k = ro.summarize(recs)
    kpi_row([Kpi("Items requiring reorder", f"{k['reorder_now']} of {k['series']}", "Position at or below the reorder point", "amber"),
             Kpi("Critical items", str(k["risk_counts"]["Critical"]), "Estimated stock-out probability 50% or more", "red"),
             Kpi("Excess inventory", str(k["status_counts"].get("Excess stock", 0)), "Position above the excess threshold x target", "blue"),
             Kpi("Total suggested order", f"{sup.fmt_int(k['units_to_order'])} units", "Sum over the items to reorder", "blue"),
             Kpi("Median days of cover", f"{k['median_days_of_cover']:.1f}", "Inventory position / average daily demand", "grey")])
    st.caption(f"Snapshot {recs['snapshot_date'].iloc[0]:%d %b %Y} | lead time {params.lead_time_days:g} d | service level {params.service_level:.1%} | "
               f"review period {params.review_period_days:g} d | demand history {params.lookback_days} d. Cards cover all rows in the sidebar filters; the table below also applies the status and risk filters.")
    if k["share_reorder"] > 0.9:
        callout("warn", "**Nearly every item is flagged.** In this dataset stock on hand covers only about two days of demand, so lead times of a few days make every item urgent. "
                        "Check the sensitivity below before reading the list as a purchase order.")

    # ------------------------------------------------------------ table
    if shown.empty:
        st.info("No items match: nothing needs reordering with the current settings and filters.")
    else:
        st.altair_chart(charts.top_urgent_chart(shown), use_container_width=True)
    cols = [*GROUP_KEYS, "category", "current_inventory", "on_order", "inventory_position", "reorder_point", "safety_stock", "target_inventory",
            "suggested_order_qty", "stockout_probability", "risk_level", "stock_status", "recommendation"]
    detail = st.toggle("Show all calculation columns", value=False, key="ro_detail")
    table = (shown if detail else shown[[c for c in cols if c in shown.columns]]).copy()
    table["risk_level"] = table["risk_level"].astype(str)
    if detail:
        table["days_of_cover"] = table["days_of_cover"].replace(np.inf, np.nan)
    show_table(table, "reorder", file_name="reorder_recommendations.csv", height=460,
               column_config={"stockout_probability": st.column_config.NumberColumn("stock-out probability", format="%.3f"),
                              **{c: st.column_config.NumberColumn(c, format="%.0f") for c in ("safety_stock", "reorder_point", "target_inventory", "projected_stock_after_lead_time", "current_inventory", "inventory_position", "on_order")}})

    # ------------------------------------------------------------ sensitivity
    section("How much does the lead-time assumption matter?", "The supplier lead time is not in the data. These charts re-run the recommendations for several lead times with your other settings unchanged.")
    try:
        sens = ad.sensitivity_cached(ctx.path, ctx.stamp, params, str(st.session_state.get("inv_snapshot")))
        c1, c2 = st.columns(2)
        c1.altair_chart(charts.sensitivity_chart(sens), use_container_width=True)
        c2.altair_chart(charts.sensitivity_units_chart(sens), use_container_width=True)
        callout("info", sensitivity_readout(sens))
        with st.expander("Sensitivity table"):
            st.dataframe(sens, use_container_width=True, hide_index=True)
    except (DataValidationError, ValueError) as exc:
        st.info(str(exc))
    with st.expander("Assumptions and limitations (read before using these numbers)"):
        st.text(ro.ASSUMPTIONS)
