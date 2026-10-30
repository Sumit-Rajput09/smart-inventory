"""Exploratory Data Analysis: charts with captions, statistical tests (H0/H1/statistic/p/interpretation) and generated interpretation."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import app_charts as charts
from src import app_data as ad
from src import app_support as sup
from src import dashboard_metrics as dm
from src import eda
from src.ui_components import callout, page_header
from src.ui_theme import BLUE, style_chart
from src.utils import DataValidationError

# label, kind, figure name, interpretation keywords, caption
EDA_MENU = [
    ("Sales trends (daily, weekly, monthly)", "trend", None, ("TREND",), "Total units sold over time. The red line is a 7-day average that smooths day-to-day noise. A flat line means no growth or decline to plan for."),
    ("Sales by product", "fig", "by_product", ("PRODUCTS",), "Average units sold per store-day for each product with 95% confidence bars. Overlapping bars mean the products are not clearly different."),
    ("Sales by store", "fig", "by_store", ("STORES",), "Average units sold per store-day with 95% confidence bars; the red dashed line is the overall mean."),
    ("Sales by category", "fig", "by_category", ("CATEGORIES",), "Average units sold per row for each category. Category is recorded row by row in this dataset."),
    ("Sales by region", "fig", "by_region", ("REGIONS",), "Average units sold per row for each region."),
    ("Category and region comparison", "compare", None, ("CATEGORIES", "REGIONS"), "Total and average units sold, side by side. Totals depend on how many rows each group has; compare the averages for demand intensity."),
    ("Top-selling and slow-moving products", "fig", "top_slow", ("PRODUCTS",), "The five highest and lowest products by mean units per store-day, with sell-through (units sold / stock)."),
    ("Inventory level distribution", "fig", "inventory", ("INVENTORY",), "How much stock is held. A flat histogram means stock levels are spread evenly between the minimum and maximum."),
    ("Sales vs inventory", "fig", "sales_vs_inventory", ("INVENTORY", "SELL-THROUGH"), "Each point is a store-day. Sales never exceed stock here, so inventory acts like a ceiling; this describes how the data was generated and does not show that more stock causes more sales."),
    ("Price and discount analysis", "fig", "price_discount", ("DISCOUNT",), "Whether price, discount or the competitor's price move sales. Flat bars mean no relationship."),
    ("Promotion and holiday comparison", "fig", "promotion", ("PROMOTION",), "Mean units sold with and without the Holiday/Promotion flag, with the Welch t-test p-value."),
    ("Correlation analysis", "corr", None, ("CORRELATIONS", "LEAKAGE"), "Correlation measures linear association, not cause. A value near 0 means no linear relationship; demand_forecast correlates ~1 because it nearly copies units sold (leakage)."),
    ("Histograms (distributions)", "fig", "histograms", (), "Shape of key numeric columns. A long right tail (positive skew) means a few very large values."),
    ("Box plots", "fig", "boxplots", (), "Spread of units sold by category, store and region. The box is the middle 50%; dots are unusual days."),
    ("Outlier analysis", "outliers", "outliers", ("OUTLIERS",), "Share of days flagged as unusual for their own store-product (1.5 x IQR rule)."),
    ("Seasonal patterns", "fig", "seasonality", ("SEASONALITY",), "Mean units sold by month and weekday. Small differences mean calendar features carry little forecasting signal."),
    ("Statistics and hypothesis tests", "stats", None, (), "Formal tests at the 5% level, each with its hypotheses, statistic, p-value and interpretation."),
]


def render(ctx: ad.Ctx) -> None:
    view = ctx.view
    page_header("Exploratory Data Analysis", "Charts respond to the sidebar filters. Each chart has a caption and an interpretation generated from the numbers.", [f"{len(view):,} rows selected", "Synthetic dataset"])
    choice = st.selectbox("Analysis", [m[0] for m in EDA_MENU], key="eda_choice")
    label, kind, name, keys, caption = next(m for m in EDA_MENU if m[0] == choice)
    path, stamp = ctx.path, ctx.stamp
    try:
        if kind == "trend":
            freq = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, key="eda_freq")
            st.altair_chart(charts.trend_chart(view, {"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq]), use_container_width=True)
            st.caption(caption + " Drag to pan, scroll to zoom. Weekly and monthly views drop incomplete first/last periods.")
        elif kind == "compare":
            c1, c2 = st.columns(2)
            for col, grp, title in ((c1, "category", "Category"), (c2, "region", "Region")):
                t = eda.sales_by(view, grp)
                t[grp] = t[grp].astype(str)
                col.altair_chart(style_chart(alt.Chart(t).mark_bar(color=BLUE, cornerRadiusEnd=3, size=22).encode(
                    y=alt.Y(f"{grp}:N", sort="-x", title=None), x=alt.X("mean_units:Q", title="Mean units per row"),
                    tooltip=[grp, alt.Tooltip("mean_units:Q", format=".1f"), alt.Tooltip("total_units:Q", format=",.0f"), "rows:Q"]).properties(height=max(110, 36 * len(t) + 30), title=f"{title}: mean units")), use_container_width=True)
                col.dataframe(t[[grp, "rows", "total_units", "mean_units"]], use_container_width=True, hide_index=True)
            st.caption(caption)
        elif kind == "corr":
            method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="eda_corr")
            st.pyplot(ad.eda_figure_cached("correlation", path, stamp, ctx.filters, method))
            st.caption(caption)
            st.dataframe(eda.correlation_tests(view), use_container_width=True, hide_index=True)
        elif kind == "stats":
            _statistics(ctx)
        else:
            st.pyplot(ad.eda_figure_cached(name, path, stamp, ctx.filters))
            st.caption(caption)
            if kind == "outliers":
                st.dataframe(eda.outlier_summary(view), use_container_width=True, hide_index=True)
                st.caption("Outliers are flagged, not removed: unusually high-sales days are valid demand the inventory plan must cover.")
            if name == "top_slow":
                top, slow = eda.top_and_slow(view, 5)
                c1, c2 = st.columns(2)
                c1.markdown("**Top 5 (highest mean units per store-day)**")
                c1.dataframe(top, use_container_width=True, hide_index=True)
                c2.markdown("**Slowest 5**")
                c2.dataframe(slow, use_container_width=True, hide_index=True)
    except DataValidationError as exc:
        st.warning(f"This analysis cannot be drawn for the current selection: {exc}")
        return
    if keys:
        try:
            lines = sup.insights_for(ad.insights_cached(path, stamp, ctx.filters), *keys)
        except DataValidationError:
            lines = []
        if lines:
            st.markdown("**Interpretation**")
            for line in lines:
                st.markdown(f"- {line}")


def _statistics(ctx: ad.Ctx) -> None:
    view = ctx.view
    st.markdown("**Descriptive statistics**")
    st.dataframe(eda.descriptive_statistics(view).reset_index(names="column"), use_container_width=True, hide_index=True)
    st.markdown("**Statistical tests on units sold** (significance level 5%)")
    try:
        table = ad.hypothesis_cached(ctx.path, ctx.stamp, ctx.filters)
    except DataValidationError as exc:
        st.info(str(exc))
        return
    st.dataframe(table, use_container_width=True, hide_index=True, column_config={"P-value": st.column_config.NumberColumn(format="%.4f"),
                                                                              "Interpretation": st.column_config.TextColumn(width="large")})
    n_sig = int((table["P-value"] < eda.ALPHA).sum())
    callout("info", f"**{n_sig} of {len(table)} tests are significant at 5%.** With this many tests, about 1 in 20 true-null tests looks significant by chance, "
                    "and a significant result with a tiny effect size (see eta2 or Cohen's d) has little practical meaning. Association is not causation.")
    st.caption("Assumptions: rows of one store-product repeat over time, so independence holds only approximately; sales are right-skewed, which is why rank-based checks (Mann-Whitney, Kruskal-Wallis) are shown beside the parametric tests.")
