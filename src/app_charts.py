"""Interactive Altair charts for the dashboard (tooltips, zoom/pan where useful). Pure functions returning styled alt charts.

Rules: one y-axis per chart (never dual-axis), thin marks, status colours always paired with a text label (legend, axis or tooltip),
actual values in blue, forecasts and machine-learning outputs in purple.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from src.eda import aggregate_sales
from src.preprocess import DATE
from src.ui_theme import (AMBER, BLUE, GREEN, GREY, HEALTH_COLORS, PURPLE, RED, RISK_COLORS, STATUS_COLORS, MUTED,
                          color_scale, style_chart)

RISK_SCALE = color_scale(RISK_COLORS)
STATUS_SCALE = color_scale(STATUS_COLORS)
HEALTH_SCALE = color_scale(HEALTH_COLORS)
RISK_ORDER = ["Low", "Medium", "High", "Critical", "Unknown"]


# ================================================================ time series
def trend_chart(df: pd.DataFrame, freq: str = "D") -> alt.Chart:
    """Total units sold per day/week/month. Daily view adds a 7-day average."""
    s = aggregate_sales(df, freq).dropna().rename("units").reset_index()
    s.columns = ["date", "units"]
    base = alt.Chart(s).encode(x=alt.X("date:T", title=None),
                               tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("units:Q", title="Units sold", format=",.0f")])
    line = base.mark_line(color=BLUE, opacity=0.55 if freq == "D" else 1, strokeWidth=1.5, point=freq != "D").encode(
        y=alt.Y("units:Q", title="Units sold (all selected store-products)"))
    if freq != "D":
        return style_chart(line.properties(height=300).interactive())
    s["average"] = s["units"].rolling(7, min_periods=1).mean()
    avg = alt.Chart(s).mark_line(color=RED, strokeWidth=2).encode(
        x="date:T", y="average:Q", tooltip=[alt.Tooltip("date:T"), alt.Tooltip("average:Q", title="7-day average", format=",.0f")])
    return style_chart((line + avg).properties(height=300).interactive())


def measure_line(frame: pd.DataFrame, column: str, title: str, color: str = BLUE, y_title: str = "", band: tuple[str, str] | None = None) -> alt.Chart:
    """One measure over time with an optional lower/upper band (columns given by name)."""
    layers = []
    if band:
        layers.append(alt.Chart(frame).mark_area(opacity=0.15, color=color).encode(x="date:T", y=f"{band[0]}:Q", y2=f"{band[1]}:Q"))
    layers.append(alt.Chart(frame).mark_line(color=color, strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y(f"{column}:Q", title=y_title, scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("date:T", title="Period"), alt.Tooltip(f"{column}:Q", title=y_title or column, format=",.1f")]))
    return style_chart(alt.layer(*layers).properties(height=240, title=title).interactive())


def sales_vs_inventory_chart(frame: pd.DataFrame) -> alt.Chart:
    """Average units sold per day and average stock on hand, same unit (units) so one axis is honest."""
    long = frame.melt(id_vars="date", value_vars=["sold_per_day", "inventory"], var_name="measure", value_name="units")
    long["measure"] = long["measure"].map({"sold_per_day": "Units sold per day", "inventory": "Stock on hand"})
    return style_chart(alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("units:Q", title="Units (all selected store-products)"),
        color=alt.Color("measure:N", title=None, scale=alt.Scale(domain=["Units sold per day", "Stock on hand"], range=[BLUE, GREY])),
        tooltip=["measure:N", alt.Tooltip("date:T", title="Period"), alt.Tooltip("units:Q", format=",.0f")]).properties(height=240, title="Sales versus inventory").interactive())


def category_bar(cat: pd.DataFrame) -> alt.Chart:
    return style_chart(alt.Chart(cat).mark_bar(color=BLUE, cornerRadiusEnd=3, size=16).encode(
        y=alt.Y("category:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=160)), x=alt.X("total_units:Q", title="Total units sold"),
        tooltip=["category:N", alt.Tooltip("total_units:Q", format=",.0f", title="Units sold"), alt.Tooltip("mean_units:Q", format=",.1f", title="Mean per store-day"),
                 alt.Tooltip("share:Q", format=".1%", title="Share")]).properties(height=max(120, 36 * len(cat) + 30), title="Demand by category"))


# ================================================================ inventory health / risk
def health_donut(counts: pd.DataFrame) -> alt.Chart:
    d = counts[counts["store_products"] > 0]
    arc = alt.Chart(d).mark_arc(innerRadius=62, outerRadius=105, stroke="#FFFFFF", strokeWidth=2).encode(
        theta=alt.Theta("store_products:Q"), color=alt.Color("group:N", scale=HEALTH_SCALE, title=None, sort=list(HEALTH_SCALE.domain), legend=alt.Legend(orient="right")),
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("store_products:Q", title="Store-products"), alt.Tooltip("share:Q", format=".1%")])
    return style_chart(arc.properties(height=260, title="Inventory health (store-products)"))


def health_bar(counts: pd.DataFrame) -> alt.Chart:
    return style_chart(alt.Chart(counts).mark_bar(cornerRadiusEnd=3, size=20).encode(
        y=alt.Y("group:N", sort=list(HEALTH_SCALE.domain), title=None), x=alt.X("store_products:Q", title="Store-products"),
        color=alt.Color("group:N", scale=HEALTH_SCALE, legend=None),
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("store_products:Q", title="Store-products"), alt.Tooltip("share:Q", format=".1%")]).properties(height=200))


def risk_donut(recs: pd.DataFrame) -> alt.Chart:
    counts = recs["risk_level"].astype(str).value_counts().rename_axis("risk").reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum()
    return style_chart(alt.Chart(counts).mark_arc(innerRadius=58, outerRadius=100, stroke="#FFFFFF", strokeWidth=2).encode(
        theta="count:Q", color=alt.Color("risk:N", scale=RISK_SCALE, sort=RISK_ORDER, title="Risk level", legend=alt.Legend(orient="right")),
        tooltip=[alt.Tooltip("risk:N", title="Risk level"), alt.Tooltip("count:Q", title="Store-products"), alt.Tooltip("share:Q", format=".1%")]).properties(height=240, title="Rule-based risk distribution"))


def risk_bar_chart(recs: pd.DataFrame) -> alt.Chart:
    counts = recs["risk_level"].astype(str).value_counts().rename_axis("risk").reset_index(name="count")
    return style_chart(alt.Chart(counts).mark_bar(cornerRadiusEnd=3, size=34).encode(
        x=alt.X("risk:N", sort=RISK_ORDER, title="Risk level", axis=alt.Axis(labelAngle=0)), y=alt.Y("count:Q", title="Store-products"),
        color=alt.Color("risk:N", scale=RISK_SCALE, legend=None), tooltip=["risk:N", "count:Q"]).properties(height=260))


def position_vs_rop_chart(recs: pd.DataFrame) -> alt.Chart:
    d = recs.copy()
    d["series"] = d["store_id"].astype(str) + " / " + d["product_id"].astype(str)
    lim = float(max(d["reorder_point"].max(), d["inventory_position"].max()) * 1.05) or 1.0
    pts = alt.Chart(d).mark_circle(size=60, opacity=0.85).encode(
        x=alt.X("reorder_point:Q", title="Reorder point (units)", scale=alt.Scale(domain=[0, lim])),
        y=alt.Y("inventory_position:Q", title="Inventory position (units)", scale=alt.Scale(domain=[0, lim])),
        color=alt.Color("stock_status:N", scale=STATUS_SCALE, title="Status"),
        tooltip=["series:N", alt.Tooltip("inventory_position:Q", format=",.0f"), alt.Tooltip("reorder_point:Q", format=",.0f"),
                 alt.Tooltip("stockout_probability:Q", format=".1%"), "risk_level:N"])
    diag = alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    return style_chart((pts + diag).properties(height=300, title="Position vs reorder point (dashed: equal)").interactive())


def top_urgent_chart(recs: pd.DataFrame, n: int = 15) -> alt.Chart:
    t = recs[recs["suggested_order_qty"] > 0].nlargest(n, "suggested_order_qty").copy()
    t["series"] = t["store_id"].astype(str) + " / " + t["product_id"].astype(str)
    return style_chart(alt.Chart(t).mark_bar(cornerRadiusEnd=3, size=14).encode(
        y=alt.Y("series:N", sort="-x", title=None), x=alt.X("suggested_order_qty:Q", title="Suggested order (units)"),
        color=alt.Color("risk_level:N", scale=RISK_SCALE, title="Risk"),
        tooltip=["series:N", alt.Tooltip("suggested_order_qty:Q", format=",.0f"), alt.Tooltip("stockout_probability:Q", format=".1%"), "risk_level:N"]
    ).properties(height=min(24 * len(t) + 40, 520)))


def sensitivity_chart(sens: pd.DataFrame) -> alt.Chart:
    long = sens.melt(id_vars="lead_time_days", value_vars=["pct_low_stock", "pct_critical_or_high"], var_name="measure", value_name="percent")
    long["measure"] = long["measure"].map({"pct_low_stock": "Low-stock status", "pct_critical_or_high": "High or Critical risk"})
    return style_chart(alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("lead_time_days:Q", title="Lead time (days)"), y=alt.Y("percent:Q", title="Items affected (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("measure:N", title=None, scale=alt.Scale(domain=["Low-stock status", "High or Critical risk"], range=[RED, AMBER]), legend=alt.Legend(orient="bottom")),
        tooltip=["measure:N", "lead_time_days:Q", alt.Tooltip("percent:Q", format=".0f")]).properties(height=250, title="Share of items affected by lead time"))


def sensitivity_units_chart(sens: pd.DataFrame) -> alt.Chart:
    """Mean reorder point vs mean stock on hand for each lead time (both in units)."""
    long = sens.melt(id_vars="lead_time_days", value_vars=["mean_reorder_point", "mean_current_inventory"], var_name="measure", value_name="units")
    long["measure"] = long["measure"].map({"mean_reorder_point": "Mean reorder point", "mean_current_inventory": "Mean stock on hand"})
    return style_chart(alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("lead_time_days:Q", title="Lead time (days)"), y=alt.Y("units:Q", title="Units per store-product"),
        color=alt.Color("measure:N", title=None, scale=alt.Scale(domain=["Mean reorder point", "Mean stock on hand"], range=[PURPLE, GREY])),
        tooltip=["measure:N", "lead_time_days:Q", alt.Tooltip("units:Q", format=",.0f")]).properties(height=260, title="Reorder point grows with lead time; stock on hand does not"))


# ================================================================ forecasting
def forecast_chart(history: pd.DataFrame, target_date, forecast: float, series_mean: float) -> alt.Chart:
    """Recent actual sales of one store-product with the forecast point and the series mean (kept for backward compatibility)."""
    h = history[[DATE, "units_sold"]].rename(columns={DATE: "date", "units_sold": "units"})
    actual = alt.Chart(h).mark_line(color=BLUE, point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("date:T", title=None), y=alt.Y("units:Q", title="Units sold"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("units:Q", title="Actual units", format=",.0f")])
    mean_rule = alt.Chart(pd.DataFrame({"y": [series_mean]})).mark_rule(color=GREY, strokeDash=[5, 4]).encode(y="y:Q")
    point = alt.Chart(pd.DataFrame({"date": [pd.Timestamp(target_date)], "units": [forecast]})).mark_point(
        color=PURPLE, size=160, filled=True, shape="diamond").encode(
        x="date:T", y="units:Q", tooltip=[alt.Tooltip("date:T", title="Forecast date"), alt.Tooltip("units:Q", title="Forecast units", format=",.1f")])
    return style_chart((actual + mean_rule + point).properties(height=300).interactive())


def forecast_vs_actual_chart(history: pd.DataFrame, target_date, forecast: float, lower: float | None, upper: float | None,
                             y_title: str = "Units sold per day") -> alt.Chart:
    """Actual history (blue line) and the forecast point (purple diamond, dashed connector) with an optional interval bar."""
    h = history.rename(columns={"units_sold": "units"})[["date", "units"]]
    last = h.iloc[-1]
    span = (pd.Timestamp(target_date) - h["date"].min()).days
    pad = pd.Timedelta(days=max(3, int(0.05 * span)))      # room on the right so the forecast marker is not clipped
    x_scale = alt.Scale(domain=[h["date"].min().isoformat(), (pd.Timestamp(target_date) + pad).isoformat()])
    actual = alt.Chart(h).mark_line(color=BLUE, strokeWidth=1.8).encode(
        x=alt.X("date:T", title=None, scale=x_scale), y=alt.Y("units:Q", title=y_title, scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("units:Q", title="Actual", format=",.0f")])
    link = alt.Chart(pd.DataFrame({"date": [last["date"], pd.Timestamp(target_date)], "units": [last["units"], forecast]})).mark_line(color=PURPLE, strokeDash=[4, 4]).encode(x=alt.X("date:T", scale=x_scale), y="units:Q")
    fc = pd.DataFrame({"date": [pd.Timestamp(target_date)], "units": [forecast], "lower": [lower if lower is not None else forecast], "upper": [upper if upper is not None else forecast]})
    layers = [actual, link]
    if lower is not None and upper is not None:
        layers.append(alt.Chart(fc).mark_rule(color=PURPLE, strokeWidth=3, opacity=0.5).encode(x="date:T", y="lower:Q", y2="upper:Q"))
    layers.append(alt.Chart(fc).mark_point(color=PURPLE, size=150, filled=True, shape="diamond").encode(
        x="date:T", y="units:Q", tooltip=[alt.Tooltip("date:T", title="Forecast date"), alt.Tooltip("units:Q", title="Forecast", format=",.1f"),
                                          alt.Tooltip("lower:Q", title="Interval low", format=",.1f"), alt.Tooltip("upper:Q", title="Interval high", format=",.1f")]))
    return style_chart(alt.layer(*layers).properties(height=300).interactive())


def forecast_bar_chart(table: pd.DataFrame, top: int = 25) -> alt.Chart:
    t = table.nlargest(top, "forecast_units").copy()
    t["series"] = t["store_id"].astype(str) + " / " + t["product_id"].astype(str)
    return style_chart(alt.Chart(t).mark_bar(color=PURPLE, cornerRadiusEnd=3, size=12).encode(
        y=alt.Y("series:N", sort="-x", title=None), x=alt.X("forecast_units:Q", title="Forecast units sold"),
        tooltip=["series:N", alt.Tooltip("forecast_units:Q", format=",.1f"), alt.Tooltip("target_date:T")]).properties(height=min(22 * len(t) + 40, 700), title=f"Top {len(t)} forecasts"))


def model_comparison_chart(table: pd.DataFrame, metric: str = "mae") -> alt.Chart:
    """Grouped bars (validation vs test) per model. `table` has columns model, <metric>_val, <metric>_test."""
    cols = [f"{metric}_val", f"{metric}_test"]
    long = table.melt(id_vars="model", value_vars=cols, var_name="period", value_name=metric)
    long["period"] = long["period"].map({cols[0]: "Validation", cols[1]: "Test"})
    return style_chart(alt.Chart(long).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X("model:N", title=None, sort=list(table["model"]), axis=alt.Axis(labelAngle=-30)), xOffset="period:N",
        y=alt.Y(f"{metric}:Q", title=metric.upper()), color=alt.Color("period:N", scale=alt.Scale(domain=["Validation", "Test"], range=[GREY, PURPLE]), title=None),
        tooltip=["model:N", "period:N", alt.Tooltip(f"{metric}:Q", format=".3f")]).properties(height=280, title=f"{metric.upper()} by model"))


def test_metric_bars(table: pd.DataFrame, metric: str, baseline: str = "Series mean") -> alt.Chart:
    """Horizontal bars of one test metric; the baseline model is grey, machine-learning models purple, simple rules blue."""
    t = table[["model", metric]].dropna().copy()
    ml = {"Linear Regression", "Random Forest", "Hist Gradient Boosting"}
    t["kind"] = np.where(t["model"] == baseline, "Baseline (series mean)", np.where(t["model"].isin(ml), "Machine learning", "Simple rule"))
    return style_chart(alt.Chart(t).mark_bar(cornerRadiusEnd=3, size=14).encode(
        y=alt.Y("model:N", sort="x" if metric not in ("r2",) else "-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=170)), x=alt.X(f"{metric}:Q", title=metric.upper()),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["Baseline (series mean)", "Machine learning", "Simple rule"], range=[GREY, PURPLE, BLUE])),
        tooltip=["model:N", alt.Tooltip(f"{metric}:Q", format=".3f"), "kind:N"]).properties(height=max(160, 34 * len(t) + 40), title=f"Test {metric.upper()}"))


def actual_vs_predicted_chart(daily: pd.DataFrame, pred_columns: list[str]) -> alt.Chart:
    """daily: date, units_sold, <model columns> (totals per day over all store-products in the test period)."""
    long = daily.melt(id_vars="date", value_vars=["units_sold", *pred_columns], var_name="series", value_name="units")
    long["series"] = long["series"].replace({"units_sold": "Actual"})
    domain = ["Actual", *pred_columns]
    return style_chart(alt.Chart(long).mark_line(strokeWidth=1.8).encode(
        x=alt.X("date:T", title=None), y=alt.Y("units:Q", title="Units sold per day (all store-products)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=domain, range=[BLUE, PURPLE, GREY, AMBER][: len(domain)])),
        tooltip=["series:N", alt.Tooltip("date:T"), alt.Tooltip("units:Q", format=",.0f")]).properties(height=280, title="Test period: actual vs predicted").interactive())


def residual_hist(residuals: pd.Series, title: str = "Residuals (predicted - actual)") -> alt.Chart:
    d = pd.DataFrame({"residual": np.asarray(residuals, float)})
    return style_chart(alt.Chart(d).mark_bar(color=PURPLE, opacity=0.8).encode(
        x=alt.X("residual:Q", bin=alt.Bin(maxbins=40), title="Predicted - actual (units)"), y=alt.Y("count()", title="Records"),
        tooltip=[alt.Tooltip("count()", title="Records")]).properties(height=240, title=title))


def residual_scatter(pred: pd.Series, resid: pd.Series, max_points: int = 3000) -> alt.Chart:
    d = pd.DataFrame({"predicted": np.asarray(pred, float), "residual": np.asarray(resid, float)})
    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)
    pts = alt.Chart(d).mark_circle(size=14, opacity=0.35, color=PURPLE).encode(x=alt.X("predicted:Q", title="Predicted units", scale=alt.Scale(zero=False)),
                                                                              y=alt.Y("residual:Q", title="Predicted - actual"))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    return style_chart((pts + zero).properties(height=240, title="Residual vs predicted"))


# ================================================================ classifier (simulated label)
def roc_chart(roc: pd.DataFrame) -> alt.Chart:
    line = alt.Chart(roc).mark_line(strokeWidth=2).encode(x=alt.X("fpr:Q", title="False-positive rate", scale=alt.Scale(domain=[0, 1])),
                                                          y=alt.Y("tpr:Q", title="True-positive rate (recall)", scale=alt.Scale(domain=[0, 1])),
                                                          color=alt.Color("model:N", title=None), tooltip=["model:N", alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")])
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    return style_chart((line + diag).properties(height=280, title="ROC curve (simulated label, test period)"))


def pr_chart(pr: pd.DataFrame, prevalence: float) -> alt.Chart:
    line = alt.Chart(pr).mark_line(strokeWidth=2).encode(x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                                                         y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
                                                         color=alt.Color("model:N", title=None), tooltip=["model:N", alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")])
    base = alt.Chart(pd.DataFrame({"y": [prevalence]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    return style_chart((line + base).properties(height=280, title="Precision-recall curve (dashed: chance = prevalence)"))


def calibration_chart(cal: pd.DataFrame) -> alt.Chart:
    pts = alt.Chart(cal).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("mean_score:Q", title="Mean predicted score (per bin)", scale=alt.Scale(domain=[0, 1])), y=alt.Y("observed_rate:Q", title="Observed simulated-stock-out rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", title=None), tooltip=["model:N", alt.Tooltip("mean_score:Q", format=".2f"), alt.Tooltip("observed_rate:Q", format=".2f"), "n:Q"])
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    return style_chart((pts + diag).properties(height=280, title="Reliability (dashed: perfectly calibrated)"))


def confusion_heatmap(tn: int, fp: int, fn: int, tp: int, title: str = "Confusion matrix") -> alt.Chart:
    d = pd.DataFrame({"actual": ["No stock-out", "No stock-out", "Stock-out", "Stock-out"], "predicted": ["No stock-out", "Stock-out", "No stock-out", "Stock-out"],
                      "count": [tn, fp, fn, tp]})
    base = alt.Chart(d).encode(x=alt.X("predicted:N", title="Predicted", sort=["No stock-out", "Stock-out"], axis=alt.Axis(labelAngle=0)),
                               y=alt.Y("actual:N", title="Actual (simulated label)", sort=["No stock-out", "Stock-out"]))
    rect = base.mark_rect(stroke="#FFFFFF", strokeWidth=2).encode(color=alt.Color("count:Q", scale=alt.Scale(range=["#EDE9FE", PURPLE]), legend=None), tooltip=["actual:N", "predicted:N", alt.Tooltip("count:Q", format=",")])
    text = base.mark_text(fontSize=15, fontWeight="bold").encode(text=alt.Text("count:Q", format=","), color=alt.condition(alt.datum.count > d["count"].max() * 0.5, alt.value("white"), alt.value("#0F172A")))
    return style_chart((rect + text).properties(height=230, width=260, title=title))


# ================================================================ data quality / policy simulation
def missing_bar(table: pd.DataFrame) -> alt.Chart:
    """table: column, missing_after_cleaning [, missing_before_cleaning]."""
    cols = [c for c in ("missing_before_cleaning", "missing_after_cleaning") if c in table.columns]
    long = table.melt(id_vars="column", value_vars=cols, var_name="stage", value_name="missing")
    long["stage"] = long["stage"].map({"missing_before_cleaning": "Before cleaning", "missing_after_cleaning": "After cleaning"})
    long["missing"] = long["missing"].fillna(0)
    return style_chart(alt.Chart(long).mark_bar(cornerRadiusEnd=2).encode(
        y=alt.Y("column:N", sort=None, title=None), x=alt.X("missing:Q", title="Missing values"), yOffset="stage:N",
        color=alt.Color("stage:N", title=None, scale=alt.Scale(domain=["Before cleaning", "After cleaning"], range=[AMBER, BLUE])),
        tooltip=["column:N", "stage:N", alt.Tooltip("missing:Q", format=",")]).properties(height=max(200, 26 * table["column"].nunique() + 40), title="Missing values per column"))


def policy_tradeoff_chart(table: pd.DataFrame) -> alt.Chart:
    d = table.copy()
    d["target"] = (100 * d["service_level_target"]).round(1).astype(str) + "%"
    pts = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=70), color=PURPLE, strokeWidth=2).encode(
        x=alt.X("avg_on_hand_units:Q", title="Average stock on hand per store-product (units)", scale=alt.Scale(zero=False)),
        y=alt.Y("fill_rate:Q", title="Fill rate (share of demand served)", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("target:N", title="Service-level target"), alt.Tooltip("fill_rate:Q", format=".1%"), alt.Tooltip("stockout_day_frequency:Q", format=".1%", title="Days with a stock-out"),
                 alt.Tooltip("avg_on_hand_units:Q", format=",.0f", title="Avg stock")])
    labels = alt.Chart(d).mark_text(dy=-12, color=MUTED, fontSize=11).encode(x="avg_on_hand_units:Q", y="fill_rate:Q", text="target:N")
    return style_chart((pts + labels).properties(height=280, title="Simulated policy: more stock buys a higher fill rate"))
