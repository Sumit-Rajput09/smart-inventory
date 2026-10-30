"""Replenishment recommendations built on transparent inventory formulas.

Run from the project root (needs `python -m src.preprocess` first):
    python -m src.reorder_optimizer
    python -m src.reorder_optimizer --lead-time 2 --service-level 0.9 --review-days 3
    python -m src.reorder_optimizer --snapshot 2023-12-15 --safety-stock-days 1

POLICY (min-max / "(s, S)" policy) evaluated at the START of a snapshot day t
    inventory position  = stock on hand on day t + quantity already on order
    average daily demand d and its std are estimated from the `lookback` days BEFORE day t (nothing from day t or later)
    lead-time demand    = d x L
    safety stock        = z x std x sqrt(L)           (or a fixed number of units / days of demand, if you set one)
    reorder point  ROP  = lead-time demand + safety stock
    target inventory    = ROP + d x R                  (R = review period: days of demand one order should cover)
    reorder now if inventory position <= ROP
    suggested order     = max(0, target - stock on hand - on order), then optional minimum order quantity and pack size

NOT AVAILABLE IN THE DATASET, therefore assumed or user-supplied: supplier lead times, quantities already on order
(default 0), minimum order quantities, pack sizes, shelf life, costs. All outputs are ESTIMATES, not guarantees.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.eda import load_cleaned
from src.preprocess import DATE, GROUP_KEYS, TARGET
from src.stockout_prediction import (
    RISK_LEVELS,
    RISK_RULES_TEXT,
    classify_risk,
    lead_time_demand,
    reorder_point,
    safety_stock,
    stockout_probability,
    z_value,
)
from src.utils import CLEANED_CSV, REPORTS_DIR, DataFileError, DataValidationError, ensure_dirs, get_logger

log = get_logger()

ASSUMPTIONS = """ASSUMPTIONS AND LIMITATIONS
- Supplier lead times are NOT in the dataset: the lead time is a setting (default 1 day, because stock on hand covers only about 2 days of demand).
- Quantities already on order are NOT in the dataset: assumed 0 unless you supply them. (The 'Units Ordered' column is not used: it is unrelated to stock or sales.)
- Demand per day is estimated from recent history (mean and standard deviation); days are assumed independent, lead time constant, and
  demand roughly normal. Real sales here are right-skewed and bounded, so probabilities are screening scores, not exact chances.
- Minimum order quantity and pack size are applied only if you set them; shelf life and costs are not available and are ignored.
- Every figure is an ESTIMATE from a synthetic dataset. It is not a guarantee, and it does not show real business savings."""


# ================================================================ settings
@dataclass(frozen=True)
class InventoryParams:
    """User-configurable inventory settings (validated on creation)."""

    lead_time_days: float = 1.0
    service_level: float = 0.95
    review_period_days: float = 1.0        # R: days of demand an order should cover beyond the lead time
    lookback_days: int = 90                # history window used for demand mean/std
    min_history_days: int = 14             # fewer days than this -> "Insufficient history"
    safety_stock_units: float | None = None   # fixed override (units)
    safety_stock_days: float | None = None    # override as days of average demand
    excess_factor: float = 1.0             # "Excess" when position > excess_factor x target
    min_order_qty: float = 0.0
    order_multiple: float = 1.0

    def __post_init__(self):
        if self.lead_time_days <= 0:
            raise ValueError("lead_time_days must be greater than 0.")
        if not 0.5 <= self.service_level < 1:
            raise ValueError("service_level must be at least 0.5 and below 1 (for example 0.95).")
        if self.review_period_days < 0:
            raise ValueError("review_period_days cannot be negative.")
        if self.lookback_days < 2 or self.min_history_days < 2 or self.min_history_days > self.lookback_days:
            raise ValueError("lookback_days must be >= 2 and >= min_history_days (which must be >= 2).")
        for name in ("safety_stock_units", "safety_stock_days"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} cannot be negative.")
        if self.excess_factor < 1:
            raise ValueError("excess_factor must be at least 1 (Excess means above the target inventory).")
        if self.min_order_qty < 0 or self.order_multiple < 1:
            raise ValueError("min_order_qty must be >= 0 and order_multiple >= 1.")


# ================================================================ demand statistics at a snapshot
def demand_stats(cleaned: pd.DataFrame, snapshot_date=None, lookback_days: int = 90, min_history_days: int = 14) -> pd.DataFrame:
    """Per store-product: stock on hand at the snapshot day, and demand mean/std from the days BEFORE it."""
    if cleaned.empty:
        raise DataValidationError("The dataset is empty - nothing to analyse. Widen the filters.")
    for c in (DATE, *GROUP_KEYS, TARGET, "inventory_level"):
        if c not in cleaned.columns:
            raise DataValidationError(f"Column '{c}' is required for replenishment analysis.")
    first, last = cleaned[DATE].min(), cleaned[DATE].max()
    snap = pd.Timestamp(snapshot_date) if snapshot_date is not None else last
    if not first < snap <= last:
        raise DataValidationError(f"Snapshot date {snap.date()} must be after {first.date()} (history is needed) and not after {last.date()}.")
    today = cleaned[cleaned[DATE] == snap]
    if today.empty:
        raise DataValidationError(f"No inventory records exist on {snap.date()}.")
    hist = cleaned[(cleaned[DATE] < snap) & (cleaned[DATE] >= snap - pd.Timedelta(days=lookback_days))]
    agg = hist.groupby(GROUP_KEYS)[TARGET].agg(avg_daily_demand="mean", demand_std="std", history_days="count").reset_index()
    cols = [*GROUP_KEYS, "inventory_level"] + (["price"] if "price" in today.columns else [])
    out = today[cols].rename(columns={"inventory_level": "current_inventory"}).merge(agg, on=GROUP_KEYS, how="left")
    out["history_days"] = out["history_days"].fillna(0).astype(int)
    out["demand_std"] = out["demand_std"].fillna(0.0)
    out["avg_daily_demand"] = out["avg_daily_demand"].fillna(0.0)
    out["history_ok"] = out["history_days"] >= min_history_days
    out.insert(2, "snapshot_date", snap)
    return out.reset_index(drop=True)


# ================================================================ order quantity
def order_quantity(target_inventory, stock_on_hand, on_order, reorder_now, min_order_qty: float = 0.0, order_multiple: float = 1.0):
    """Suggested order = max(0, target - on hand - on order) when a reorder is triggered, else 0.

    A positive quantity is raised to the minimum order quantity (if any) and rounded UP to a whole number of packs.
    """
    gap = np.maximum(0.0, np.asarray(target_inventory, float) - np.asarray(stock_on_hand, float) - np.asarray(on_order, float))
    qty = np.where(np.asarray(reorder_now, bool), gap, 0.0)
    qty = np.where(qty > 0, np.maximum(qty, min_order_qty), 0.0)
    mult = max(float(order_multiple), 1.0)
    return np.where(qty > 0, np.ceil(qty / mult) * mult, 0.0).astype(np.int64)


# ================================================================ recommendations
def _on_order_series(stats: pd.DataFrame, on_order) -> np.ndarray:
    if on_order is None:
        return np.zeros(len(stats))
    if np.isscalar(on_order):
        if on_order < 0:
            raise ValueError("on_order cannot be negative.")
        return np.full(len(stats), float(on_order))
    oo = pd.DataFrame(on_order)
    need = {*GROUP_KEYS, "on_order"}
    if not need <= set(oo.columns):
        raise DataValidationError(f"on_order table needs columns {sorted(need)}.")
    if (oo["on_order"] < 0).any():
        raise ValueError("on_order cannot be negative.")
    merged = stats[GROUP_KEYS].merge(oo.groupby(GROUP_KEYS, as_index=False)["on_order"].sum(), on=GROUP_KEYS, how="left")
    return merged["on_order"].fillna(0.0).to_numpy()


def compute_recommendations(stats: pd.DataFrame, params: InventoryParams, on_order=None) -> pd.DataFrame:
    """Apply the policy to the per-series statistics. Returns one row per store-product, most urgent first."""
    if stats.empty:
        raise DataValidationError("No store-product records to analyse.")
    p, L = params, params.lead_time_days
    r = stats.copy()
    d, sd = r["avg_daily_demand"].to_numpy(float), r["demand_std"].to_numpy(float)
    r["on_order"] = _on_order_series(r, on_order)
    r["lead_time_days"] = L
    r["lead_time_demand"] = lead_time_demand(d, L)
    if p.safety_stock_units is not None:
        r["safety_stock"] = float(p.safety_stock_units)
    elif p.safety_stock_days is not None:
        r["safety_stock"] = p.safety_stock_days * d
    else:
        r["safety_stock"] = safety_stock(sd, L, p.service_level)
    r["reorder_point"] = r["lead_time_demand"] + r["safety_stock"]
    r["target_inventory"] = r["reorder_point"] + d * p.review_period_days
    r["inventory_position"] = r["current_inventory"] + r["on_order"]
    r["projected_stock_after_lead_time"] = r["inventory_position"] - r["lead_time_demand"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r["days_of_cover"] = np.where(d > 0, r["inventory_position"] / d, np.inf)
    r["stockout_probability"] = stockout_probability(r["inventory_position"], d, sd, L)
    r["risk_level"] = classify_risk(r["stockout_probability"], p.service_level)
    r["reorder_now"] = (r["inventory_position"] <= r["reorder_point"]) & r["history_ok"]
    r["suggested_order_qty"] = order_quantity(r["target_inventory"], r["current_inventory"], r["on_order"], r["reorder_now"],
                                              p.min_order_qty, p.order_multiple)
    status = np.select([r["inventory_position"] <= r["reorder_point"], r["inventory_position"] > p.excess_factor * r["target_inventory"]],
                       ["Low stock", "Excess stock"], default="Adequate stock")
    r["stock_status"] = status
    ok = r["history_ok"].to_numpy()
    r.loc[~ok, "stock_status"] = "Insufficient history"
    r["risk_level"] = r["risk_level"].astype(object)
    r.loc[~ok, "risk_level"] = "Unknown"
    r["risk_level"] = pd.Categorical(r["risk_level"], categories=[*RISK_LEVELS, "Unknown"], ordered=True)

    def text(row):
        if not row["history_ok"]:
            return f"Only {row['history_days']} days of history - cannot estimate demand"
        if row["reorder_now"]:
            return f"Order {row['suggested_order_qty']:,} units now ({row['risk_level']} risk)"
        if row["stock_status"] == "Excess stock":
            return "Overstocked - do not reorder"
        return "No order needed"

    r["recommendation"] = r.apply(text, axis=1)
    r = r.sort_values(["reorder_now", "stockout_probability", "projected_stock_after_lead_time"], ascending=[False, False, True])
    ordered = [*GROUP_KEYS, "snapshot_date", "current_inventory", "on_order", "inventory_position", "avg_daily_demand", "demand_std",
               "lead_time_days", "lead_time_demand", "safety_stock", "reorder_point", "target_inventory",
               "projected_stock_after_lead_time", "days_of_cover", "stockout_probability", "risk_level", "stock_status",
               "reorder_now", "suggested_order_qty", "recommendation", "history_days"]
    return r[ordered].reset_index(drop=True)


def recommend(cleaned: pd.DataFrame, params: InventoryParams = InventoryParams(), snapshot_date=None, on_order=None) -> pd.DataFrame:
    """Convenience wrapper: demand statistics at the snapshot, then recommendations."""
    stats = demand_stats(cleaned, snapshot_date, params.lookback_days, params.min_history_days)
    return compute_recommendations(stats, params, on_order)


def summarize(recs: pd.DataFrame) -> dict:
    """KPI numbers for the dashboard and the text report."""
    return {"series": int(len(recs)), "reorder_now": int(recs["reorder_now"].sum()),
            "share_reorder": float(recs["reorder_now"].mean()), "units_to_order": int(recs["suggested_order_qty"].sum()),
            "risk_counts": {k: int((recs["risk_level"] == k).sum()) for k in [*RISK_LEVELS, "Unknown"]},
            "status_counts": recs["stock_status"].value_counts().to_dict(),
            "median_days_of_cover": float(np.median(recs.loc[np.isfinite(recs["days_of_cover"]), "days_of_cover"])) if np.isfinite(recs["days_of_cover"]).any() else float("nan"),
            "mean_stockout_probability": float(recs["stockout_probability"].mean())}


# ================================================================ sensitivity and back-test
def lead_time_sensitivity(cleaned: pd.DataFrame, params: InventoryParams = InventoryParams(), snapshot_date=None,
                          lead_times=(0.5, 1, 2, 3, 5, 7)) -> pd.DataFrame:
    """How the recommendations change with the lead-time assumption (the one input the data cannot supply)."""
    stats = demand_stats(cleaned, snapshot_date, params.lookback_days, params.min_history_days)
    rows = []
    for L in lead_times:
        r = compute_recommendations(stats, replace(params, lead_time_days=float(L)))
        rows.append({"lead_time_days": L, "pct_low_stock": 100 * (r["stock_status"] == "Low stock").mean(),
                     "pct_critical_or_high": 100 * r["risk_level"].isin(["Critical", "High"]).mean(),
                     "mean_reorder_point": r["reorder_point"].mean(), "mean_current_inventory": r["current_inventory"].mean(),
                     "total_units_to_order": int(r["suggested_order_qty"].sum())})
    return pd.DataFrame(rows)


def service_level_backtest(cleaned: pd.DataFrame, params: InventoryParams = InventoryParams(), test_days: int = 90) -> dict:
    """Does the safety-stock formula deliver the promised service level on data it has not seen?

    For every series: mean/std from the `lookback_days` before the test period -> reorder point (without overrides).
    Then every L-day window inside the test period is checked: was the demand within the reorder point?
    The share of windows that were covered is the achieved cycle service level (target = params.service_level).
    """
    L = int(math.ceil(params.lead_time_days))
    last = cleaned[DATE].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    hist = cleaned[(cleaned[DATE] < test_start) & (cleaned[DATE] >= test_start - pd.Timedelta(days=params.lookback_days))]
    if hist.empty or cleaned[cleaned[DATE] >= test_start].empty:
        raise DataValidationError("Not enough data for a back-test: need history before the test period and data inside it.")
    st = hist.groupby(GROUP_KEYS)[TARGET].agg(m="mean", s="std").fillna(0.0)
    st["rop"] = reorder_point(st["m"], st["s"], L, params.service_level)[2]
    covered, total, per_series = 0, 0, []
    for key, g in cleaned[cleaned[DATE] >= test_start - pd.Timedelta(days=L - 1)].groupby(GROUP_KEYS):
        if key not in st.index:
            continue
        s = g.set_index(DATE)[TARGET].astype(float).asfreq("D")
        win = s.rolling(L, min_periods=L).sum()
        win = win[win.index >= test_start + pd.Timedelta(days=L - 1)].dropna()
        if win.empty:
            continue
        ok = (win <= st.loc[key, "rop"]).sum()
        covered, total = covered + ok, total + len(win)
        per_series.append(ok / len(win))
    if total == 0:
        raise DataValidationError("No complete windows available for the back-test.")
    return {"lead_time_days_used": L, "target_service_level": params.service_level, "achieved_service_level": covered / total,
            "windows": int(total), "series": len(per_series), "series_min": float(np.min(per_series)), "series_max": float(np.max(per_series)),
            "test_period": [str(test_start.date()), str(last.date())]}


POLICY_SIM_ASSUMPTIONS = (
    "Demand = the recorded units sold of each day (real demand may be higher on days when stock ran out: the file never shows lost sales).",
    "Mean and standard deviation of daily demand are estimated only from the days BEFORE the test period; nothing from the test period is used to set the policy.",
    "Every series starts the test period with stock equal to its target inventory; an order placed at the end of a day arrives after the lead time (whole days, rounded up).",
    "The policy is the same (s, S) rule as the recommendations: order up to the target when the inventory position is at or below the reorder point.",
    "Unmet demand is lost (no back-orders). Supplier delays, order costs and holding costs are not modelled.",
    "This is a simulation on one synthetic file: it illustrates how the policy behaves, it does not prove real-world performance.",
)


def simulate_policy(demand: np.ndarray, mean: np.ndarray, std: np.ndarray, lead_days: int, service_level: float,
                    review_period_days: float = 1.0, min_order_qty: float = 0.0, order_multiple: float = 1.0) -> dict:
    """Run the (s, S) policy day by day over a demand matrix (rows = series, columns = days). Returns total metrics.

    End-of-day sequence: receive deliveries due today -> sell min(stock, demand) -> compute the inventory position
    (stock + on order) -> if position <= ROP order (target - position), which arrives `lead_days` later.
    """
    demand = np.asarray(demand, dtype=float)
    if demand.ndim != 2 or demand.shape[1] < 1:
        raise DataValidationError("The simulation needs a (series x days) demand matrix with at least one day.")
    n, days = demand.shape
    L = max(int(lead_days), 1)
    rop = mean * L + z_value(service_level) * std * math.sqrt(L)
    target = rop + mean * review_period_days
    on_hand, on_order = target.copy(), np.zeros(n)
    arrivals = np.zeros((n, days + L + 1))
    sold_total = lost_total = stock_sum = 0.0
    stockout_days = orders = 0
    for t in range(days):
        on_hand += arrivals[:, t]
        on_order -= arrivals[:, t]
        sold = np.minimum(on_hand, demand[:, t])
        lost = demand[:, t] - sold
        on_hand -= sold
        position = on_hand + on_order
        trigger = position <= rop + 1e-9
        qty = order_quantity(target, on_hand, on_order, trigger, min_order_qty, order_multiple).astype(float)
        arrivals[:, t + L] += qty
        on_order += qty
        orders += int((qty > 0).sum())
        sold_total, lost_total, stock_sum = sold_total + sold.sum(), lost_total + lost.sum(), stock_sum + on_hand.sum()
        stockout_days += int((lost > 1e-9).sum())
    demand_total = sold_total + lost_total
    return {"service_level_target": float(service_level), "fill_rate": float(sold_total / demand_total) if demand_total > 0 else float("nan"),
            "stockout_day_frequency": float(stockout_days / (n * days)), "avg_on_hand_units": float(stock_sum / (n * days)),
            "orders_per_series": float(orders / n), "units_demanded": float(demand_total), "units_lost": float(lost_total),
            "series": int(n), "days": int(days)}


def policy_backtest(cleaned: pd.DataFrame, params: InventoryParams = InventoryParams(), test_days: int = 90,
                    levels: tuple[float, ...] = (0.80, 0.90, 0.95, 0.99)) -> dict:
    """Simulate the replenishment policy on the last `test_days` days for several service-level targets.

    Returns {"table": DataFrame (one row per target level), "chosen": row for params.service_level, "test_period", "assumptions"}.
    """
    if cleaned.empty:
        raise DataValidationError("The dataset is empty - nothing to simulate.")
    last = cleaned[DATE].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    hist = cleaned[(cleaned[DATE] < test_start) & (cleaned[DATE] >= test_start - pd.Timedelta(days=params.lookback_days))]
    test = cleaned[cleaned[DATE] >= test_start]
    if hist.empty or test.empty:
        raise DataValidationError("Not enough data for the policy simulation: need history before the test period and data inside it.")
    st = hist.groupby(GROUP_KEYS)[TARGET].agg(m="mean", s="std").fillna(0.0)
    mat = test.pivot_table(index=GROUP_KEYS, columns=DATE, values=TARGET, aggfunc="sum").reindex(columns=pd.date_range(test_start, last)).fillna(0.0)
    mat = mat.loc[mat.index.intersection(st.index)]
    if mat.empty:
        raise DataValidationError("No store-product has both history and test-period data.")
    st = st.loc[mat.index]
    L = int(math.ceil(params.lead_time_days))
    wanted = sorted({round(float(x), 4) for x in (*levels, params.service_level)})
    rows = [simulate_policy(mat.to_numpy(), st["m"].to_numpy(), st["s"].to_numpy(), L, lv, params.review_period_days,
                            params.min_order_qty, params.order_multiple) for lv in wanted]
    table = pd.DataFrame(rows)
    chosen = table.loc[(table["service_level_target"] - params.service_level).abs().idxmin()].to_dict()
    return {"table": table, "chosen": chosen, "lead_time_days_used": L, "test_period": [str(test_start.date()), str(last.date())],
            "assumptions": list(POLICY_SIM_ASSUMPTIONS)}


# ================================================================ plots (each returns a Figure)
RISK_COLORS = {"Low": "#54A24B", "Medium": "#EECA3B", "High": "#F58518", "Critical": "#E45756", "Unknown": "#9D9D9D"}
STATUS_COLORS = {"Low stock": "#E45756", "Adequate stock": "#54A24B", "Excess stock": "#4C78A8", "Insufficient history": "#9D9D9D"}


def plot_risk_overview(recs: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(12, 4.8), layout="constrained")
    a, b = fig.subplots(1, 2)
    rc = recs["risk_level"].value_counts().reindex([*RISK_LEVELS, "Unknown"]).fillna(0)
    rc = rc[rc > 0] if (rc > 0).any() else rc
    bars = a.bar(rc.index.astype(str), rc.to_numpy(), color=[RISK_COLORS[str(k)] for k in rc.index])
    a.bar_label(bars, padding=2)
    a.set(title="Stock-out risk level (rule-based estimate)", xlabel="Risk level", ylabel="Store-products")
    sc = recs["stock_status"].value_counts()
    bars = b.bar(sc.index, sc.to_numpy(), color=[STATUS_COLORS.get(k, "#9D9D9D") for k in sc.index])
    b.bar_label(bars, padding=2)
    b.set(title="Stock status", xlabel="Status", ylabel="Store-products")
    return fig


def plot_position_vs_rop(recs: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(12, 5), layout="constrained")
    a, b = fig.subplots(1, 2)
    for status, g in recs.groupby("stock_status"):
        a.scatter(g["reorder_point"], g["inventory_position"], s=28, alpha=0.8, color=STATUS_COLORS.get(status, "#9D9D9D"), label=status)
    lim = max(recs["reorder_point"].max(), recs["inventory_position"].max()) * 1.05
    a.plot([0, lim], [0, lim], color="black", ls="--", lw=1, label="Position = reorder point")
    a.set(title="Inventory position vs reorder point", xlabel="Reorder point (units)", ylabel="Inventory position (units)")
    a.legend(fontsize=8)
    b.hist(recs["projected_stock_after_lead_time"], bins=25, color="#4C78A8")
    b.axvline(0, color="#E45756", ls="--", label="Expected stock-out (below 0)")
    b.set(title="Projected stock after lead-time demand", xlabel="Position - expected lead-time demand (units)", ylabel="Store-products")
    b.legend()
    return fig


def plot_lead_time_sensitivity(sens: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(12, 4.8), layout="constrained")
    a, b = fig.subplots(1, 2)
    a.plot(sens["lead_time_days"], sens["pct_low_stock"], marker="o", label="% in Low-stock status")
    a.plot(sens["lead_time_days"], sens["pct_critical_or_high"], marker="s", label="% at High or Critical risk")
    a.set(title="Effect of the lead-time assumption", xlabel="Lead time (days)", ylabel="Share of store-products (%)", ylim=(0, 105))
    a.legend()
    b.plot(sens["lead_time_days"], sens["mean_reorder_point"], marker="o", label="Mean reorder point")
    b.axhline(sens["mean_current_inventory"].iloc[0], color="#E45756", ls="--", label="Mean stock on hand")
    b.set(title="Reorder point grows faster than stock on hand", xlabel="Lead time (days)", ylabel="Units")
    b.legend()
    return fig


# ================================================================ report + CLI
def build_report(recs: pd.DataFrame, params: InventoryParams, sens: pd.DataFrame, backtest: dict | None) -> str:
    k = summarize(recs)
    fmt = lambda t: t.to_string(index=False, float_format=lambda v: f"{v:,.2f}")  # noqa: E731
    snap = recs["snapshot_date"].iloc[0].date()
    lines = [f"REPLENISHMENT RECOMMENDATIONS - snapshot at the start of {snap}", "=" * 78,
             "Settings: " + ", ".join(f"{a}={b}" for a, b in asdict(params).items() if b is not None),
             f"z (service level {params.service_level:.0%}) = {z_value(params.service_level):.3f}", "",
             f"Store-products analysed: {k['series']} | to reorder now: {k['reorder_now']} ({k['share_reorder']:.0%}) | total suggested units: {k['units_to_order']:,}",
             f"Risk levels: {k['risk_counts']}", f"Stock status: {k['status_counts']}",
             f"Median days of cover: {k['median_days_of_cover']:.2f} | mean stock-out probability: {k['mean_stockout_probability']:.1%}", "",
             RISK_RULES_TEXT, "",
             "TOP 10 MOST URGENT", fmt(recs.head(10)[[*GROUP_KEYS, "current_inventory", "reorder_point", "target_inventory",
                                                     "stockout_probability", "risk_level", "suggested_order_qty"]]), "",
             "LEAD-TIME SENSITIVITY (the lead time is a setting, not data)", fmt(sens), ""]
    low_share = float(sens.loc[sens["lead_time_days"] == sens["lead_time_days"].max(), "pct_low_stock"].iloc[0])
    if low_share > 90:
        lines.append(f"FINDING: with a long lead time {low_share:.0f}% of items are below their reorder point: stock on hand is only about "
                     f"{k['median_days_of_cover']:.1f} days of demand in this dataset, so any lead time of a few days makes every item look urgent. "
                     "This is a property of the synthetic data, not a recommendation to order everything.")
        lines.append("")
    if backtest:
        gap = backtest["achieved_service_level"] - backtest["target_service_level"]
        lines += ["BACK-TEST OF THE SAFETY-STOCK FORMULA (data the formula had not seen)",
                  f"Test period {backtest['test_period'][0]}..{backtest['test_period'][1]}, lead time {backtest['lead_time_days_used']} day(s), {backtest['windows']:,} windows over {backtest['series']} series.",
                  f"Target service level {backtest['target_service_level']:.1%} vs achieved {backtest['achieved_service_level']:.1%} ({gap:+.1%}); per-series range "
                  f"{backtest['series_min']:.1%}..{backtest['series_max']:.1%}. "
                  + ("The normal-demand assumption slightly overstates protection." if gap < -0.01 else
                     "The formula delivers about the promised protection." if abs(gap) <= 0.01 else "The formula is slightly conservative."), ""]
    lines.append(ASSUMPTIONS)
    return "\n".join(lines) + "\n"


def run_reorder_report(cleaned: pd.DataFrame, params: InventoryParams, snapshot_date=None, reports_dir: Path = REPORTS_DIR,
                       on_order=None) -> dict:
    ensure_dirs()
    recs = recommend(cleaned, params, snapshot_date, on_order)
    sens = lead_time_sensitivity(cleaned, params, snapshot_date)
    try:
        bt = service_level_backtest(cleaned, params)
    except DataValidationError:
        bt = None
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    recs.to_csv(reports_dir / "reorder_recommendations.csv", index=False)
    sens.to_csv(reports_dir / "lead_time_sensitivity.csv", index=False)
    (reports_dir / "reorder_summary.txt").write_text(build_report(recs, params, sens, bt), encoding="utf-8")
    plot_risk_overview(recs).savefig(fig_dir / "reorder_risk_overview.png", dpi=120)
    plot_position_vs_rop(recs).savefig(fig_dir / "reorder_position_vs_rop.png", dpi=120)
    plot_lead_time_sensitivity(sens).savefig(fig_dir / "reorder_lead_time_sensitivity.png", dpi=120)
    return {"recommendations": recs, "sensitivity": sens, "backtest": bt, "summary": summarize(recs)}


def main() -> None:
    p = argparse.ArgumentParser(description="Rule-based stock-out risk and replenishment recommendations.")
    p.add_argument("--input", default=str(CLEANED_CSV))
    p.add_argument("--snapshot", default=None, help="date (YYYY-MM-DD); default = last date in the data")
    p.add_argument("--lead-time", type=float, default=1.0, help="supplier lead time in days")
    p.add_argument("--service-level", type=float, default=0.95)
    p.add_argument("--review-days", type=float, default=1.0, help="days of demand one order should cover beyond the lead time")
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--safety-stock-units", type=float, default=None)
    p.add_argument("--safety-stock-days", type=float, default=None)
    p.add_argument("--excess-factor", type=float, default=1.0)
    p.add_argument("--min-order-qty", type=float, default=0.0)
    p.add_argument("--order-multiple", type=float, default=1.0)
    p.add_argument("--reports-dir", default=str(REPORTS_DIR))
    a = p.parse_args()
    try:
        params = InventoryParams(a.lead_time, a.service_level, a.review_days, a.lookback, 14, a.safety_stock_units,
                                 a.safety_stock_days, a.excess_factor, a.min_order_qty, a.order_multiple)
        out = run_reorder_report(load_cleaned(a.input), params, a.snapshot, Path(a.reports_dir))
    except (DataFileError, DataValidationError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    k = out["summary"]
    log.info("Snapshot %s | %d store-products | reorder now: %d | units to order: %s | risk: %s",
             out["recommendations"]["snapshot_date"].iloc[0].date(), k["series"], k["reorder_now"], f"{k['units_to_order']:,}",
             {a: b for a, b in k["risk_counts"].items() if b})
    log.info("Saved reports/reorder_recommendations.csv, reorder_summary.txt, lead_time_sensitivity.csv and 3 figures")


if __name__ == "__main__":
    main()
