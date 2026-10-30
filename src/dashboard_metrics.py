"""Calculations behind the dashboard (no Streamlit): inventory health, KPIs, insights, tables and statistical tests.

Everything is computed from the data passed in; no sentence or number is hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import eda
from src.preprocess import GROUP_KEYS
from src.utils import DataValidationError

HEALTH_LEVELS = ["Healthy", "Low", "Critical", "Excess", "Not assessed"]
HEALTH_RULES = (
    "Each store-product is placed in exactly ONE group, checked in this order: "
    "**Not assessed** (fewer than the minimum days of demand history) -> **Excess** (inventory position above the excess threshold x target inventory) -> "
    "**Critical** (estimated stock-out probability of 50% or more during the lead time) -> **Low** (inventory position at or below the reorder point) -> **Healthy**."
)


# ================================================================ inventory health and KPIs
def inventory_health(recs: pd.DataFrame) -> pd.Series:
    """Mutually exclusive health group per row of the recommendation table (see HEALTH_RULES)."""
    need = {"stock_status", "risk_level"}
    if not need <= set(recs.columns):
        raise DataValidationError(f"Recommendation table needs the columns {sorted(need)}.")
    status, risk = recs["stock_status"].astype(str), recs["risk_level"].astype(str)
    out = np.select([status == "Insufficient history", status == "Excess stock", risk == "Critical", status == "Low stock"],
                    ["Not assessed", "Excess", "Critical", "Low"], default="Healthy")
    return pd.Series(pd.Categorical(out, categories=HEALTH_LEVELS, ordered=True), index=recs.index, name="inventory_health")


def health_counts(recs: pd.DataFrame) -> pd.DataFrame:
    """Counts and shares per health group (always all groups, zeros included; counts add up to len(recs))."""
    h = inventory_health(recs)
    counts = h.value_counts().reindex(HEALTH_LEVELS).fillna(0).astype(int)
    total = int(counts.sum())
    return pd.DataFrame({"group": HEALTH_LEVELS, "store_products": counts.to_numpy(),
                         "share": (counts / total).to_numpy() if total else np.zeros(len(HEALTH_LEVELS))})


def attach_price(recs: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Add the recorded `price` of each store-product on the snapshot day (NaN if the file has no price)."""
    out = recs.copy()
    if "price" not in cleaned.columns or recs.empty:
        out["price"] = np.nan
        return out
    snap = recs["snapshot_date"].iloc[0]
    today = cleaned.loc[cleaned["date"] == snap, [*GROUP_KEYS, "price"]].drop_duplicates(GROUP_KEYS)
    for c in GROUP_KEYS:
        out[c] = out[c].astype(str)
        today[c] = today[c].astype(str)
    return out.drop(columns=["price"], errors="ignore").merge(today, on=GROUP_KEYS, how="left")


def estimated_inventory_value(recs: pd.DataFrame) -> float | None:
    """Sum of stock on hand x recorded price. An ESTIMATE based on the dataset's price field (not an acquisition cost)."""
    if "price" not in recs.columns or recs["price"].notna().sum() == 0:
        return None
    return float((recs["current_inventory"] * recs["price"]).sum(skipna=True))


def home_kpis(view: pd.DataFrame, recs: pd.DataFrame | None) -> dict:
    """KPI numbers. Period figures come from the filtered rows; stock figures from the snapshot table."""
    out = {"products": int(view["product_id"].nunique()), "stores": int(view["store_id"].nunique()),
           "units_sold": float(view["units_sold"].sum()), "records": int(len(view)), "days": int(view["date"].nunique())}
    if recs is not None and not recs.empty:
        counts = health_counts(recs).set_index("group")["store_products"]
        out.update({"inventory_units": float(recs["current_inventory"].sum()), "inventory_value": estimated_inventory_value(recs),
                    "reorder_now": int(recs["reorder_now"].sum()), "series": int(len(recs)),
                    "critical": int(counts["Critical"]), "excess": int(counts["Excess"]), "healthy": int(counts["Healthy"]),
                    "low": int(counts["Low"]), "not_assessed": int(counts["Not assessed"]),
                    "units_to_order": int(recs["suggested_order_qty"].sum())})
    return out


# ================================================================ trends
def trend_frame(view: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Per period: total units sold, total stock on hand (mean of daily totals), mean daily demand per store-product, sell-through.

    Weekly/monthly periods that are incomplete in the selection are dropped (a partial week would look like a crash).
    """
    if view.empty:
        raise DataValidationError("No data to build trends from.")
    daily = view.groupby("date").agg(units_sold=("units_sold", "sum"), inventory=("inventory_level", "sum"), series=("store_id", "size"))
    daily = daily.asfreq("D")
    if freq == "D":
        res = daily.copy()
    else:
        rule = "W-SUN" if freq == "W" else "MS"
        r = daily.resample(rule)
        res = pd.DataFrame({"units_sold": r["units_sold"].sum(min_count=1), "inventory": r["inventory"].mean(), "series": r["series"].mean(),
                            "n_days": r["units_sold"].count()})
        expected = 7 if freq == "W" else res.index.days_in_month
        res = res[res["n_days"] >= expected].drop(columns="n_days")
    div = 1 if freq == "D" else (7 if freq == "W" else res.index.days_in_month)   # days in each period
    res["sold_per_day"] = res["units_sold"] / div
    res["demand_per_series_day"] = res["sold_per_day"] / res["series"]
    res["sell_through"] = res["sold_per_day"] / res["inventory"]
    return res.dropna(subset=["units_sold"]).reset_index(names="date")


def category_demand(view: pd.DataFrame) -> pd.DataFrame:
    if "category" not in view.columns:
        raise DataValidationError("The data has no category column.")
    g = view.groupby("category", observed=True)["units_sold"].agg(total_units="sum", mean_units="mean", rows="count").reset_index()
    g["category"] = g["category"].astype(str)
    g["share"] = g["total_units"] / g["total_units"].sum()
    return g.sort_values("total_units", ascending=False).reset_index(drop=True)


# ================================================================ insights (all computed)
def demand_variation(view: pd.DataFrame, min_days: int = 14) -> pd.DataFrame:
    """Coefficient of variation (std / mean of daily units sold) per store-product series."""
    g = view.groupby(GROUP_KEYS)["units_sold"].agg(mean="mean", std="std", days="count").reset_index()
    g = g[(g["days"] >= min_days) & (g["mean"] > 0)].copy()
    g["cv"] = g["std"] / g["mean"]
    return g.sort_values("cv", ascending=False).reset_index(drop=True)


def _series_label(row) -> str:
    return f"{row['store_id']} / {row['product_id']}"


def generate_actionable_insights(view: pd.DataFrame, recs: pd.DataFrame | None) -> list[dict]:
    """Insight cards {title, detail, tone} computed from the filtered data and the snapshot table. Empty input -> no cards."""
    out: list[dict] = []
    if view.empty:
        return out
    overall = float(view["units_sold"].mean())
    prod = view.groupby("product_id")["units_sold"].agg(mean="mean", total="sum").sort_values("mean", ascending=False)
    if len(prod):
        top = prod.head(3)
        text = ", ".join(f"{p} ({r['mean']:.1f}/store-day, {100 * (r['mean'] / overall - 1):+.1f}% vs average)" for p, r in top.iterrows())
        note = ""
        try:
            if view["product_id"].nunique() >= 3 and eda.compare_many_groups(view, "units_sold", "product_id")["anova_p"] >= eda.ALPHA:
                note = " Differences between products are not statistically significant, so treat the ranking with caution."
        except DataValidationError:
            pass
        out.append({"title": "Highest-demand products", "detail": text + "." + note, "tone": "blue"})
    if "category" in view.columns:
        cat = category_demand(view)
        if len(cat):
            c = cat.iloc[0]
            out.append({"title": "Top-selling category", "detail": f"{c['category']} sold {c['total_units']:,.0f} units ({c['share']:.0%} of the selected total).", "tone": "blue"})
    var = demand_variation(view)
    if len(var):
        v = var.head(3)
        out.append({"title": "Most variable demand", "detail": "Highest coefficient of variation: " + ", ".join(f"{_series_label(r)} (CV {r['cv']:.2f})" for _, r in v.iterrows())
                    + f". The typical series has CV {var['cv'].median():.2f}.", "tone": "purple"})
    if recs is not None and not recs.empty:
        h = inventory_health(recs)
        crit = recs[h == "Critical"].sort_values("stockout_probability", ascending=False)
        low = recs[h.isin(["Low", "Critical"])]
        if len(crit):
            worst = ", ".join(f"{_series_label(r)} ({r['stockout_probability']:.0%})" for _, r in crit.head(3).iterrows())
            out.append({"title": "Needs attention now", "detail": f"{len(crit)} of {len(recs)} store-products have a Critical estimated stock-out risk. Highest: {worst}.", "tone": "red"})
        else:
            out.append({"title": "Needs attention now", "detail": "No store-product has a Critical estimated stock-out risk under the current settings.", "tone": "green"})
        if len(low):
            agg = low.groupby("product_id").agg(pos=("inventory_position", "sum"), dem=("avg_daily_demand", "sum"), n=("store_id", "size"))
            agg = agg[agg["dem"] > 0].assign(cover=lambda d: d["pos"] / d["dem"]).sort_values("cover").head(3)
            detail = f"{len(low)} store-products are at or below their reorder point."
            if len(agg):
                detail += " Lowest days of cover by product: " + ", ".join(f"{p} ({r['cover']:.1f} days, {int(r['n'])} stores)" for p, r in agg.iterrows()) + "."
            out.append({"title": "Low stock", "detail": detail, "tone": "amber"})
        else:
            out.append({"title": "Low stock", "detail": "No store-product is at or below its reorder point under the current settings.", "tone": "green"})
        exc = recs[h == "Excess"]
        if len(exc):
            ratio = (exc["inventory_position"] / exc["target_inventory"].replace(0, np.nan)).sort_values(ascending=False)
            worst = ", ".join(f"{_series_label(recs.loc[i])} ({ratio[i]:.1f}x target)" for i in ratio.dropna().head(3).index)
            out.append({"title": "Excess stock", "detail": f"{len(exc)} store-products hold more than the excess threshold x target inventory. Highest: {worst}.", "tone": "blue"})
        else:
            out.append({"title": "Excess stock", "detail": "No store-product is above the excess threshold under the current settings.", "tone": "green"})
    return out


# ================================================================ critical items table
CRITICAL_COLUMNS = {"store_id": "Store ID", "product_id": "Product ID", "category": "Category", "current_inventory": "Current inventory",
                    "reorder_point": "Reorder point", "stockout_probability": "Stock-out risk", "risk_level": "Risk level",
                    "suggested_order_qty": "Suggested order qty", "recommendation": "Recommendation"}


def critical_items(recs: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Most urgent items. Order: items that must be reordered first, then higher stock-out probability, then lower projected stock after the lead time."""
    if recs.empty:
        return pd.DataFrame(columns=list(CRITICAL_COLUMNS.values()))
    r = recs.sort_values(["reorder_now", "stockout_probability", "projected_stock_after_lead_time"], ascending=[False, False, True]).head(n)
    cols = [c for c in CRITICAL_COLUMNS if c in r.columns]
    out = r[cols].copy()
    out["risk_level"] = out["risk_level"].astype(str)
    return out.rename(columns=CRITICAL_COLUMNS).reset_index(drop=True)


# ================================================================ dataset explorer helpers
COLUMN_DESCRIPTIONS = {
    "date": "Calendar day of the record", "store_id": "Store identifier", "product_id": "Product identifier", "category": "Product category (varies by row)",
    "region": "Store region (varies by row)", "inventory_level": "Units in stock that day", "units_sold": "Units sold that day (the demand being forecast)",
    "units_ordered": "Units ordered from the supplier that day", "demand_forecast": "Forecast supplied in the file (leaks the answer; not used by models)",
    "price": "Selling price", "discount": "Discount percentage", "weather": "Weather condition", "holiday_promo": "1 if a holiday or promotion applied",
    "competitor_price": "Competitor's price", "seasonality": "Season label from the file",
    "outlier_units_sold": "Flag added by cleaning: unusual units sold for its store-product", "outlier_inventory_level": "Flag added by cleaning: unusual inventory level for its store-product",
}


def column_info(df: pd.DataFrame, n_examples: int = 3) -> pd.DataFrame:
    """Name, type, description, missing, unique and example values of every column (from the actual data)."""
    rows = []
    for c in df.columns:
        s = df[c]
        examples = s.dropna().astype(str).drop_duplicates().head(n_examples).tolist()
        if pd.api.types.is_datetime64_any_dtype(s):
            examples = [str(pd.Timestamp(v).date()) for v in s.dropna().drop_duplicates().head(n_examples)]
        rows.append({"column": c, "type": str(s.dtype), "description": COLUMN_DESCRIPTIONS.get(c, ""), "missing": int(s.isna().sum()),
                     "missing %": round(100 * s.isna().mean(), 2) if len(s) else 0.0, "unique": int(s.nunique()), "examples": ", ".join(examples)})
    return pd.DataFrame(rows)


def paginate(df: pd.DataFrame, page: int, page_size: int) -> tuple[pd.DataFrame, int]:
    """(rows of the 1-based `page`, number of pages). A page past the end is clamped to the last page."""
    if page_size < 1:
        raise ValueError("page_size must be at least 1.")
    pages = max(1, -(-len(df) // page_size))
    page = min(max(int(page), 1), pages)
    return df.iloc[(page - 1) * page_size: page * page_size], pages


def sort_table(df: pd.DataFrame, column: str | None, ascending: bool = True) -> pd.DataFrame:
    if not column or column not in df.columns:
        return df
    return df.sort_values(column, ascending=ascending, kind="mergesort")


def numeric_range_filter(df: pd.DataFrame, column: str, low: float, high: float) -> pd.DataFrame:
    if column not in df.columns:
        raise DataValidationError(f"Column '{column}' is not in the data.")
    if low > high:
        raise ValueError("The lower bound is above the upper bound.")
    return df[df[column].between(low, high)]


def duplicate_analysis(df: pd.DataFrame) -> dict:
    """Duplicate checks on the cleaned data: identical rows, and repeated (date, store, product) keys."""
    keys = [c for c in ("date", *GROUP_KEYS) if c in df.columns]
    return {"rows": int(len(df)), "exact_duplicates": int(df.duplicated().sum()), "duplicate_keys": int(df.duplicated(subset=keys).sum()) if keys else 0,
            "series": int(df.groupby(GROUP_KEYS, observed=True).ngroups) if set(GROUP_KEYS) <= set(df.columns) else 0}


# ================================================================ hypothesis tests in one table
def _verdict(p: float, h0_text: str, h1_text: str) -> str:
    if np.isnan(p):
        return "Could not be evaluated."
    return (f"Reject H0 at 5%: {h1_text}" if p < eda.ALPHA else f"Fail to reject H0: no evidence that {h0_text}")


def hypothesis_test_table(view: pd.DataFrame) -> pd.DataFrame:
    """Test name, H0, H1, statistic, p-value and a plain interpretation for the main tests on units sold."""
    rows = []

    def add(name, h0, h1, stat_label, stat, p, h0_text, h1_text, note=""):
        rows.append({"Test": name, "Null hypothesis (H0)": h0, "Alternative (H1)": h1, "Statistic": f"{stat_label} = {stat:.3f}" if stat == stat else "n/a",
                     "P-value": float(p), "Interpretation": _verdict(p, h0_text, h1_text) + (f" {note}" if note else "")})

    try:
        r = eda.compare_two_groups(view)
        add("Welch t-test: Holiday/Promotion flag", "Mean units sold is equal with and without the flag", "The means differ", "t", r["t_stat"], r["welch_p"],
            "the flag changes mean sales", "the flag is associated with different mean sales.", f"Cohen's d = {r['cohens_d']:.3f}.")
        add("Mann-Whitney U: Holiday/Promotion flag", "Both groups have the same distribution of units sold", "The distributions differ", "U", r["mannwhitney_u"], r["mannwhitney_p"],
            "the distributions differ", "the distributions differ.")
    except DataValidationError:
        pass
    for grp, label in (("store_id", "store"), ("product_id", "product"), ("category", "category"), ("region", "region"), ("discount", "discount level")):
        if grp not in view.columns:
            continue
        try:
            g = eda.compare_many_groups(view, "units_sold", grp)
        except DataValidationError:
            continue
        add(f"One-way ANOVA: by {label}", f"Mean units sold is equal across {label}s", f"At least one {label} has a different mean", "F", g["f_stat"], g["anova_p"],
            f"mean sales differ between {label}s", f"mean sales differ between {label}s.", f"Effect size eta2 = {g['eta_squared']:.5f}.")
        add(f"Kruskal-Wallis: by {label}", f"The distribution of units sold is the same across {label}s", "At least one distribution differs", "H", g["kruskal_h"], g["kruskal_p"],
            f"sales distributions differ between {label}s", f"sales distributions differ between {label}s.")
    for c in ("inventory_level", "price", "discount", "holiday_promo"):
        if c in view.columns and view[c].nunique() > 1 and len(view) >= 3:
            r_, p_ = stats.pearsonr(view["units_sold"].astype(float), view[c].astype(float))
            add(f"Pearson correlation: units sold vs {c.replace('_', ' ')}", "The linear correlation is 0", "The linear correlation is not 0", "r", r_, p_,
                "there is a linear association", "a linear association exists.", "Correlation is not causation.")
    try:
        tr = eda.trend_test(view)
        add("Linear trend of daily total sales", "The trend slope is 0", "The slope is not 0", "slope/day", tr["slope_per_day"], tr["p_value"],
            "sales trend up or down over time", "sales trend over time.", f"About {tr['pct_change_per_year']:+.1f}% per year.")
    except DataValidationError:
        pass
    if not rows:
        raise DataValidationError("Not enough variation in the current selection to run statistical tests.")
    return pd.DataFrame(rows)


# ================================================================ forecast helpers
def restrict_selection(df: pd.DataFrame, stores=None, products=None, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of `df` for the chosen stores / products and/or exact (store_id, product_id) pairs. None = no restriction."""
    t = df
    if stores is not None:
        t = t[t["store_id"].astype(str).isin([str(x) for x in stores])]
    if products is not None:
        t = t[t["product_id"].astype(str).isin([str(x) for x in products])]
    if pairs is not None:
        keys = pairs[GROUP_KEYS].astype(str).drop_duplicates()
        t = t.assign(_s=t["store_id"].astype(str), _p=t["product_id"].astype(str)).merge(
            keys.rename(columns={"store_id": "_s", "product_id": "_p"}), on=["_s", "_p"], how="inner").drop(columns=["_s", "_p"])
    return t


def selection_residuals(test_predictions: pd.DataFrame, model: str, stores=None, products=None, aggregate: bool = True,
                        pairs: pd.DataFrame | None = None) -> pd.Series:
    """Held-out residuals (actual - predicted) of `model` for a selection.

    aggregate=True: per day, summed over the selected store-products (matches a forecast TOTAL for the selection);
    False: one residual per store-product-day (matches a single-series forecast).
    """
    need = {"date", "units_sold", model}
    if not need <= set(test_predictions.columns):
        raise DataValidationError(f"Saved test predictions lack the columns {sorted(need - set(test_predictions.columns))}.")
    t = restrict_selection(test_predictions, stores, products, pairs)
    if t.empty:
        raise DataValidationError("No held-out predictions match this selection.")
    if aggregate:
        g = t.groupby("date")[["units_sold", model]].sum()
        return (g["units_sold"] - g[model]).rename("residual")
    return (t["units_sold"] - t[model]).rename("residual")


def interval_coverage(residuals: pd.Series, interval: dict) -> float:
    """Share of the residuals that fall inside the interval offsets (in-sample check of the calibration set itself)."""
    r = np.asarray(residuals, float)
    return float(((r >= interval["lower_offset"]) & (r <= interval["upper_offset"])).mean())


def forecast_history(cleaned: pd.DataFrame, stores=None, products=None, days: int = 90, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Actual units sold per day for the selection (summed over the selected store-products), last `days` days."""
    h = restrict_selection(cleaned, stores, products, pairs)
    if h.empty:
        raise DataValidationError("No history matches this selection.")
    daily = h.groupby("date")["units_sold"].sum().reset_index()
    return daily[daily["date"] > daily["date"].max() - pd.Timedelta(days=days)]


# ================================================================ classifier curves from saved test predictions
def roc_pr_curves(pred: pd.DataFrame, models: list[str], label_col: str = "stockout_label") -> dict:
    """ROC and precision-recall points, plus Brier score and a reliability table, for each score column."""
    from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_curve

    if label_col not in pred.columns:
        raise DataValidationError("Saved predictions have no label column.")
    y = pred[label_col].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise DataValidationError("The test period contains a single class; curves cannot be drawn.")
    roc, pr, cal, brier = [], [], [], {}
    for m in models:
        if m not in pred.columns:
            continue
        p = pred[m].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        roc.append(pd.DataFrame({"model": m, "fpr": fpr, "tpr": tpr}))
        prec, rec, _ = precision_recall_curve(y, p)
        pr.append(pd.DataFrame({"model": m, "recall": rec, "precision": prec}))
        brier[m] = float(brier_score_loss(y, np.clip(p, 0, 1)))
        bins = pd.cut(np.clip(p, 0, 1), np.linspace(0, 1, 11), include_lowest=True)
        c = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin", observed=True).agg(mean_score=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index(drop=True)
        c.insert(0, "model", m)
        cal.append(c)
    if not roc:
        raise DataValidationError("None of the requested score columns are in the saved predictions.")
    return {"roc": pd.concat(roc, ignore_index=True), "pr": pd.concat(pr, ignore_index=True), "calibration": pd.concat(cal, ignore_index=True),
            "brier": brier, "prevalence": float(y.mean())}
