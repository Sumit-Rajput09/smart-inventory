"""Exploratory Data Analysis: statistics, hypothesis tests, charts and insights.

Run from the project root (needs `python -m src.preprocess` first):
    python -m src.eda

Every chart function takes a (possibly filtered) DataFrame and RETURNS a
matplotlib Figure, so the same code feeds the PNG files written here and the
Streamlit dashboard (st.pyplot(fig)) built later. Nothing is hard-coded: every
sentence in the insights section is generated from the numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

from src.utils import CLEANED_CSV, REPORTS_DIR, DataFileError, DataValidationError, ensure_dirs, get_logger

sns.set_theme(style="whitegrid", context="notebook")
log = get_logger()

NUMERIC_FOR_EDA = ["units_sold", "inventory_level", "units_ordered", "demand_forecast",
                   "price", "discount", "competitor_price", "holiday_promo"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ALPHA = 0.05


# =============================================================== loading and filtering
def load_cleaned(path: str | Path = CLEANED_CSV) -> pd.DataFrame:
    """Read the cleaned CSV produced by src.preprocess."""
    path = Path(path)
    if not path.exists():
        raise DataFileError(f"Cleaned dataset not found: {path}\nRun first:  python -m src.preprocess")
    df = pd.read_csv(path, parse_dates=["date"])
    _require(df, ["date", "store_id", "product_id", "units_sold"])
    return df


def _require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"Column(s) {missing} are needed for this analysis but are not in the data.")


def apply_filters(df: pd.DataFrame, date_range: tuple | None = None, stores=None, products=None,
                  categories=None, regions=None) -> pd.DataFrame:
    """Filter by date range and/or selected values. None = no filter on that field.

    Raises DataValidationError with a friendly message if nothing is left
    (the dashboard shows this message instead of crashing).
    """
    mask = pd.Series(True, index=df.index)
    if date_range is not None:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= df["date"].between(start, end)
    for col, values in (("store_id", stores), ("product_id", products), ("category", categories), ("region", regions)):
        if values is not None:
            mask &= df[col].isin(list(values))
    out = df[mask]
    if out.empty:
        raise DataValidationError("No records match the selected filters. Widen the date range or select more stores/products.")
    return out


# =============================================================== aggregation tables
def aggregate_sales(df: pd.DataFrame, freq: str = "D") -> pd.Series:
    """Total units sold per day ('D'), week ('W', Mon-Sun) or month ('M').

    Incomplete first/last weeks or months are dropped: a 1-day "month" would look like a crash.
    """
    if freq not in {"D", "W", "M"}:
        raise ValueError("freq must be 'D', 'W' or 'M'")
    if df.empty:
        raise DataValidationError("Cannot aggregate an empty dataset.")
    daily = df.groupby("date")["units_sold"].sum().asfreq("D")
    if freq == "D":
        return daily
    rule = "W-SUN" if freq == "W" else "MS"
    res = daily.resample(rule)
    total, n_days = res.sum(min_count=1), res.count()
    expected = np.full(len(total), 7) if freq == "W" else total.index.days_in_month.to_numpy()
    return total[(n_days.to_numpy() >= expected)]


def sales_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Rows, total and mean units sold (with standard error) for each value of `by`."""
    _require(df, [by])
    g = df.groupby(by, observed=True)["units_sold"]
    t = g.agg(rows="count", total_units="sum", mean_units="mean", std_units="std")
    t["sem"] = t["std_units"] / np.sqrt(t["rows"])
    return t.sort_values("mean_units", ascending=False).reset_index()


def product_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Per product: total units, mean units per store-day, mean stock and sell-through (units sold / stock)."""
    _require(df, ["inventory_level"])
    g = df.groupby("product_id", observed=True)
    t = g.agg(total_units=("units_sold", "sum"), mean_units=("units_sold", "mean"),
              mean_inventory=("inventory_level", "mean"))
    t["sell_through"] = t["mean_units"] / t["mean_inventory"]
    t = t.sort_values("mean_units", ascending=False)
    t.insert(0, "rank", range(1, len(t) + 1))
    return t.reset_index()


def top_and_slow(df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(top-n, slowest-n) products by mean units sold per store-day."""
    r = product_ranking(df)
    n = max(1, min(n, len(r) // 2 if len(r) > 1 else 1))
    return r.head(n), r.tail(n).sort_values("mean_units")


# =============================================================== statistics
def descriptive_statistics(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Mean, median, std, variance, min, max, range, Q1, Q3, IQR, skewness, kurtosis, CV%."""
    cols = [c for c in (cols or NUMERIC_FOR_EDA) if c in df.columns]
    if not cols:
        raise DataValidationError("No numeric columns available for descriptive statistics.")
    x = df[cols].astype(float)
    t = pd.DataFrame({
        "count": x.count(), "mean": x.mean(), "median": x.median(), "std": x.std(), "variance": x.var(),
        "min": x.min(), "max": x.max(), "range": x.max() - x.min(),
        "q1": x.quantile(0.25), "q3": x.quantile(0.75),
    })
    t["iqr"] = t["q3"] - t["q1"]
    t["skewness"] = x.skew()
    t["kurtosis"] = x.kurt()
    t["cv_percent"] = 100 * t["std"] / t["mean"].replace(0, np.nan)
    return t


def compare_two_groups(df: pd.DataFrame, value: str = "units_sold", group: str = "holiday_promo") -> dict:
    """Hypothesis test: is the mean of `value` different between the two values of `group`?

    H0: both groups have the same mean.  H1: the means differ (two-sided), alpha = 0.05.
    Welch's t-test (does not assume equal variances) is the main test; the Mann-Whitney U test is
    shown as a check because units_sold is right-skewed.
    Assumptions: independent observations (only approximately true: rows of one store-product
    series repeat over time), and a large n so the sample mean is close to normal (CLT).
    """
    _require(df, [value, group])
    levels = sorted(df[group].dropna().unique())
    if len(levels) != 2:
        raise DataValidationError(f"'{group}' must have exactly 2 groups in the selected data, found {len(levels)}: {levels}")
    a = df.loc[df[group] == levels[0], value].astype(float)
    b = df.loc[df[group] == levels[1], value].astype(float)
    if len(a) < 2 or len(b) < 2:
        raise DataValidationError("Each group needs at least 2 observations for a t-test.")
    t_stat, p = stats.ttest_ind(b, a, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = b.mean() - a.mean()
    crit = stats.t.ppf(1 - ALPHA / 2, dof)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    mw = stats.mannwhitneyu(b, a, alternative="two-sided")
    mw_p = mw.pvalue
    return {"value": value, "group": group, "level_a": levels[0], "level_b": levels[1],
            "n_a": len(a), "n_b": len(b), "mean_a": a.mean(), "mean_b": b.mean(), "diff_b_minus_a": diff,
            "ci_low": diff - crit * se, "ci_high": diff + crit * se, "t_stat": t_stat, "welch_p": p,
            "cohens_d": diff / pooled if pooled else np.nan, "mannwhitney_p": mw_p,
            "mannwhitney_u": float(mw.statistic)}


def compare_many_groups(df: pd.DataFrame, value: str = "units_sold", group: str = "discount") -> dict:
    """One-way ANOVA (and Kruskal-Wallis as non-parametric check) across the values of `group`.

    H0: all group means are equal. eta_squared = share of the variance explained by the grouping.
    """
    _require(df, [value, group])
    samples = [g[value].astype(float).to_numpy() for _, g in df.groupby(group, observed=True) if len(g) >= 2]
    if len(samples) < 3:
        raise DataValidationError(f"Need at least 3 groups of 2+ rows in '{group}', found {len(samples)}.")
    f_stat, p = stats.f_oneway(*samples)
    kw = stats.kruskal(*samples)
    kw_p = kw.pvalue
    allv = np.concatenate(samples)
    ss_total = ((allv - allv.mean()) ** 2).sum()
    ss_between = sum(len(s) * (s.mean() - allv.mean()) ** 2 for s in samples)
    return {"value": value, "group": group, "n_groups": len(samples), "f_stat": f_stat, "anova_p": p,
            "kruskal_p": kw_p, "kruskal_h": float(kw.statistic), "eta_squared": ss_between / ss_total if ss_total else np.nan}


def correlation_tests(df: pd.DataFrame, target: str = "units_sold", others: list[str] | None = None) -> pd.DataFrame:
    """Pearson and Spearman correlation of `target` with other numeric columns, with p-values."""
    others = [c for c in (others or NUMERIC_FOR_EDA) if c in df.columns and c != target]
    rows = []
    for c in others:
        sub = df[[target, c]].dropna()
        if len(sub) < 3 or sub[c].nunique() < 2:
            continue
        pr, pp = stats.pearsonr(sub[target], sub[c])
        sr, sp = stats.spearmanr(sub[target], sub[c])
        rows.append({"variable": c, "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp})
    return pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False).reset_index(drop=True)


def trend_test(df: pd.DataFrame) -> dict:
    """Linear trend of total daily sales (least squares). H0: slope = 0."""
    daily = aggregate_sales(df, "D").dropna()
    if len(daily) < 10:
        raise DataValidationError("Need at least 10 days of data for a trend test.")
    res = stats.linregress(np.arange(len(daily)), daily.to_numpy())
    mean = daily.mean()
    return {"slope_per_day": res.slope, "p_value": res.pvalue, "r_squared": res.rvalue ** 2,
            "pct_change_per_year": 100 * res.slope * 365 / mean if mean else np.nan, "n_days": len(daily)}


def outlier_summary(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """IQR (1.5x) fences and counts, plus |z| > 3 counts, over the whole selected data."""
    cols = [c for c in (cols or ["units_sold", "inventory_level", "units_ordered", "price"]) if c in df.columns]
    rows = []
    for c in cols:
        x = df[c].astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((x < lo) | (x > hi)).sum())
        z = (x - x.mean()) / x.std() if x.std() else x * 0
        rows.append({"column": c, "lower_fence": lo, "upper_fence": hi, "n_iqr_outliers": n_iqr,
                     "pct_iqr_outliers": 100 * n_iqr / len(x), "n_abs_z_gt_3": int((z.abs() > 3).sum())})
    return pd.DataFrame(rows)


def missing_summary(df: pd.DataFrame, quality_report: str | Path | None = None) -> pd.DataFrame:
    """Missing values in the cleaned data and (if the report exists) before cleaning."""
    t = pd.DataFrame({"missing_after_cleaning": df.isna().sum()})
    t = t[~t.index.str.startswith("outlier_")]  # helper flag columns added by preprocessing
    t.index.name = "column"
    qpath = Path(quality_report) if quality_report else REPORTS_DIR / "data_quality_report.csv"
    if qpath.exists():
        q = pd.read_csv(qpath)
        before = q[q["step"] == "1_missing_before_cleaning"].set_index("column")["affected_rows"]
        t["missing_before_cleaning"] = before.reindex(t.index)
    return t.reset_index()


def seasonal_summary(df: pd.DataFrame) -> dict:
    """Mean units sold by month and weekday, relative spread, and Kruskal-Wallis tests."""
    d = df["date"].dt
    month = df.groupby(d.month)["units_sold"].mean()
    dow = df.groupby(d.dayofweek)["units_sold"].mean()
    overall = df["units_sold"].mean()
    out = {"month_means": month, "dow_means": dow,
           "month_spread_pct": 100 * (month.max() - month.min()) / overall,
           "dow_spread_pct": 100 * (dow.max() - dow.min()) / overall,
           "month_p": np.nan, "dow_p": np.nan}
    if month.size >= 3:
        out["month_p"] = stats.kruskal(*[g.to_numpy() for _, g in df.groupby(d.month)["units_sold"]]).pvalue
    if dow.size >= 3:
        out["dow_p"] = stats.kruskal(*[g.to_numpy() for _, g in df.groupby(d.dayofweek)["units_sold"]]).pvalue
    return out


# =============================================================== plotting helpers
def _new_fig(nrows: int = 1, ncols: int = 1, figsize=(10, 5)):
    fig = Figure(figsize=figsize, layout="constrained")
    return fig, fig.subplots(nrows, ncols)


def _bar_with_error(ax, table: pd.DataFrame, label_col: str, horizontal: bool, overall: float):
    labels = table[label_col].astype(str)
    err = 1.96 * table["sem"].to_numpy()
    if horizontal:
        ax.barh(labels, table["mean_units"], xerr=err, color="#4C78A8", capsize=2)
        ax.invert_yaxis()
        ax.axvline(overall, color="#E45756", ls="--", lw=1.2, label=f"Overall mean ({overall:.1f})")
        ax.set_xlabel("Mean units sold per store-day (bars: 95% CI)")
    else:
        ax.bar(labels, table["mean_units"], yerr=err, color="#4C78A8", capsize=3)
        ax.axhline(overall, color="#E45756", ls="--", lw=1.2, label=f"Overall mean ({overall:.1f})")
        ax.set_ylabel("Mean units sold per store-day (bars: 95% CI)")
    ax.legend(loc="lower right")


# =============================================================== figures (each returns a Figure)
def plot_sales_trends(df: pd.DataFrame) -> Figure:
    fig, axes = _new_fig(3, 1, (11, 10))
    daily = aggregate_sales(df, "D")
    axes[0].plot(daily.index, daily.to_numpy(), lw=0.7, alpha=0.55, label="Daily total")
    axes[0].plot(daily.index, daily.rolling(7, min_periods=1).mean().to_numpy(), lw=2, color="#E45756", label="7-day average")
    axes[0].set(title="Daily total units sold", ylabel="Units sold")
    axes[0].legend()
    for ax, freq, name in ((axes[1], "W", "Weekly"), (axes[2], "M", "Monthly")):
        s = aggregate_sales(df, freq)
        ax.plot(s.index, s.to_numpy(), marker="o", ms=3, lw=1.5)
        ax.set(title=f"{name} total units sold (incomplete first/last periods excluded)", ylabel="Units sold")
        if len(s):
            ax.axhline(s.mean(), color="grey", ls="--", lw=1, label=f"Mean {name.lower()} total")
            ax.legend()
    axes[2].set_xlabel("Date")
    return fig


def plot_sales_by(df: pd.DataFrame, by: str) -> Figure:
    t = sales_by(df, by)
    horizontal = len(t) > 6
    fig, ax = _new_fig(1, 1, (9, 0.35 * len(t) + 2.5 if horizontal else 5))
    _bar_with_error(ax, t, by, horizontal, df["units_sold"].mean())
    ax.set_title(f"Average units sold by {by.replace('_', ' ')}")
    if horizontal:
        ax.set_ylabel(by.replace("_", " ").title())
    else:
        ax.set_xlabel(by.replace("_", " ").title())
    return fig


def plot_top_slow_products(df: pd.DataFrame, n: int = 5) -> Figure:
    top, slow = top_and_slow(df, n)
    fig, axes = _new_fig(1, 2, (12, 5), )
    for ax, t, title, color in ((axes[0], top, f"Top {len(top)} products (highest mean units/store-day)", "#54A24B"),
                                (axes[1], slow, f"Slowest {len(slow)} products (lowest mean units/store-day)", "#E45756")):
        ax.barh(t["product_id"].astype(str), t["mean_units"], color=color)
        ax.invert_yaxis()
        for y, (mu, st) in enumerate(zip(t["mean_units"], t["sell_through"])):
            ax.text(mu, y, f" {mu:.1f}  (sell-through {st:.0%})", va="center", fontsize=9)
        ax.set(title=title, xlabel="Mean units sold per store-day", ylabel="Product")
        ax.set_xlim(0, t["mean_units"].max() * 1.35)
    return fig


def plot_inventory_distribution(df: pd.DataFrame) -> Figure:
    _require(df, ["inventory_level"])
    has_cat = "category" in df.columns
    fig, axes = _new_fig(1, 2 if has_cat else 1, (12 if has_cat else 7, 5))
    axes = np.atleast_1d(axes)
    sns.histplot(df["inventory_level"], bins=40, kde=True, ax=axes[0], color="#4C78A8")
    axes[0].axvline(df["inventory_level"].mean(), color="#E45756", ls="--", label=f"Mean {df['inventory_level'].mean():.0f}")
    axes[0].axvline(df["inventory_level"].median(), color="#54A24B", ls=":", label=f"Median {df['inventory_level'].median():.0f}")
    axes[0].set(title="Inventory level distribution", xlabel="Inventory level (units)", ylabel="Number of records")
    axes[0].legend()
    if has_cat:
        sns.boxplot(data=df, x="category", y="inventory_level", ax=axes[1], color="#9ecae9", fliersize=2)
        axes[1].set(title="Inventory level by category", xlabel="Category", ylabel="Inventory level (units)")
        axes[1].tick_params(axis="x", rotation=20)
    return fig


def plot_sales_vs_inventory(df: pd.DataFrame) -> Figure:
    _require(df, ["inventory_level"])
    fig, axes = _new_fig(1, 2, (12, 5))
    hb = axes[0].hexbin(df["inventory_level"], df["units_sold"], gridsize=40, bins="log", cmap="viridis", mincnt=1)
    lim = max(df["inventory_level"].max(), df["units_sold"].max())
    axes[0].plot([0, lim], [0, lim], color="#E45756", ls="--", label="Units sold = inventory")
    r = df["units_sold"].corr(df["inventory_level"])
    axes[0].set(title=f"Units sold vs inventory level (Pearson r = {r:.2f})", xlabel="Inventory level (units)", ylabel="Units sold")
    axes[0].legend(loc="upper left")
    fig.colorbar(hb, ax=axes[0], label="Records (log scale)")
    ratio = (df["units_sold"] / df["inventory_level"].where(df["inventory_level"] > 0)).dropna()
    sns.histplot(ratio.clip(upper=1.5), bins=40, ax=axes[1], color="#F58518")
    axes[1].axvline(ratio.median(), color="black", ls="--", label=f"Median {ratio.median():.2f}")
    axes[1].set(title="Sell-through ratio (units sold / inventory level)", xlabel="Sell-through ratio", ylabel="Number of records")
    axes[1].legend()
    return fig


def plot_price_discount(df: pd.DataFrame) -> Figure:
    _require(df, ["price", "discount", "competitor_price"])
    fig, axes = _new_fig(2, 2, (12, 9))
    d = df.groupby("discount")["units_sold"].agg(["mean", "std", "count"])
    axes[0, 0].bar(d.index.astype(str), d["mean"], yerr=1.96 * d["std"] / np.sqrt(d["count"]), color="#4C78A8", capsize=3)
    axes[0, 0].set(title="Mean units sold by discount level", xlabel="Discount (%)", ylabel="Mean units sold (bars: 95% CI)")
    bins = pd.qcut(df["price"], 10, duplicates="drop")
    p = df.groupby(bins, observed=True).agg(price=("price", "mean"), units=("units_sold", "mean"))
    axes[0, 1].plot(p["price"], p["units"], marker="o")
    axes[0, 1].axhline(df["units_sold"].mean(), color="#E45756", ls="--", lw=1, label="Overall mean")
    axes[0, 1].set(title="Mean units sold by price decile", xlabel="Average price in decile", ylabel="Mean units sold",
                   ylim=(0, p["units"].max() * 1.2))  # axis starts at 0 so small wiggles are not exaggerated
    axes[0, 1].legend()
    hb = axes[1, 0].hexbin(df["price"], df["competitor_price"], gridsize=35, bins="log", cmap="viridis", mincnt=1)
    lim = max(df["price"].max(), df["competitor_price"].max())
    axes[1, 0].plot([0, lim], [0, lim], color="#E45756", ls="--", label="Equal price")
    axes[1, 0].set(title="Own price vs competitor price", xlabel="Own price", ylabel="Competitor price")
    axes[1, 0].legend()
    fig.colorbar(hb, ax=axes[1, 0], label="Records (log)")
    cheaper = np.where(df["price"] < df["competitor_price"], "Cheaper than competitor", "Same or dearer")
    c = df.groupby(cheaper)["units_sold"].agg(["mean", "std", "count"])
    axes[1, 1].bar(c.index, c["mean"], yerr=1.96 * c["std"] / np.sqrt(c["count"]), color=["#54A24B", "#E45756"][: len(c)], capsize=3)
    axes[1, 1].set(title="Mean units sold: own price vs competitor", ylabel="Mean units sold (bars: 95% CI)")
    return fig


def plot_promo_comparison(df: pd.DataFrame) -> Figure:
    _require(df, ["holiday_promo"])
    res = compare_two_groups(df)
    fig, axes = _new_fig(1, 2, (11, 5))
    labels = [f"No promotion/holiday ({res['level_a']})", f"Promotion/holiday ({res['level_b']})"]
    means = [res["mean_a"], res["mean_b"]]
    sems = [df.loc[df.holiday_promo == lv, "units_sold"].sem() for lv in (res["level_a"], res["level_b"])]
    axes[0].bar(labels, means, yerr=[1.96 * s for s in sems], color=["#9ecae9", "#F58518"], capsize=4)
    axes[0].set(title=f"Mean units sold (Welch t-test p = {res['welch_p']:.3f})", ylabel="Mean units sold (bars: 95% CI)")
    axes[0].tick_params(axis="x", labelsize=9)
    sns.boxplot(data=df, x="holiday_promo", y="units_sold", ax=axes[1], palette=["#9ecae9", "#F58518"], hue="holiday_promo", legend=False)
    axes[1].set(title="Distribution of units sold", xlabel="Holiday/Promotion flag", ylabel="Units sold")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> Figure:
    cols = [c for c in NUMERIC_FOR_EDA if c in df.columns]
    corr = df[cols].corr(method=method)
    fig, ax = _new_fig(1, 1, (9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True, ax=ax,
                cbar_kws={"label": f"{method.title()} correlation"})
    ax.set_title(f"Correlation heatmap ({method.title()})")
    ax.tick_params(axis="x", rotation=40)
    return fig


def plot_histograms(df: pd.DataFrame) -> Figure:
    cols = [c for c in ("units_sold", "inventory_level", "price", "units_ordered") if c in df.columns]
    fig, axes = _new_fig(2, 2, (12, 8))
    for ax, c in zip(np.ravel(axes), cols):
        sns.histplot(df[c], bins=40, kde=True, ax=ax, color="#4C78A8")
        ax.axvline(df[c].mean(), color="#E45756", ls="--", label=f"Mean {df[c].mean():.1f}")
        ax.axvline(df[c].median(), color="#54A24B", ls=":", label=f"Median {df[c].median():.1f}")
        ax.set(title=f"Distribution of {c.replace('_', ' ')} (skew {df[c].skew():.2f})", xlabel=c.replace("_", " ").title(), ylabel="Records")
        ax.legend()
    return fig


def plot_boxplots(df: pd.DataFrame) -> Figure:
    dims = [c for c in ("category", "store_id", "region") if c in df.columns]
    fig, axes = _new_fig(1, len(dims), (5 * len(dims), 5))
    for ax, c in zip(np.atleast_1d(axes), dims):
        sns.boxplot(data=df, x=c, y="units_sold", ax=ax, color="#9ecae9", fliersize=2)
        ax.set(title=f"Units sold by {c.replace('_', ' ')}", xlabel=c.replace("_", " ").title(), ylabel="Units sold")
        ax.tick_params(axis="x", rotation=25)
    return fig


def _iqr_flags(df: pd.DataFrame) -> pd.Series:
    """Per store-product 1.5*IQR flag (same rule as preprocessing). Used if the column is absent."""
    if "outlier_units_sold" in df.columns:
        return df["outlier_units_sold"].astype(bool)
    g = df.groupby(["store_id", "product_id"], observed=True)["units_sold"]
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    return (df["units_sold"] < q1 - 1.5 * iqr) | (df["units_sold"] > q3 + 1.5 * iqr)


def plot_outliers(df: pd.DataFrame) -> Figure:
    fig, axes = _new_fig(1, 2, (13, 5.5))
    order = sorted(df["product_id"].unique())
    sns.boxplot(data=df, x="product_id", y="units_sold", order=order, ax=axes[0], color="#9ecae9", fliersize=2)
    axes[0].set(title="Units sold by product (points beyond whiskers = 1.5 x IQR outliers)", xlabel="Product", ylabel="Units sold")
    axes[0].tick_params(axis="x", rotation=90, labelsize=8)
    flags = _iqr_flags(df)
    pct = flags.groupby(df["product_id"]).mean().reindex(order) * 100
    axes[1].bar(pct.index.astype(str), pct.to_numpy(), color="#E45756")
    axes[1].set(title=f"Share of records flagged as outliers per store-product series (total {100 * flags.mean():.2f}%)",
                xlabel="Product", ylabel="Outlier records (%)")
    axes[1].tick_params(axis="x", rotation=90, labelsize=8)
    return fig


def plot_seasonality(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(12, 9), layout="constrained")
    ax = fig.subplot_mosaic([["m", "d"], ["h", "h"]])
    s = seasonal_summary(df)
    overall = df["units_sold"].mean()
    ax["m"].bar(s["month_means"].index.astype(str), s["month_means"].to_numpy(), color="#4C78A8")
    ax["m"].axhline(overall, color="#E45756", ls="--", label=f"Overall mean {overall:.1f}")
    ax["m"].set(title=f"Mean units sold by month (spread {s['month_spread_pct']:.1f}%)", xlabel="Month", ylabel="Mean units sold")
    ax["m"].legend(loc="lower right")
    ax["d"].bar([DAY_NAMES[i] for i in s["dow_means"].index], s["dow_means"].to_numpy(), color="#54A24B")
    ax["d"].axhline(overall, color="#E45756", ls="--")
    ax["d"].set(title=f"Mean units sold by weekday (spread {s['dow_spread_pct']:.1f}%)", xlabel="Weekday", ylabel="Mean units sold")
    heat = df.assign(month=df["date"].dt.month, dow=df["date"].dt.dayofweek).pivot_table(index="month", columns="dow", values="units_sold", aggfunc="mean")
    heat.columns = [DAY_NAMES[i] for i in heat.columns]
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax["h"], cbar_kws={"label": "Mean units sold"})
    ax["h"].set(title="Mean units sold: month x weekday", xlabel="Weekday", ylabel="Month")
    return fig


def plot_missing(df: pd.DataFrame) -> Figure:
    t = missing_summary(df)
    fig, ax = _new_fig(1, 1, (10, 5))
    if "missing_before_cleaning" in t.columns:
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["missing_before_cleaning"].fillna(0), 0.4, label="Before cleaning", color="#E45756")
        ax.bar(x + 0.2, t["missing_after_cleaning"], 0.4, label="After cleaning", color="#4C78A8")
        ax.set_xticks(x, t["column"], rotation=60, ha="right")
        ax.legend()
    else:
        ax.bar(t["column"], t["missing_after_cleaning"], color="#4C78A8")
        ax.tick_params(axis="x", rotation=60)
    if t.filter(like="missing").to_numpy().sum() == 0:
        ax.text(0.5, 0.5, "No missing values in any column", transform=ax.transAxes, ha="center", fontsize=14, color="grey")
    ax.set(title="Missing values per column", ylabel="Number of missing records")
    return fig


# name -> function(df) used by the CLI and (later) by the Streamlit app
FIGURES = {
    "01_sales_trends": plot_sales_trends,
    "02_sales_by_product": lambda d: plot_sales_by(d, "product_id"),
    "03_sales_by_store": lambda d: plot_sales_by(d, "store_id"),
    "04_sales_by_category": lambda d: plot_sales_by(d, "category"),
    "05_sales_by_region": lambda d: plot_sales_by(d, "region"),
    "06_top_and_slow_products": plot_top_slow_products,
    "07_inventory_distribution": plot_inventory_distribution,
    "08_sales_vs_inventory": plot_sales_vs_inventory,
    "09_price_and_discount": plot_price_discount,
    "10_promotion_comparison": plot_promo_comparison,
    "11_correlation_heatmap": plot_correlation_heatmap,
    "12_histograms": plot_histograms,
    "13_boxplots": plot_boxplots,
    "14_outliers": plot_outliers,
    "15_seasonality": plot_seasonality,
    "16_missing_values": plot_missing,
}


# =============================================================== insights (generated from numbers)
def _pstr(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Plain-language findings. Wording depends on the computed results, never fixed in advance."""
    out: list[str] = []

    tr = trend_test(df)
    if tr["p_value"] < ALPHA:
        d = "upward" if tr["slope_per_day"] > 0 else "downward"
        out.append(f"TREND: total daily sales show a statistically significant {d} linear trend of {tr['pct_change_per_year']:+.1f}% per year "
                   f"({_pstr(tr['p_value'])}, R2 = {tr['r_squared']:.3f}).")
    else:
        out.append(f"TREND: no significant linear trend in total daily sales ({tr['pct_change_per_year']:+.1f}% per year, {_pstr(tr['p_value'])}). "
                   "Demand is flat over the period, so growth-based planning is not supported by this data.")

    s = seasonal_summary(df)
    small = max(s["month_spread_pct"], s["dow_spread_pct"]) < 10
    out.append(f"SEASONALITY: monthly means differ by {s['month_spread_pct']:.1f}% of the overall mean (Kruskal-Wallis {_pstr(s['month_p'])}); "
               f"weekday means differ by {s['dow_spread_pct']:.1f}% ({_pstr(s['dow_p'])}). "
               + ("The differences are small in practical terms, so month/weekday alone will not give strong forecasting signal."
                  if small else "The spread is large enough to matter for planning; calendar features should help forecasting."))

    if "inventory_level" in df.columns:
        top, slow = top_and_slow(df, 3)
        gap = 100 * (top.mean_units.iloc[0] - slow.mean_units.iloc[0]) / df["units_sold"].mean()
        pg = compare_many_groups(df, "units_sold", "product_id")
        out.append(f"PRODUCTS: highest mean is {top.product_id.iloc[0]} ({top.mean_units.iloc[0]:.1f} units/store-day), lowest is "
                   f"{slow.product_id.iloc[0]} ({slow.mean_units.iloc[0]:.1f}); the gap is {gap:.1f}% of the overall mean. ANOVA across products: {_pstr(pg['anova_p'])}"
                   + (", so products genuinely differ in demand." if pg["anova_p"] < ALPHA else ", so the ranking could be due to chance and should not drive decisions."))

    for grp, label in (("store_id", "STORES"), ("category", "CATEGORIES"), ("region", "REGIONS")):
        if grp in df.columns and df[grp].nunique() >= 3:
            r = compare_many_groups(df, "units_sold", grp)
            if r["anova_p"] < 0.01 and r["eta_squared"] < 0.01:
                verdict = "a statistically detectable but practically tiny difference."
            elif r["anova_p"] < 0.01:
                verdict = "a meaningful difference."
            elif r["anova_p"] < ALPHA:
                verdict = "borderline (p between 0.01 and 0.05); with several tests run this may be chance, and the effect size is negligible." if r["eta_squared"] < 0.01 else "borderline but a non-trivial effect size; worth checking."
            else:
                verdict = "no evidence that average demand differs between them."
            out.append(f"{label}: ANOVA across {r['n_groups']} groups {_pstr(r['anova_p'])}, eta2 = {r['eta_squared']:.5f} -> {verdict}")

    if "holiday_promo" in df.columns and df["holiday_promo"].nunique() == 2:
        r = compare_two_groups(df)
        eff = "negligible" if abs(r["cohens_d"]) < 0.2 else "small" if abs(r["cohens_d"]) < 0.5 else "medium or large"
        out.append(f"PROMOTION/HOLIDAY: mean units sold {r['mean_b']:.1f} with the flag vs {r['mean_a']:.1f} without (difference {r['diff_b_minus_a']:+.2f}, "
                   f"95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}], Welch {_pstr(r['welch_p'])}, Mann-Whitney {_pstr(r['mannwhitney_p'])}, Cohen's d = {r['cohens_d']:.3f}, {eff} effect). "
                   + ("There is evidence of an effect." if r["welch_p"] < ALPHA else
                      "There is no evidence that promotions/holidays change sales here - unusual for real retail data."))

    if "discount" in df.columns and df["discount"].nunique() >= 3:
        r = compare_many_groups(df, "units_sold", "discount")
        out.append(f"DISCOUNT: ANOVA across {r['n_groups']} discount levels {_pstr(r['anova_p'])}, eta2 = {r['eta_squared']:.5f}. "
                   + ("Discount level is associated with sales." if r["anova_p"] < ALPHA else "Higher discounts are not associated with higher sales in this data."))

    ct = correlation_tests(df)
    if not ct.empty:
        strong = ct[ct.pearson_r.abs() >= 0.3]
        bits = ", ".join(f"{r.variable} (r = {r.pearson_r:+.2f})" for r in strong.itertuples())
        out.append("CORRELATIONS WITH UNITS SOLD: " + (f"moderate/strong: {bits}." if bits else "no variable reaches |r| = 0.3."))
        if "demand_forecast" in strong.variable.values:
            out.append("LEAKAGE WARNING: demand_forecast is almost the same series as units_sold, so it is excluded from model inputs.")

    if "inventory_level" in df.columns:
        over = int((df["units_sold"] > df["inventory_level"]).sum())
        ratio = (df["units_sold"] / df["inventory_level"].where(df["inventory_level"] > 0)).median()
        out.append(f"INVENTORY: median sell-through is {ratio:.0%} of stock per day; units sold exceeded inventory level in {over} records. "
                   + ("Stock level behaves like a ceiling on sales in this dataset, which is why inventory and sales correlate - "
                      "this reflects how the data was generated, not that stocking more causes more demand." if over == 0 else
                      "Sales above recorded stock may mean inventory is measured after sales or restocking happened during the day."))

    if "inventory_level" in df.columns:
        ratio = (df["units_sold"] / df["inventory_level"].where(df["inventory_level"] > 0)).dropna()
        if len(ratio) >= 30 and ratio.max() <= 1.0:
            ks = stats.kstest(ratio, "uniform")
            out.append(f"SELL-THROUGH SHAPE: units_sold / inventory_level has mean {ratio.mean():.3f} and a maximum of {ratio.max():.2f}; a Kolmogorov-Smirnov test "
                       f"against Uniform(0,1) gives D = {ks.statistic:.4f}, {_pstr(ks.pvalue)}. "
                       + ("The ratio is almost perfectly evenly spread between 0 and 1 (D is tiny; a small p-value here only reflects the huge sample and "
                          "integer rounding): each day's sales look like a random share of that day's stock. Real retail demand is not generated this way, "
                          "which is strong evidence the file is simulated; it also means a day's sales cannot be predicted from history, only from "
                          "same-day stock, which is unknown at forecast time."
                          if ks.statistic < 0.02 else
                          "The ratio departs noticeably from a uniform spread; check the sell-through histogram for its shape."))

    flags = _iqr_flags(df)
    out.append(f"OUTLIERS: {flags.mean() * 100:.2f}% of records lie outside 1.5 x IQR of their own store-product series; they are flagged but kept, "
               "because high-sales days are valid demand that the inventory plan must cover.")
    miss = int(df.isna().sum().sum())
    out.append(f"MISSING VALUES: {miss} missing cell(s) remain after cleaning"
               + (" (see the missing-values table; they are left for train-only imputation inside the model pipeline)." if miss else "."))
    out.append("CAUTION: the dataset is synthetic. These findings describe this file only and are not evidence about real retail behaviour or savings. "
               "Several tests are run above; at alpha = 0.05 about 1 in 20 true-null tests will look significant by chance.")
    return out


# =============================================================== report + CLI
def build_report(df: pd.DataFrame) -> str:
    def block(title: str, body: str) -> str:
        return f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{body}\n"

    fmt = lambda t: t.to_string(float_format=lambda v: f"{v:,.4g}")  # noqa: E731
    parts = [f"EDA REPORT - {len(df)} records, {df['date'].min().date()} to {df['date'].max().date()}, "
             f"{df['store_id'].nunique()} stores, {df['product_id'].nunique()} products"]
    parts.append(block("DESCRIPTIVE STATISTICS", fmt(descriptive_statistics(df))))
    parts.append(block("CORRELATION WITH UNITS SOLD (Pearson and Spearman)", fmt(correlation_tests(df))))
    if "holiday_promo" in df.columns and df["holiday_promo"].nunique() == 2:
        r = compare_two_groups(df)
        parts.append(block("HYPOTHESIS TEST: does the Holiday/Promotion flag change mean units sold?",
                           "H0: mean units sold is equal with and without the flag; H1: it differs. alpha = 0.05.\n"
                           "Test: Welch two-sample t-test (unequal variances), checked with Mann-Whitney U because units_sold is right-skewed.\n"
                           "Assumptions: independent records (approximate, repeated store-product series) and large n (CLT).\n"
                           + "\n".join(f"  {k}: {v:.5g}" if isinstance(v, float) else f"  {k}: {v}" for k, v in r.items())))
    parts.append(block("OUTLIER SUMMARY", fmt(outlier_summary(df))))
    parts.append(block("TOP AND SLOWEST PRODUCTS", "Top:\n" + fmt(top_and_slow(df)[0]) + "\n\nSlowest:\n" + fmt(top_and_slow(df)[1])))
    parts.append(block("MISSING VALUES", fmt(missing_summary(df))))
    parts.append(block("INSIGHTS AND BUSINESS INTERPRETATION", "\n".join(f"{i}. {t}" for i, t in enumerate(generate_insights(df), 1))))
    return "".join(parts)


def run_eda(input_path: str | Path = CLEANED_CSV, out_dir: Path = REPORTS_DIR) -> dict:
    """Load cleaned data, save all figures, tables and the text report. Returns what was written."""
    ensure_dirs()
    df = load_cleaned(input_path)
    fig_dir, tab_dir = out_dir / "figures", out_dir / "eda_tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, fn in FIGURES.items():
        fig = fn(df)
        path = fig_dir / f"{name}.png"
        fig.savefig(path, dpi=120)
        saved.append(path)
    tables = {"descriptive_statistics": descriptive_statistics(df).reset_index(names="column"),
              "correlation_tests": correlation_tests(df), "outlier_summary": outlier_summary(df),
              "product_ranking": product_ranking(df), "missing_summary": missing_summary(df)}
    for name, t in tables.items():
        t.to_csv(tab_dir / f"{name}.csv", index=False)
    report = out_dir / "eda_report.txt"
    report.write_text(build_report(df), encoding="utf-8")
    return {"figures": saved, "tables": list(tab_dir.glob("*.csv")), "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis on the cleaned dataset.")
    parser.add_argument("--input", default=str(CLEANED_CSV))
    args = parser.parse_args()
    try:
        res = run_eda(args.input)
    except (DataFileError, DataValidationError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    log.info("Saved %d figures to %s", len(res["figures"]), res["figures"][0].parent)
    log.info("Saved %d tables to %s", len(res["tables"]), res["tables"][0].parent)
    log.info("Saved report %s", res["report"])


if __name__ == "__main__":
    main()
