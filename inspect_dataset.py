"""Dataset inspection script (Phase 1, step 1).

Run from the project root:
    python inspect_dataset.py
    python inspect_dataset.py --input data\\raw\\retail_store_inventory.csv

It only READS the CSV. It prints everything and saves the same text to
reports/dataset_inspection_report.txt. Nothing here modifies your data.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from src.data_loader import map_columns, read_csv_safely
from src.utils import RAW_CSV, REPORTS_DIR, InventoryProjectError, ensure_dirs

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 40)

LINES: list[str] = []


def out(text: object = "") -> None:
    """Print and remember a line (ASCII only so Windows consoles never fail)."""
    s = str(text)
    print(s)
    LINES.append(s)


def title(text: str) -> None:
    out()
    out("=" * 78)
    out(text)
    out("=" * 78)


NH_SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def inspect(path) -> None:
    findings: list[str] = []

    title("1. LOADING")
    raw = read_csv_safely(path)
    out(f"File: {path}")
    df, mapping, unrecognised = map_columns(raw)

    title("2. COLUMN NAMES, DATA TYPES AND MAPPING TO INTERNAL NAMES")
    table = pd.DataFrame({"original_name": list(mapping.keys()), "internal_name": list(mapping.values()),
                          "dtype": [str(raw[c].dtype) for c in mapping]})
    out(table.to_string(index=False))
    out(f"Unrecognised columns (kept, not discarded): {unrecognised if unrecognised else 'none'}")

    title("3. FIRST AND LAST FIVE RECORDS")
    out("-- head --")
    out(df.head(5).to_string())
    out("-- tail --")
    out(df.tail(5).to_string())

    title("4. SHAPE")
    out(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")

    title("5. MISSING VALUES AND DUPLICATES")
    miss = pd.DataFrame({"missing": df.isna().sum(), "pct": (100 * df.isna().mean()).round(3)})
    out(miss.to_string())
    n_dup = int(df.duplicated().sum())
    out(f"Exact duplicate rows: {n_dup}")
    if {"date", "store_id", "product_id"} <= set(df.columns):
        n_key = int(df.duplicated(["date", "store_id", "product_id"]).sum())
        out(f"Duplicate (date, store_id, product_id) keys: {n_key}")

    title("6. UNIQUE COUNTS OF CATEGORICAL COLUMNS")
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != "date"]
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        top = {str(k): int(v) for k, v in vc.head(8).items()}
        out(f"{c}: {df[c].nunique()} unique -> {top}")

    title("7. NUMERIC SUMMARY")
    num = df.select_dtypes(include="number")
    summary = num.describe().T
    summary["variance"] = num.var()
    summary["skew"] = num.skew()
    out(summary.round(3).to_string())

    title("8. DATE INSPECTION")
    date_text = df["date"].astype("string").str.strip()
    iso_share = date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").mean()
    parsed = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
    out(f"Share of values in ISO format YYYY-MM-DD: {100 * iso_share:.2f}%")
    out(f"Unparseable / missing dates: {int(parsed.isna().sum())}")
    out(f"Date range: {parsed.min().date()} to {parsed.max().date()}  ({parsed.nunique()} distinct days)")
    full_days = (parsed.max() - parsed.min()).days + 1
    out(f"Calendar days in range: {full_days}; missing calendar days overall: {full_days - parsed.nunique()}")
    if {"store_id", "product_id"} <= set(df.columns):
        sizes = df.assign(_d=parsed).groupby(["store_id", "product_id"]).size()
        out(f"Store-product series: {len(sizes)}; rows per series min/median/max = "
            f"{sizes.min()}/{int(sizes.median())}/{sizes.max()}")

    title("9. INVALID VALUES AND INCONSISTENT RECORDS")
    checks: list[tuple[str, int]] = []
    for c in ("units_sold", "inventory_level", "units_ordered", "demand_forecast"):
        if c in df:
            checks.append((f"negative {c}", int((df[c] < 0).sum())))
    for c in ("price", "competitor_price"):
        if c in df:
            checks.append((f"{c} <= 0", int((df[c] <= 0).sum())))
    if "price" in df:
        checks.append(("price > 10x median price (suspiciously high)", int((df["price"] > 10 * df["price"].median()).sum())))
    if "discount" in df:
        checks.append(("discount outside 0-100", int(((df["discount"] < 0) | (df["discount"] > 100)).sum())))
    if "holiday_promo" in df:
        checks.append(("holiday_promo not in {0,1}", int((~df["holiday_promo"].isin([0, 1])).sum())))
    if {"units_sold", "inventory_level"} <= set(df.columns):
        checks.append(("units_sold > inventory_level", int((df["units_sold"] > df["inventory_level"]).sum())))
        checks.append(("inventory_level == 0", int((df["inventory_level"] == 0).sum())))
    for name, n in checks:
        out(f"  {name:55s}: {n}")

    if {"store_id", "product_id", "category", "region"} <= set(df.columns):
        cats = df.groupby("product_id")["category"].nunique()
        regs = df.groupby("store_id")["region"].nunique()
        out(f"  products that appear in more than one category        : {int((cats > 1).sum())} of {len(cats)}")
        out(f"  stores that appear in more than one region            : {int((regs > 1).sum())} of {len(regs)}")
        if (cats > 1).any() or (regs > 1).any():
            findings.append("category/region are NOT stable per product/store (they vary row by row) - they are attributes of the "
                            "row, not master data. Category/region analysis is possible but a 'product' has no single category.")
    if "seasonality" in df:
        match = (parsed.dt.month.map(NH_SEASON) == df["seasonality"]).mean()
        out(f"  'seasonality' label equals the northern-hemisphere season of the date: {100 * match:.1f}% (random would be ~25%)")
        if match < 0.5:
            findings.append("the Seasonality column does not follow the calendar -> not used as a feature; use month/day-of-year instead.")

    title("10. WHAT CAN THE DATA SUPPORT? (checks that decide the project scope)")
    if {"units_sold", "demand_forecast"} <= set(df.columns):
        corr = df["units_sold"].corr(df["demand_forecast"])
        mae = (df["units_sold"] - df["demand_forecast"]).abs().mean()
        out(f"  corr(Units Sold, Demand Forecast) = {corr:.4f};  mean |Units Sold - Demand Forecast| = {mae:.2f}")
        if corr > 0.95:
            findings.append(f"Demand Forecast is almost identical to Units Sold (corr {corr:.3f}). It cannot be used as an input "
                            "(target leakage). It is only useful as a 'vendor forecast' benchmark, and even then it was probably "
                            "generated from the actual sales.")
    if {"units_sold", "store_id", "product_id"} <= set(df.columns):
        tmp = df.assign(_d=parsed).sort_values(["store_id", "product_id", "_d"])
        g = tmp.groupby(["store_id", "product_id"])["units_sold"]
        acs = {k: tmp["units_sold"].corr(g.shift(k)) for k in (1, 7, 14, 28)}
        out("  autocorrelation of units_sold within a store-product series: " +
            ", ".join(f"lag {k}: {v:+.3f}" for k, v in acs.items()))
        out(f"  std of units_sold overall: {tmp['units_sold'].std():.1f};  std of the 100 series means: {g.mean().std():.1f}")
        if max(abs(v) for v in acs.values()) < 0.05:
            findings.append("no autocorrelation: past sales carry almost no information about future sales. Expect lag/rolling-feature "
                            "models to perform about the same as a simple mean baseline. This must be reported honestly, not hidden.")
        mdow = tmp.groupby(tmp["_d"].dt.dayofweek)["units_sold"].mean()
        mmon = tmp.groupby(tmp["_d"].dt.month)["units_sold"].mean()
        out(f"  mean units_sold by weekday: min {mdow.min():.1f}, max {mdow.max():.1f};  by month: min {mmon.min():.1f}, max {mmon.max():.1f}")
        if (mdow.max() - mdow.min()) / mdow.mean() < 0.05 and (mmon.max() - mmon.min()) / mmon.mean() < 0.1:
            findings.append("weekday and monthly seasonality are weak (<5% / <10% spread).")
    if "units_ordered" in df and "inventory_level" in df:
        out(f"  corr(Units Ordered, Inventory Level) = {df['units_ordered'].corr(df['inventory_level']):+.3f}")
    if "units_sold" in df and "inventory_level" in df:
        out(f"  corr(Units Sold, Inventory Level)    = {df['units_sold'].corr(df['inventory_level']):+.3f}")

    title("FINDINGS THAT AFFECT THE PROJECT")
    findings.append("The dataset is SYNTHETIC (Kaggle). Results must not be presented as real business savings.")
    findings.append("There are no real stock-out labels and no supplier lead times; the rule-based stock-out analysis must use a "
                    "configurable lead time, and any classifier would need clearly labelled SIMULATED labels.")
    for i, f in enumerate(findings, 1):
        out(f"  {i}. {f}")

    ensure_dirs()
    report = REPORTS_DIR / "dataset_inspection_report.txt"
    report.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nReport saved to: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the raw inventory CSV.")
    parser.add_argument("--input", default=str(RAW_CSV))
    args = parser.parse_args()
    try:
        inspect(args.input)
    except InventoryProjectError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
