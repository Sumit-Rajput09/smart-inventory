"""Cleaning, feature engineering and chronological splitting.

Run from the project root:
    python -m src.preprocess
    python -m src.preprocess --horizon 7

Design rules (see reports/preprocessing_summary.txt after running):
  * Cleaning never uses future rows (only causal forward-fill inside a store-product series).
  * The target (units_sold) is never imputed; rows without it are dropped.
  * Outliers are FLAGGED, not deleted.
  * Lag / rolling features are built per store-product series, by calendar date,
    and use only observations at least `horizon` days older than the target day.
  * Anything that is learned from data (imputer, encoder, scaler) is fitted later
    on the training period only - nothing is fitted here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import check_required_columns, load_dataset
from src.utils import (
    CLEANED_CSV,
    FEATURES_CSV,
    RAW_CSV,
    REPORTS_DIR,
    DataValidationError,
    ensure_dirs,
    get_logger,
)

log = get_logger()

GROUP_KEYS = ["store_id", "product_id"]
TARGET = "units_sold"
DATE = "date"
TEXT_COLUMNS = ["store_id", "product_id", "category", "region", "weather", "seasonality"]
NUMERIC_COLUMNS = [
    "inventory_level", "units_sold", "units_ordered", "demand_forecast",
    "price", "discount", "competitor_price", "holiday_promo",
]
NON_NEGATIVE = ["inventory_level", "units_sold", "units_ordered", "demand_forecast"]
POSITIVE = ["price", "competitor_price"]
MAX_FFILL_DAYS = 7  # a stale value older than this many rows is not trusted

# Columns that must never be model inputs for same-day demand, with the reason.
LEAKAGE_COLUMNS: dict[str, str] = {
    "units_sold": "this is the prediction target itself",
    "demand_forecast": "is almost a copy of units_sold in this dataset (correlation ~0.997), i.e. it already contains the answer",
    "units_ordered": "the order quantity is decided on the same day; it is not known before the demand is observed",
}
# Columns left out of the feature set for other reasons.
OTHER_EXCLUSIONS: dict[str, str] = {
    "weather": "not known in advance at forecast time (would need its own forecast)",
    "seasonality": "label is inconsistent with the calendar month in this dataset (see inspection report)",
    "inventory_level": "same-day stock is bounded by sales and is not known before the day; kept for stock-out rules, not for demand",
}


# ---------------------------------------------------------------- logging of quality checks
@dataclass
class QualityLog:
    """Collects one row per check/action; becomes reports/data_quality_report.csv."""

    n_rows: int = 0
    rows: list[dict] = field(default_factory=list)

    def add(self, step: str, check: str, column: str, count: int, action: str) -> None:
        pct = round(100 * count / self.n_rows, 4) if self.n_rows else 0.0
        self.rows.append({"step": step, "check": check, "column": column,
                          "affected_rows": int(count), "pct_of_rows": pct, "action": action})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["step", "check", "column", "affected_rows", "pct_of_rows", "action"])


# ---------------------------------------------------------------- cleaning helpers
def parse_dates(series: pd.Series) -> tuple[pd.Series, int, int]:
    """Parse dates. ISO 'YYYY-MM-DD' first; anything else via flexible parsing.

    Returns (parsed, n_unparseable, n_parsed_by_fallback). Unparseable -> NaT.
    """
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    retry = parsed.isna() & text.notna()
    n_fallback = 0
    if retry.any():
        second = pd.to_datetime(text[retry], format="mixed", errors="coerce")
        n_fallback = int(second.notna().sum())
        parsed = parsed.copy()
        parsed[retry] = second
    return parsed, int(parsed.isna().sum()), n_fallback


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityLog]:
    """Return (cleaned_df, quality_log). Input must already use canonical column names."""
    check_required_columns(df)
    df = df.copy()
    qlog = QualityLog(n_rows=len(df))
    n_start = len(df)

    # 0. whitespace -> NaN for text columns
    for col in [c for c in TEXT_COLUMNS if c in df.columns]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    # 1. missing values BEFORE cleaning (report all columns)
    for col in df.columns:
        qlog.add("1_missing_before_cleaning", "missing values", col, int(df[col].isna().sum()), "reported")

    # 2. exact duplicates
    n_dup = int(df.duplicated().sum())
    qlog.add("2_duplicates", "exact duplicate rows", "(all)", n_dup, "dropped (keep first)")
    df = df.drop_duplicates(keep="first")

    # 3. dates
    df[DATE], n_bad_date, n_fallback = parse_dates(df[DATE])
    qlog.add("3_dates", "dates parsed with non-ISO fallback", DATE, n_fallback, "parsed, kept")
    qlog.add("3_dates", "unparseable / missing dates", DATE, n_bad_date, "dropped (cannot be placed on the timeline)")
    df = df[df[DATE].notna()]

    # 4. missing identifiers
    missing_ids = df[GROUP_KEYS].isna().any(axis=1)
    qlog.add("4_identifiers", "missing store_id/product_id", "store_id,product_id", int(missing_ids.sum()), "dropped")
    df = df[~missing_ids]

    # 5. numeric coercion
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        before_na = df[col].isna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_new_na = int((df[col].isna() & ~before_na).sum())
        qlog.add("5_numeric_validation", "non-numeric text in numeric column", col, n_new_na, "set to missing")

    # 6. impossible values -> missing (never silently clipped to a "nice" number)
    for col in [c for c in NON_NEGATIVE if c in df.columns]:
        bad = df[col] < 0
        qlog.add("6_impossible_values", "negative value", col, int(bad.sum()), "set to missing")
        df.loc[bad, col] = np.nan
    for col in [c for c in POSITIVE if c in df.columns]:
        bad = df[col] <= 0
        qlog.add("6_impossible_values", "price <= 0", col, int(bad.sum()), "set to missing")
        df.loc[bad, col] = np.nan
    if "discount" in df.columns:
        bad = (df["discount"] < 0) | (df["discount"] > 100)
        qlog.add("6_impossible_values", "discount outside 0-100", "discount", int(bad.sum()), "set to missing")
        df.loc[bad, "discount"] = np.nan
    if "holiday_promo" in df.columns:
        bad = ~df["holiday_promo"].isin([0, 1]) & df["holiday_promo"].notna()
        qlog.add("6_impossible_values", "flag not in {0,1}", "holiday_promo", int(bad.sum()), "set to missing")
        df.loc[bad, "holiday_promo"] = np.nan

    # 7. duplicate (date, store, product) keys with different values
    n_key_dup = int(df.duplicated(subset=[DATE, *GROUP_KEYS], keep="last").sum())
    qlog.add("7_duplicate_keys", "same date+store+product appears again", "date,store_id,product_id", n_key_dup,
             "dropped, last record kept (assumed to be the correction)")
    df = df.drop_duplicates(subset=[DATE, *GROUP_KEYS], keep="last")

    # 8. order chronologically inside each series (required for every causal operation below)
    df = df.sort_values([*GROUP_KEYS, DATE], kind="mergesort").reset_index(drop=True)

    # 9. target is never imputed
    n_no_target = int(df[TARGET].isna().sum())
    qlog.add("9_target", "missing / invalid target", TARGET, n_no_target, "dropped (target is never imputed)")
    df = df[df[TARGET].notna()].reset_index(drop=True)

    # 10. causal imputation of the remaining numeric columns
    grouped = df.groupby(GROUP_KEYS, sort=False)
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns and c != TARGET]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = grouped[col].ffill(limit=MAX_FFILL_DAYS)
        left = int(df[col].isna().sum())
        qlog.add("10_imputation", f"missing filled by past value (max {MAX_FFILL_DAYS} rows back)", col,
                 n_missing - left, "forward-fill inside store-product series (past only)")
        qlog.add("10_imputation", "still missing after causal fill", col, left,
                 "left as NaN; imputed with TRAIN-only median inside the model pipeline")
    for col in [c for c in TEXT_COLUMNS if c in df.columns]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna("Unknown")
            qlog.add("10_imputation", "missing text", col, n_missing, "set to 'Unknown'")

    # 11. consistency information (reported, not changed)
    if "inventory_level" in df.columns:
        qlog.add("11_consistency", "units_sold > inventory_level", "units_sold,inventory_level",
                 int((df[TARGET] > df["inventory_level"]).sum()), "reported only (stock may be measured at end of day)")

    # 12. calendar gaps inside series (matters for lag features)
    span = df.groupby(GROUP_KEYS, sort=False)[DATE].agg(["min", "max", "count"])
    expected = (span["max"] - span["min"]).dt.days + 1
    qlog.add("12_calendar_gaps", "missing calendar days inside series", "date", int((expected - span["count"]).sum()),
             "reported; lag/rolling features are built by calendar date so gaps do not shift them")

    # 13. outliers: flag only (per store-product, 1.5 * IQR rule)
    for col in [c for c in ("units_sold", "inventory_level") if c in df.columns]:
        q1 = df.groupby(GROUP_KEYS, sort=False)[col].transform(lambda s: s.quantile(0.25))
        q3 = df.groupby(GROUP_KEYS, sort=False)[col].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        flag = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        df[f"outlier_{col}"] = flag
        qlog.add("13_outliers", "outside 1.5*IQR of its store-product series", col, int(flag.sum()),
                 "flagged in column outlier_<name>, NOT removed (valid unusual values are kept)")

    # 14. tidy dtypes
    for col in ("units_sold", "inventory_level", "units_ordered", "discount", "holiday_promo"):
        if col in df.columns and df[col].notna().all():
            df[col] = df[col].astype("int32")
    for col in ("category", "region", "weather", "seasonality"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in GROUP_KEYS:
        df[col] = df[col].astype("string")

    if df.empty:
        raise DataValidationError("No rows left after cleaning - check data_quality_report.csv for the reason.")
    qlog.add("99_result", "rows kept / rows read", "(all)", len(df), f"{len(df)} of {n_start} rows kept")
    return df, qlog


# ---------------------------------------------------------------- feature engineering
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df[DATE].dt
    df["year"] = d.year
    df["month"] = d.month
    df["day"] = d.day
    df["day_of_week"] = d.dayofweek
    df["day_of_year"] = d.dayofyear
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price/discount/promotion are assumed to be PLANNED in advance, so same-day values are allowed."""
    df = df.copy()
    if {"price", "discount"} <= set(df.columns):
        df["effective_price"] = df["price"] * (1 - df["discount"] / 100.0)
    if {"price", "competitor_price"} <= set(df.columns):
        df["price_vs_competitor"] = df["price"] / df["competitor_price"] - 1.0
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 7, 14),
    windows: tuple[int, ...] = (7, 14, 28),
    horizon: int = 1,
) -> pd.DataFrame:
    """Add lag and rolling features of units_sold for each store-product series.

    horizon=h means: the row for day t is predicted using information up to day t-h.
      * lag_k  = units_sold on day t-k          (only k >= h are allowed)
      * roll_w = mean/std of units_sold over the w days ending at day t-h
    Everything is looked up BY CALENDAR DATE, so missing days cannot shift a lag,
    and the current day's target is never inside its own features.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    lags = tuple(k for k in lags if k >= horizon)
    if TARGET not in df.columns:
        raise DataValidationError(f"'{TARGET}' column is required to build lag features.")

    df = df.sort_values([*GROUP_KEYS, DATE], kind="mergesort").reset_index(drop=True)
    out = {f"{TARGET}_lag_{k}": np.full(len(df), np.nan) for k in lags}
    for w in windows:
        out[f"roll_mean_{w}"] = np.full(len(df), np.nan)
        out[f"roll_std_{w}"] = np.full(len(df), np.nan)

    for _, idx in df.groupby(GROUP_KEYS, sort=False).indices.items():
        s = pd.Series(df[TARGET].to_numpy()[idx], index=pd.DatetimeIndex(df[DATE].to_numpy()[idx]))
        for k in lags:
            out[f"{TARGET}_lag_{k}"][idx] = s.reindex(s.index - pd.Timedelta(days=k)).to_numpy()
        for w in windows:
            # closed="left": the window [t-w, t) excludes day t itself, so it ends at t-1
            roll = s.rolling(f"{w}D", closed="left", min_periods=max(2, w // 2))
            for name, series in ((f"roll_mean_{w}", roll.mean()), (f"roll_std_{w}", roll.std())):
                shifted = series.reindex(s.index - pd.Timedelta(days=horizon - 1)).to_numpy()
                out[name][idx] = shifted

    df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
    if {"roll_mean_7", "roll_mean_28"} <= set(df.columns):
        df["trend_7_vs_28"] = df["roll_mean_7"] / df["roll_mean_28"] - 1.0
    return df


def build_features(cleaned: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = add_calendar_features(cleaned)
    df = add_price_features(df)
    return add_lag_features(df, horizon=horizon)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns that may be used as model inputs (numeric, leakage-free)."""
    candidates = [
        "year", "month", "day", "day_of_week", "day_of_year", "week_of_year", "is_weekend",
        "price", "discount", "competitor_price", "effective_price", "price_vs_competitor", "holiday_promo",
    ]
    engineered = [c for c in df.columns if "_lag_" in c or c.startswith("roll_") or c == "trend_7_vs_28"]
    cols = [c for c in candidates + engineered if c in df.columns]
    assert_no_leakage(cols)
    return cols


def assert_no_leakage(feature_columns: list[str]) -> None:
    """Raise if a forbidden same-day column is used as a feature."""
    bad = [c for c in feature_columns if c in LEAKAGE_COLUMNS]
    if bad:
        reasons = "; ".join(f"{c}: {LEAKAGE_COLUMNS[c]}" for c in bad)
        raise DataValidationError(f"Target leakage - these columns cannot be features: {reasons}")


# ---------------------------------------------------------------- chronological split
def chronological_split(
    df: pd.DataFrame, test_days: int = 90, val_days: int = 90
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by DATE: [... train ...][ validation ][ test ] with no shuffling and no overlap."""
    if test_days < 1 or val_days < 1:
        raise ValueError("test_days and val_days must be >= 1")
    last = df[DATE].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df[df[DATE] < val_start]
    val = df[(df[DATE] >= val_start) & (df[DATE] < test_start)]
    test = df[df[DATE] >= test_start]
    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty:
            raise DataValidationError(f"The {name} period is empty - the data is too short for val_days={val_days}, test_days={test_days}.")
    assert train[DATE].max() < val[DATE].min() <= val[DATE].max() < test[DATE].min()
    return train, val, test


# ---------------------------------------------------------------- reports + pipeline
def write_summary(path: Path, raw_shape: tuple[int, int], mapping: dict[str, str], cleaned: pd.DataFrame,
                  features: pd.DataFrame, qlog: QualityLog, horizon: int) -> None:
    q = qlog.to_frame()
    train, val, test = chronological_split(features)
    lines = [
        "PREPROCESSING SUMMARY",
        "=" * 70,
        f"Raw data      : {raw_shape[0]} rows x {raw_shape[1]} columns",
        f"Cleaned data  : {len(cleaned)} rows x {cleaned.shape[1]} columns",
        f"Date range    : {cleaned[DATE].min().date()} to {cleaned[DATE].max().date()}",
        f"Series        : {cleaned.groupby(GROUP_KEYS, observed=True).ngroups} store-product combinations",
        f"Forecast horizon used for features: {horizon} day(s)",
        "",
        "COLUMN MAPPING (original -> internal)",
        *[f"  {a:22s} -> {b}" for a, b in mapping.items()],
        "",
        "CLEANING ACTIONS (non-zero only; full list in data_quality_report.csv)",
        *[f"  [{r.step}] {r.check} ({r.column}): {r.affected_rows} rows -> {r.action}"
          for r in q.itertuples() if r.affected_rows and not r.step.startswith(("1_", "99"))],
        "",
        "MISSING-VALUE STRATEGY",
        "  target (units_sold)   : never imputed, rows dropped",
        f"  other numeric columns : forward-fill from the PAST inside each store-product series (max {MAX_FFILL_DAYS} rows);",
        "                          anything still missing is imputed later with a median fitted on the TRAIN period only",
        "  text columns          : 'Unknown'",
        "",
        "FEATURES",
        f"  {len(get_feature_columns(features))} model features: " + ", ".join(get_feature_columns(features)),
        "",
        "LEAKAGE - columns that must NOT be used to predict same-day demand",
        *[f"  {c}: {why}" for c, why in LEAKAGE_COLUMNS.items()],
        "OTHER EXCLUDED COLUMNS",
        *[f"  {c}: {why}" for c, why in OTHER_EXCLUSIONS.items()],
        "",
        "CHRONOLOGICAL SPLIT (default 90-day validation, 90-day test)",
        f"  train      : {train[DATE].min().date()} to {train[DATE].max().date()}  ({len(train)} rows)",
        f"  validation : {val[DATE].min().date()} to {val[DATE].max().date()}  ({len(val)} rows)",
        f"  test       : {test[DATE].min().date()} to {test[DATE].max().date()}  ({len(test)} rows)",
        "",
        "LIMITATION: the dataset is SYNTHETIC. Results do not demonstrate real retail savings.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(raw_path: str | Path = RAW_CSV, horizon: int = 1, reports_dir: Path = REPORTS_DIR,
                 cleaned_path: Path = CLEANED_CSV, features_path: Path = FEATURES_CSV) -> dict:
    """Load -> clean -> features -> save. Returns the produced objects (used by tests and the app)."""
    ensure_dirs()
    raw, mapping = load_dataset(raw_path)
    log.info("Loaded %d rows x %d columns", *raw.shape)
    cleaned, qlog = clean_dataset(raw)
    features = build_features(cleaned, horizon=horizon)

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)
    features.to_csv(features_path, index=False)
    qlog.to_frame().to_csv(reports_dir / "data_quality_report.csv", index=False)
    write_summary(reports_dir / "preprocessing_summary.txt", raw.shape, mapping, cleaned, features, qlog, horizon)
    log.info("Saved %s", cleaned_path)
    log.info("Saved %s", features_path)
    log.info("Saved %s", reports_dir / "data_quality_report.csv")
    log.info("Saved %s", reports_dir / "preprocessing_summary.txt")
    return {"raw": raw, "cleaned": cleaned, "features": features, "quality": qlog.to_frame()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the inventory dataset and build features.")
    parser.add_argument("--input", default=str(RAW_CSV), help="path to the raw CSV")
    parser.add_argument("--horizon", type=int, default=1, help="forecast horizon in days (default 1)")
    args = parser.parse_args()
    run_pipeline(args.input, horizon=args.horizon)


if __name__ == "__main__":
    main()
