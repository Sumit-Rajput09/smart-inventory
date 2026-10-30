"""Non-visual helpers for the Streamlit dashboard (kept out of app.py so they can be tested).

Nothing here imports streamlit. File locations come from get_paths(), which the app calls on every run, so
tests (and a different project folder via the SMART_INVENTORY_ROOT environment variable) can redirect it.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.preprocess import GROUP_KEYS
from src.utils import PROJECT_ROOT, DataValidationError

ENV_ROOT = "SMART_INVENTORY_ROOT"

PAGES = [
    "Home / Executive Summary",
    "Dataset Explorer",
    "Data Quality and Preprocessing",
    "Exploratory Data Analysis",
    "Demand Forecasting",
    "Stock-Out Risk",
    "Reorder Recommendations",
    "Model Evaluation",
    "Inventory Report",
    "Project Documentation / About",
]
# Material icons shown next to the page names in the sidebar (the page NAME stays the value, so navigation keys never change)
PAGE_ICONS = {
    "Home / Executive Summary": "dashboard", "Dataset Explorer": "table_view", "Data Quality and Preprocessing": "fact_check",
    "Exploratory Data Analysis": "insights", "Demand Forecasting": "trending_up", "Stock-Out Risk": "warning",
    "Reorder Recommendations": "shopping_cart", "Model Evaluation": "analytics", "Inventory Report": "description",
    "Project Documentation / About": "menu_book",
}


@dataclass(frozen=True)
class AppPaths:
    root: Path
    raw_csv: Path
    cleaned_csv: Path
    features_csv: Path
    models_dir: Path
    reports_dir: Path

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


def get_paths(root: str | Path | None = None) -> AppPaths:
    """All project locations. Uses SMART_INVENTORY_ROOT if set, else the real project folder."""
    base = Path(root or os.environ.get(ENV_ROOT) or PROJECT_ROOT)
    return AppPaths(root=base, raw_csv=base / "data" / "raw" / "retail_store_inventory.csv",
                    cleaned_csv=base / "data" / "processed" / "retail_inventory_cleaned.csv",
                    features_csv=base / "data" / "processed" / "retail_inventory_features.csv",
                    models_dir=base / "models", reports_dir=base / "reports")


def mtime(path: Path) -> float:
    """Modification time used as a cache key so cached data refreshes when a file is regenerated."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def artifact_status(paths: AppPaths) -> list[dict]:
    """Which pipeline outputs exist, and the command that creates each missing one."""
    models = sorted(paths.models_dir.glob("demand_model_h*.joblib")) if paths.models_dir.exists() else []
    clfs = sorted(paths.models_dir.glob("stockout_classifier_simulated_L*.joblib")) if paths.models_dir.exists() else []
    rows = [
        ("Raw dataset", paths.raw_csv.exists(), "Download retail_store_inventory.csv from Kaggle into data\\raw\\"),
        ("Cleaned dataset", paths.cleaned_csv.exists(), "python -m src.preprocess"),
        ("Data quality report", (paths.reports_dir / "data_quality_report.csv").exists(), "python -m src.preprocess"),
        ("Demand forecasting models", bool(models), "python -m src.demand_forecasting"),
        ("Simulated stock-out classifier", bool(clfs), "python -m src.stockout_prediction"),
    ]
    return [{"item": a, "ready": b, "command": c} for a, b, c in rows]


def read_text(path: Path) -> str | None:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def search_dataframe(df: pd.DataFrame, query: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Case-insensitive search. Every word must appear (in any searched column) for a row to match."""
    terms = [t for t in str(query or "").lower().split() if t]
    if not terms or df.empty:
        return df
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    if not cols:
        return df
    text = {c: df[c].astype(str).str.lower() for c in cols}
    keep = pd.Series(True, index=df.index)
    for t in terms:
        hit = pd.Series(False, index=df.index)
        for c in cols:
            hit |= text[c].str.contains(t, regex=False, na=False)
        keep &= hit
    return df[keep]


def parse_on_order_upload(data: bytes | io.BytesIO) -> pd.DataFrame:
    """Validate an optional CSV of quantities already on order: columns store_id, product_id, on_order."""
    try:
        raw = pd.read_csv(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    except Exception as exc:  # noqa: BLE001 - any parse problem becomes a friendly message
        raise DataValidationError(f"Could not read the uploaded file as CSV: {exc}") from exc
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    need = {*GROUP_KEYS, "on_order"}
    if not need <= set(raw.columns):
        raise DataValidationError(f"The CSV needs the columns {sorted(need)}; found {list(raw.columns)}.")
    out = raw[[*GROUP_KEYS, "on_order"]].copy()
    out["on_order"] = pd.to_numeric(out["on_order"], errors="coerce")
    if out["on_order"].isna().any():
        raise DataValidationError("Some on_order values are missing or not numbers.")
    if (out["on_order"] < 0).any():
        raise DataValidationError("on_order values cannot be negative.")
    for c in GROUP_KEYS:
        out[c] = out[c].astype(str).str.strip()
    return out


def attach_snapshot_attributes(recs: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Add the category and region recorded for each store-product on the snapshot day (they vary row by row in this dataset)."""
    snap = recs["snapshot_date"].iloc[0]
    cols = [c for c in ("category", "region") if c in cleaned.columns]
    if not cols:
        return recs
    today = cleaned.loc[cleaned["date"] == snap, [*GROUP_KEYS, *cols]].drop_duplicates(GROUP_KEYS)
    out = recs.copy()
    for c in GROUP_KEYS:
        out[c] = out[c].astype(str)
        today[c] = today[c].astype(str)
    return out.merge(today, on=GROUP_KEYS, how="left")


def filter_recommendations(recs: pd.DataFrame, stores=None, products=None, categories=None, regions=None) -> pd.DataFrame:
    """Row filters for a recommendation table (None = no filter). Raises a friendly error if nothing remains."""
    mask = pd.Series(True, index=recs.index)
    for col, vals in (("store_id", stores), ("product_id", products), ("category", categories), ("region", regions)):
        if vals is not None and col in recs.columns:
            mask &= recs[col].astype(str).isin([str(v) for v in vals])
    out = recs[mask]
    if out.empty:
        raise DataValidationError("No store-products match the selected filters.")
    return out


def insights_for(insights: list[str], *keywords: str) -> list[str]:
    """Pick generated insight sentences that start with one of the keywords (e.g. 'TREND')."""
    return [i for i in insights if any(i.startswith(k) for k in keywords)]


def fmt_int(x) -> str:
    return f"{int(round(x)):,}"


def fmt_pct(x, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


# What each column of the Kaggle file means, and what the project found out about it in Phases 1-2.
COLUMN_DICTIONARY = [
    ("Date", "Calendar day of the record (2022-01-01 to 2024-01-01, one row per store-product per day).", "Complete: 731 days for each of the 100 series."),
    ("Store ID", "Store identifier (S001-S005).", "5 stores."),
    ("Product ID", "Product identifier (P0001-P0020).", "20 products."),
    ("Category", "Product category: Groceries, Toys, Electronics, Furniture, Clothing.", "Varies row by row for the same product."),
    ("Region", "Store region: North, South, East, West.", "Varies row by row for the same store."),
    ("Inventory Level", "Units in stock on that day.", "Used as stock on hand at the start of the day. Sales never exceed it."),
    ("Units Sold", "Units sold that day. This is the demand the project forecasts.", "Right-skewed (mean about 136, maximum 499)."),
    ("Units Ordered", "Units ordered from the supplier that day.", "Unrelated to stock or sales; not used."),
    ("Demand Forecast", "A forecast of demand supplied in the file.", "Almost identical to Units Sold (correlation 0.997): target leakage, excluded from models."),
    ("Price", "Selling price.", "Not related to sales in this data."),
    ("Discount", "Discount percentage (0, 5, 10, 15, 20).", "Not related to sales in this data."),
    ("Weather Condition", "Sunny, Rainy, Cloudy or Snowy.", "Not known in advance, so not used as a forecasting feature."),
    ("Holiday/Promotion", "1 if a holiday or promotion applied, else 0.", "No effect on sales found (p = 0.92)."),
    ("Competitor Pricing", "Competitor's price for the product.", "Almost the same as Price (correlation 0.99)."),
    ("Seasonality", "Season label.", "Matches the real season only 25% of the time (chance level): not used."),
]
