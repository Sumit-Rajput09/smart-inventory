"""Safe CSV loading and explicit column-name mapping.

The Kaggle file uses names such as "Units Sold" and "Holiday/Promotion".
Inside the project we use canonical snake_case names (units_sold, holiday_promo).
Mapping is explicit and nothing is silently dropped: columns we do not recognise
are kept (with a cleaned-up name) and reported.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils import RAW_CSV, DataFileError, SchemaError

# canonical name -> accepted spellings (compared after normalisation)
CANONICAL_ALIASES: dict[str, list[str]] = {
    "date": ["date", "sales date", "transaction date", "order date"],
    "store_id": ["store id", "store", "store code"],
    "product_id": ["product id", "product", "sku", "item id"],
    "category": ["category", "product category"],
    "region": ["region", "store region"],
    "inventory_level": ["inventory level", "inventory", "stock level", "stock on hand", "on hand"],
    "units_sold": ["units sold", "sales", "quantity sold", "sold"],
    "units_ordered": ["units ordered", "order quantity", "ordered"],
    "demand_forecast": ["demand forecast", "forecast"],
    "price": ["price", "unit price"],
    "discount": ["discount", "discount pct", "discount percent"],
    "weather": ["weather condition", "weather"],
    "holiday_promo": ["holiday/promotion", "holiday promotion", "holiday", "promotion", "promo"],
    "competitor_price": ["competitor pricing", "competitor price"],
    "seasonality": ["seasonality", "season"],
}

# Columns without which the project cannot work at all.
REQUIRED_COLUMNS = ["date", "store_id", "product_id", "units_sold", "inventory_level"]


def normalize_name(name: str) -> str:
    """'Holiday/Promotion' -> 'holiday promotion'; 'store_id' -> 'store id'."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


_LOOKUP = {normalize_name(alias): canon for canon, aliases in CANONICAL_ALIASES.items() for alias in aliases}
_LOOKUP.update({normalize_name(canon): canon for canon in CANONICAL_ALIASES})


def read_csv_safely(path: str | Path = RAW_CSV, encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")) -> pd.DataFrame:
    """Read a CSV with helpful errors and encoding fallback. Columns keep their original names."""
    path = Path(path)
    if not path.exists():
        raise DataFileError(
            f"Dataset not found: {path}\n"
            "Download 'retail_store_inventory.csv' from Kaggle "
            "(anirudhchauhan/retail-store-inventory-forecasting-dataset) and place it in data\\raw\\."
        )
    if not path.is_file():
        raise DataFileError(f"Expected a file but found a folder: {path}")
    if path.stat().st_size == 0:
        raise DataFileError(f"The file is empty: {path}")

    last_error: Exception | None = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:  # try the next encoding
            last_error = exc
    else:
        raise DataFileError(f"Could not decode {path} with encodings {encodings}: {last_error}")
    return _check_not_empty(df, path)


def _check_not_empty(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if df.shape[1] < 2:
        raise DataFileError(f"{path} has {df.shape[1]} column(s); it does not look like a comma-separated dataset.")
    if len(df) == 0:
        raise DataFileError(f"{path} has a header but no data rows.")
    return df


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Rename recognised columns to canonical names.

    Returns (renamed_df, mapping {original: new}, list_of_unrecognised_original_names).
    Unrecognised columns are KEPT under a normalised snake_case name.
    """
    mapping: dict[str, str] = {}
    unrecognised: list[str] = []
    for col in df.columns:
        key = normalize_name(col)
        if key in _LOOKUP:
            mapping[col] = _LOOKUP[key]
        else:
            unrecognised.append(col)
            mapping[col] = key.replace(" ", "_") or "unnamed"

    targets = list(mapping.values())
    duplicates = sorted({t for t in targets if targets.count(t) > 1})
    if duplicates:
        raise SchemaError(f"Several columns map to the same name {duplicates}. Rename them in the CSV. Mapping: {mapping}")
    return df.rename(columns=mapping), mapping, unrecognised


def check_required_columns(df: pd.DataFrame, required: list[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Required column(s) missing after mapping: {missing}.\n"
            f"Columns found: {list(df.columns)}\n"
            "Add the spelling used in your file to CANONICAL_ALIASES in src/data_loader.py."
        )


def load_dataset(path: str | Path = RAW_CSV) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read -> map names -> verify required columns. Returns (df, mapping)."""
    raw = read_csv_safely(path)
    df, mapping, _ = map_columns(raw)
    check_required_columns(df)
    return df, mapping
