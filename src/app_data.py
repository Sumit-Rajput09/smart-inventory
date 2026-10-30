"""Cached data access and shared dashboard state (imports Streamlit; the pure logic lives in dashboard_metrics / app_support).

Cache rule: every cached function receives the file's modification time as an argument, so regenerating a CSV or a model
refreshes the cache, and every inventory setting is part of the cache key (InventoryParams is a frozen dataclass), so changing a
setting can never show stale results.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src import app_support as sup
from src import dashboard_metrics as dm
from src import demand_forecasting as dfc
from src import eda
from src import reorder_optimizer as ro
from src import stockout_prediction as sp
from src.utils import DataFileError, DataValidationError

PERSIST_KEYS = ["inv_lead", "inv_sl", "inv_review", "inv_lookback", "inv_ss_mode", "inv_ss_units", "inv_ss_days",
                "inv_moq", "inv_mult", "inv_excess", "inv_snapshot"]
EMPTY_FILTERS = {"date_range": None, "stores": None, "products": None, "categories": None, "regions": None}


@dataclass
class Ctx:
    """Everything a page needs: file locations, the full cleaned data, the filtered view and the active filters."""

    paths: sup.AppPaths
    cleaned: pd.DataFrame
    view: pd.DataFrame
    filters: dict

    @property
    def path(self) -> str:
        return str(self.paths.cleaned_csv)

    @property
    def stamp(self) -> float:
        return sup.mtime(self.paths.cleaned_csv)


# ================================================================ cached loaders
@st.cache_data(show_spinner="Loading cleaned data...")
def load_cleaned_cached(path: str, stamp: float) -> pd.DataFrame:
    return eda.load_cleaned(path)


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def full_csv_bytes_cached(path: str, stamp: float) -> bytes:
    """The whole cleaned dataset as CSV bytes (built once per file version, not on every rerun)."""
    df = load_cleaned_cached(path, stamp)
    return sup.df_to_csv_bytes(df.assign(date=df["date"].dt.strftime("%Y-%m-%d")))


@st.cache_resource(show_spinner=False)
def load_model_cached(horizon: int, models_dir: str, stamp: float):
    return dfc.load_model(horizon, Path(models_dir))


@st.cache_resource(show_spinner=False)
def load_classifier_cached(lead_time: int, models_dir: str, stamp: float):
    return sp.load_classifier(lead_time, Path(models_dir))


@st.cache_data(show_spinner="Forecasting...")
def forecast_cached(horizon: int, models_dir: str, model_stamp: float, cleaned_path: str, data_stamp: float) -> pd.DataFrame:
    model, _ = load_model_cached(horizon, models_dir, model_stamp)
    return dfc.forecast_future(model, load_cleaned_cached(cleaned_path, data_stamp), horizon)


@st.cache_data(show_spinner="Calculating recommendations...")
def recommendations_cached(cleaned_path: str, stamp: float, params: ro.InventoryParams, snapshot: str | None,
                           on_order_bytes: bytes | None) -> pd.DataFrame:
    cleaned = load_cleaned_cached(cleaned_path, stamp)
    on_order = sup.parse_on_order_upload(on_order_bytes) if on_order_bytes else None
    recs = sup.attach_snapshot_attributes(ro.recommend(cleaned, params, snapshot, on_order), cleaned)
    return dm.attach_price(recs, cleaned)


@st.cache_data(show_spinner="Running lead-time sensitivity...")
def sensitivity_cached(cleaned_path: str, stamp: float, params: ro.InventoryParams, snapshot: str | None) -> pd.DataFrame:
    return ro.lead_time_sensitivity(load_cleaned_cached(cleaned_path, stamp), params, snapshot)


@st.cache_data(show_spinner="Running back-test...")
def backtest_cached(cleaned_path: str, stamp: float, params: ro.InventoryParams) -> dict:
    return ro.service_level_backtest(load_cleaned_cached(cleaned_path, stamp), params)


@st.cache_data(show_spinner="Simulating the inventory policy...")
def policy_backtest_cached(cleaned_path: str, stamp: float, params: ro.InventoryParams) -> dict:
    return ro.policy_backtest(load_cleaned_cached(cleaned_path, stamp), params)


@st.cache_data(show_spinner="Scoring with the simulated-label classifier...")
def classifier_scores_cached(lead_time: int, models_dir: str, model_stamp: float, cleaned_path: str, data_stamp: float,
                             snapshot: str | None) -> pd.DataFrame:
    model, _ = load_classifier_cached(lead_time, models_dir, model_stamp)
    return sp.score_snapshot(model, load_cleaned_cached(cleaned_path, data_stamp), snapshot)


@st.cache_data(show_spinner="Computing insights...")
def insights_cached(cleaned_path: str, stamp: float, filters: dict) -> list[str]:
    return eda.generate_insights(eda.apply_filters(load_cleaned_cached(cleaned_path, stamp), **filters))


@st.cache_data(show_spinner="Running statistical tests...")
def hypothesis_cached(cleaned_path: str, stamp: float, filters: dict) -> pd.DataFrame:
    return dm.hypothesis_test_table(eda.apply_filters(load_cleaned_cached(cleaned_path, stamp), **filters))


@st.cache_data(show_spinner="Drawing chart...")
def eda_figure_cached(name: str, cleaned_path: str, stamp: float, filters: dict, option: str = ""):
    view = eda.apply_filters(load_cleaned_cached(cleaned_path, stamp), **filters)
    if name == "correlation":
        return eda.plot_correlation_heatmap(view, option or "pearson")
    return EDA_FIGURES[name](view)


EDA_FIGURES = {
    "by_product": lambda d: eda.plot_sales_by(d, "product_id"),
    "by_store": lambda d: eda.plot_sales_by(d, "store_id"),
    "by_category": lambda d: eda.plot_sales_by(d, "category"),
    "by_region": lambda d: eda.plot_sales_by(d, "region"),
    "top_slow": eda.plot_top_slow_products,
    "inventory": eda.plot_inventory_distribution,
    "sales_vs_inventory": eda.plot_sales_vs_inventory,
    "price_discount": eda.plot_price_discount,
    "promotion": eda.plot_promo_comparison,
    "histograms": eda.plot_histograms,
    "boxplots": eda.plot_boxplots,
    "outliers": eda.plot_outliers,
    "seasonality": eda.plot_seasonality,
    "missing": eda.plot_missing,
}


# ================================================================ inventory settings (session state)
def keep_widget_state() -> None:
    """Streamlit forgets widget values on pages where the widget is not drawn; this keeps the inventory settings."""
    for k in PERSIST_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def init_inventory_defaults(cleaned: pd.DataFrame) -> None:
    """Default inventory settings (set once; user changes are kept)."""
    first, last = cleaned["date"].min().date(), cleaned["date"].max().date()
    ss = st.session_state
    for k, v in (("inv_lead", 1.0), ("inv_sl", 95.0), ("inv_review", 1.0), ("inv_lookback", 90), ("inv_ss_mode", "Statistical (service level)"),
                 ("inv_ss_units", 100.0), ("inv_ss_days", 1.0), ("inv_moq", 0.0), ("inv_mult", 1), ("inv_excess", 1.0), ("inv_snapshot", last)):
        ss.setdefault(k, v)
    if not (first + dt.timedelta(days=15) <= ss["inv_snapshot"] <= last):
        ss["inv_snapshot"] = last


def build_params() -> tuple[ro.InventoryParams | None, str | None]:
    """Inventory settings from session state (defaults when the widgets were never drawn)."""
    s = st.session_state
    mode = s.get("inv_ss_mode", "Statistical (service level)")
    try:
        params = ro.InventoryParams(
            lead_time_days=float(s.get("inv_lead", 1.0)), service_level=float(s.get("inv_sl", 95.0)) / 100,
            review_period_days=float(s.get("inv_review", 1.0)), lookback_days=int(s.get("inv_lookback", 90)),
            safety_stock_units=float(s.get("inv_ss_units", 0.0)) if mode == "Fixed units" else None,
            safety_stock_days=float(s.get("inv_ss_days", 1.0)) if mode == "Days of demand" else None,
            excess_factor=float(s.get("inv_excess", 1.0)), min_order_qty=float(s.get("inv_moq", 0.0)),
            order_multiple=float(s.get("inv_mult", 1.0)))
        return params, None
    except ValueError as exc:
        return None, str(exc)


def settings_text(params: ro.InventoryParams, snapshot) -> str:
    ss = ("fixed " + f"{params.safety_stock_units:g} units" if params.safety_stock_units is not None
          else f"{params.safety_stock_days:g} days of demand" if params.safety_stock_days is not None else f"statistical at {params.service_level:.1%} service level")
    return (f"Inventory settings: snapshot {snapshot}, lead time {params.lead_time_days:g} day(s), safety stock {ss}, review period {params.review_period_days:g} day(s), "
            f"demand history {params.lookback_days} days, minimum order {params.min_order_qty:g}, pack size {params.order_multiple:g}, excess above {params.excess_factor:g} x target.")


def get_recommendations(paths: sup.AppPaths, on_order_bytes: bytes | None = None, quiet: bool = False) -> tuple[pd.DataFrame | None, ro.InventoryParams | None]:
    params, err = build_params()
    if err:
        if not quiet:
            st.error(f"Invalid inventory settings: {err}")
        return None, None
    snap = st.session_state.get("inv_snapshot")
    try:
        recs = recommendations_cached(str(paths.cleaned_csv), sup.mtime(paths.cleaned_csv), params, str(snap) if snap else None, on_order_bytes)
    except (DataValidationError, ValueError) as exc:
        if not quiet:
            st.warning(str(exc))
        return None, params
    return recs, params


def restrict_recs(recs: pd.DataFrame, filters: dict, quiet: bool = False) -> pd.DataFrame | None:
    try:
        return sup.filter_recommendations(recs, filters["stores"], filters["products"], filters["categories"], filters["regions"])
    except DataValidationError as exc:
        if not quiet:
            st.warning(str(exc) + " Widen the sidebar filters.")
        return None


def filtered_recommendations(ctx: Ctx, quiet: bool = False):
    """(recommendation table restricted to the sidebar filters, params) - either may be None."""
    recs, params = get_recommendations(ctx.paths, None, quiet)
    if recs is not None:
        recs = restrict_recs(recs, ctx.filters, quiet)
    return recs, params


def available_classifier_lead_times(paths: sup.AppPaths) -> list[int]:
    return sorted(int(p.stem.split("_L")[1]) for p in paths.models_dir.glob("stockout_classifier_simulated_L*.joblib")) if paths.models_dir.exists() else []


def forecast_horizons(paths: sup.AppPaths) -> list[int]:
    return dfc.available_horizons(paths.models_dir) if paths.models_dir.exists() else []


def filter_summary(ctx: Ctx) -> list[str]:
    """Plain-language list of the active filters (used in headers and the report)."""
    f, c = ctx.filters, ctx.cleaned
    lines = []
    if f["date_range"] is not None:
        lines.append(f"Date range: {f['date_range'][0]} to {f['date_range'][1]}")
    for key, col, label in (("stores", "store_id", "Stores"), ("products", "product_id", "Products"), ("categories", "category", "Categories"), ("regions", "region", "Regions")):
        total = c[col].nunique()
        if f[key] is None:
            lines.append(f"{label}: all {total}")
        else:
            lines.append(f"{label}: {len(f[key])} of {total} ({', '.join(map(str, f[key][:4]))}{'...' if len(f[key]) > 4 else ''})")
    return lines


__all__ = ["Ctx", "DataFileError", "DataValidationError"]
