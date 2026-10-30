"""Sidebar: brand, navigation, dataset/model status, filters, inventory settings and the synthetic-data disclaimer."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src import app_data as ad
from src import app_support as sup
from src.ui_components import badge_html
from src.utils import DataValidationError

FILTER_KEYS = ["f_dates", "f_stores", "f_products", "f_categories", "f_regions"]
PAGES_WITH_INVENTORY_SETTINGS = {"Home / Executive Summary", "Stock-Out Risk", "Reorder Recommendations", "Model Evaluation", "Inventory Report"}


def brand() -> None:
    st.sidebar.markdown('<div class="si-brand">Smart Inventory Analytics<small>Forecasting &middot; Stock-out risk &middot; Replenishment</small></div>', unsafe_allow_html=True)


def navigation() -> str:
    def label(p: str) -> str:
        return f":material/{sup.PAGE_ICONS.get(p, 'chevron_right')}: {p}"
    return st.sidebar.radio("Navigate", sup.PAGES, key="nav_page", format_func=label, label_visibility="collapsed")


def status_badges(paths: sup.AppPaths, cleaned: pd.DataFrame | None) -> None:
    """Dataset and model status indicators (what exists on disk right now)."""
    if cleaned is not None:
        data = badge_html(f"Data ready - {len(cleaned):,} rows", "green")
    else:
        data = badge_html("Data missing", "red")
    horizons = ad.forecast_horizons(paths)
    clf = ad.available_classifier_lead_times(paths)
    fc = badge_html(f"Forecast models: {', '.join(f'{h}d' for h in horizons)}", "purple") if horizons else badge_html("Forecast models: none", "amber")
    cl = badge_html("Classifier (simulated): ready", "purple") if clf else badge_html("Classifier: not trained", "amber")
    st.sidebar.markdown(f'<div class="si-status">{data}{fc}{cl}</div>', unsafe_allow_html=True)


def reset_filters() -> None:
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


def selection_or_none(selected: list, options: list):
    return None if set(selected) == set(options) else selected


def filters(cleaned: pd.DataFrame) -> dict:
    """Global filters (date, store, product, category, region). Returns kwargs for eda.apply_filters."""
    with st.sidebar.expander("Filters", expanded=True):
        dmin, dmax = cleaned["date"].min().date(), cleaned["date"].max().date()
        dates = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="f_dates")
        if not isinstance(dates, (tuple, list)) or len(dates) != 2:
            st.caption("Pick an end date to apply the range.")
            dates = (dmin, dmax)
        out = {"date_range": (dates[0], dates[1])}
        for field, col, label in (("stores", "store_id", "Stores"), ("products", "product_id", "Products"),
                                  ("categories", "category", "Categories"), ("regions", "region", "Regions")):
            options = sorted(cleaned[col].astype(str).unique())
            chosen = st.multiselect(label, options, default=options, key=f"f_{field}")
            out[field] = selection_or_none(chosen, options)
        st.button("Reset filters", on_click=reset_filters, key="reset_filters", use_container_width=True)
        st.caption("Category and region change from row to row in this dataset, so they filter rows, not products. "
                   "Stock snapshots (reorder, stock-out) use the snapshot date below and ignore the date range.")
    return out


def inventory_settings(cleaned: pd.DataFrame, expanded: bool = True) -> None:
    first, last = cleaned["date"].min().date(), cleaned["date"].max().date()
    ss = st.session_state
    ad.init_inventory_defaults(cleaned)
    with st.sidebar.expander("Inventory settings", expanded=expanded):
        st.slider("Supplier lead time (days)", 0.5, 14.0, step=0.5, key="inv_lead",
                  help="Not in the dataset: this is YOUR assumption. Stock on hand covers only about 2 days of demand here.")
        st.slider("Service level (%)", 80.0, 99.9, step=0.1, key="inv_sl", help="Target chance of not running out during the lead time.")
        st.slider("Review period (days of demand per order)", 0.0, 14.0, step=0.5, key="inv_review")
        st.radio("Safety stock", ["Statistical (service level)", "Fixed units", "Days of demand"], key="inv_ss_mode")
        if ss["inv_ss_mode"] == "Fixed units":
            st.number_input("Safety stock (units)", min_value=0.0, step=10.0, key="inv_ss_units")
        elif ss["inv_ss_mode"] == "Days of demand":
            st.number_input("Safety stock (days of average demand)", min_value=0.0, step=0.5, key="inv_ss_days")
        st.date_input("Snapshot date (start of day)", min_value=first + dt.timedelta(days=15), max_value=last, key="inv_snapshot",
                      help="Stock is read on this day; demand is estimated only from earlier days.")
        st.number_input("Demand history used (days)", min_value=14, max_value=365, step=7, key="inv_lookback")
        st.markdown("**Optional order rules**")
        st.number_input("Minimum order quantity (units)", min_value=0.0, step=10.0, key="inv_moq", help="Not in the dataset; applied only if you set it.")
        st.number_input("Pack size (order in multiples of)", min_value=1, step=1, key="inv_mult")
        st.number_input("Excess = position above target x", min_value=1.0, step=0.1, key="inv_excess")


def disclaimer() -> None:
    st.sidebar.markdown('<div class="si-note"><b>Synthetic dataset.</b> Results describe this file only and do not show real business savings.</div>', unsafe_allow_html=True)


def apply_view(cleaned: pd.DataFrame, flt: dict) -> pd.DataFrame | None:
    from src import eda
    try:
        return eda.apply_filters(cleaned, **flt)
    except DataValidationError as exc:
        st.warning(str(exc))
        return None
