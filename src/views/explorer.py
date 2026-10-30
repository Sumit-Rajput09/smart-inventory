"""Dataset Explorer: search, sort, filter, paginate, column information and downloads."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import app_data as ad
from src import app_support as sup
from src import dashboard_metrics as dm
from src import eda
from src.ui_components import Kpi, callout, kpi_row, page_header
from src.utils import DataValidationError

PAGE_SIZES = [25, 50, 100, 250]
SEARCH_TEXT_COLUMNS = ["date", "store_id", "product_id", "category", "region", "weather", "seasonality"]


def _column_config(frame: pd.DataFrame) -> dict:
    cfg = {}
    for c in frame.columns:
        n_missing = int(frame[c].isna().sum())
        help_text = f"{frame[c].dtype}" + (f" | {n_missing:,} missing" if n_missing else " | no missing values")
        if pd.api.types.is_float_dtype(frame[c]):
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f", help=help_text)
        elif pd.api.types.is_integer_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c]):
            cfg[c] = st.column_config.NumberColumn(c, format="%d", help=help_text)
        else:
            cfg[c] = st.column_config.Column(c, help=help_text)
    return cfg


def render(ctx: ad.Ctx) -> None:
    view = ctx.view
    page_header("Dataset Explorer", "Cleaned data after the sidebar filters. Search, sort, filter by value range, page through the rows and download.",
                [f"{len(view):,} of {len(ctx.cleaned):,} rows", "Synthetic dataset"])
    n_missing = int(view.isna().sum().sum())
    kpi_row([Kpi("Rows after filters", sup.fmt_int(len(view)), f"of {len(ctx.cleaned):,} in the file"), Kpi("Columns", str(view.shape[1]), "Including cleaning flags"),
             Kpi("Stores / products", f"{view['store_id'].nunique()} / {view['product_id'].nunique()}", "In the selected rows"),
             Kpi("Missing cells", f"{n_missing:,}", "After cleaning", "green" if n_missing == 0 else "amber")])
    tab1, tab2, tab3 = st.tabs(["Records", "Column information", "Descriptive statistics"])

    with tab1:
        table = view.assign(date=view["date"].dt.strftime("%Y-%m-%d"))
        big = len(table) > 20000
        text_cols = [c for c in SEARCH_TEXT_COLUMNS if c in table.columns]
        r1 = st.columns([3, 2, 1])
        query = r1[0].text_input("Search records" + (" (text columns and dates)" if big else ""), key="explorer_search", placeholder="Type words, e.g.  S003 toys 2023-05  (all words must match)")
        sort_col = r1[1].selectbox("Sort by", ["(none)", *table.columns], key="explorer_sort")
        ascending = r1[2].radio("Order", ["Ascending", "Descending"], key="explorer_order", horizontal=False) == "Ascending"
        with st.expander("Filter by value range"):
            num_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c]) and not pd.api.types.is_bool_dtype(table[c]) and table[c].nunique() > 1]
            if num_cols:
                col = st.selectbox("Numeric column", num_cols, key="explorer_numcol")
                lo, hi = float(table[col].min()), float(table[col].max())
                rng = st.slider("Range", lo, hi, (lo, hi), key=f"explorer_range_{col}")
                try:
                    table = dm.numeric_range_filter(table, col, *rng)
                except (DataValidationError, ValueError) as exc:
                    st.warning(str(exc))
            else:
                st.caption("No numeric columns vary in the current selection.")
        found = sup.search_dataframe(table, query, text_cols if big else None)
        found = dm.sort_table(found, None if sort_col == "(none)" else sort_col, ascending)
        if found.empty:
            st.info("No rows match the search and filters.")
        else:
            r2 = st.columns([1, 1, 3])
            size = r2[0].selectbox("Rows per page", PAGE_SIZES, index=1, key="explorer_size")
            pages = max(1, -(-len(found) // size))
            page = r2[1].number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="explorer_page")
            chunk, pages = dm.paginate(found, int(page), size)
            r2[2].caption(f"Showing rows {(int(page) - 1) * size + 1:,}-{(int(page) - 1) * size + len(chunk):,} of {len(found):,} (page {int(page)} of {pages:,}). "
                          "Hover a column header to see its data type and missing values.")
            st.dataframe(chunk, use_container_width=True, hide_index=True, column_config=_column_config(found), height=420)
            missing_cols = [c for c in found.columns if found[c].isna().any()]
            st.caption("Missing values: " + (", ".join(f"{c} ({int(found[c].isna().sum()):,})" for c in missing_cols) if missing_cols else "none in the rows shown by the search and filters."))
            if st.toggle("Prepare CSV downloads", key="explorer_prep", help="Building a CSV of the whole dataset takes a moment, so it only happens when you ask."):
                d1, d2 = st.columns(2)
                d1.download_button(f"Download filtered rows ({len(found):,})", sup.df_to_csv_bytes(found), "filtered_inventory_data.csv", "text/csv", key="dl_filtered", use_container_width=True)
                d2.download_button("Download full cleaned dataset", ad.full_csv_bytes_cached(ctx.path, ctx.stamp), "retail_inventory_cleaned.csv", "text/csv", key="dl_cleaned_full", use_container_width=True)

    with tab2:
        info = dm.column_info(view)
        st.dataframe(info, use_container_width=True, hide_index=True, column_config={
            "missing": st.column_config.NumberColumn(format="%d"), "missing %": st.column_config.NumberColumn(format="%.2f%%"), "unique": st.column_config.NumberColumn(format="%d")})
        st.caption("Computed from the rows currently selected. Descriptions of the original Kaggle columns and what this project found are below.")
        with st.expander("What each original column means and what this project found"):
            st.dataframe(pd.DataFrame(sup.COLUMN_DICTIONARY, columns=["Original column", "Meaning", "What this project found"]), use_container_width=True, hide_index=True)

    with tab3:
        try:
            st.dataframe(eda.descriptive_statistics(view).reset_index(names="column"), use_container_width=True, hide_index=True)
            st.caption("std = standard deviation, iqr = Q3 - Q1, cv_percent = std / mean x 100, skewness > 0 means a long right tail.")
        except DataValidationError as exc:
            st.info(str(exc))
