"""Data Quality and Preprocessing: quality KPIs, missing values, duplicates, leakage rules, preprocessing summary."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import app_charts as charts
from src import app_data as ad
from src import app_support as sup
from src import dashboard_metrics as dm
from src import eda
from src.preprocess import LEAKAGE_COLUMNS, OTHER_EXCLUSIONS
from src.ui_components import Kpi, callout, kpi_row, page_header, show_table


def _affected(quality: pd.DataFrame, step: str, column: str | None = None) -> int:
    q = quality[quality["step"] == step]
    if column is not None:
        q = q[q["column"] == column]
    return int(q["affected_rows"].sum())


def render(paths: sup.AppPaths, cleaned: pd.DataFrame | None) -> None:
    page_header("Data Quality and Preprocessing", "What the cleaning step found and did, how leakage is prevented, and what remains.", ["Synthetic dataset"])
    quality = sup.read_csv_optional(paths.reports_dir / "data_quality_report.csv")
    summary = sup.read_text(paths.reports_dir / "preprocessing_summary.txt")
    inspection = sup.read_text(paths.reports_dir / "dataset_inspection_report.txt")

    if quality is not None:
        result = quality[quality["step"].str.startswith("99")]
        kept = f"{int(result['affected_rows'].iloc[0]):,}" if len(result) else "n/a"
        missing_before = _affected(quality, "1_missing_before_cleaning")
        missing_left = int(cleaned.drop(columns=[c for c in cleaned.columns if c.startswith("outlier_")]).isna().sum().sum()) if cleaned is not None else None
        out_flags = _affected(quality, "13_outliers", "units_sold")
        kpi_row([Kpi("Rows kept", kept, "After all cleaning steps", "green"), Kpi("Duplicates removed", f"{_affected(quality, '2_duplicates') + _affected(quality, '7_duplicate_keys'):,}", "Exact rows and repeated date-store-product keys",
                                                                              "amber" if _affected(quality, "2_duplicates") + _affected(quality, "7_duplicate_keys") else "green"),
                 Kpi("Missing cells before", f"{missing_before:,}", "Raw file", "amber" if missing_before else "green"),
                 Kpi("Missing cells after", "n/a" if missing_left is None else f"{missing_left:,}", "Cleaned file", "green" if not missing_left else "amber"),
                 Kpi("Units-sold outliers flagged", f"{out_flags:,}", "Flagged, not removed", "blue")])
    else:
        callout("info", "No quality report yet. Run:  `python -m src.preprocess`")

    tabs = st.tabs(["Quality report", "Missing values", "Duplicates and outliers", "Leakage and exclusions", "Preprocessing summary", "Raw inspection", "Pipeline status"])
    with tabs[0]:
        if quality is None:
            st.info("No quality report yet. Run:  `python -m src.preprocess`")
        else:
            st.caption("Every check made while cleaning, how many rows it touched and what was done. Rows with 0 mean the check found nothing.")
            only = st.toggle("Show only checks that found something", value=True, key="q_only")
            shown = quality[(quality["affected_rows"] > 0) | ~quality["step"].str.startswith("1_")] if only else quality
            shown = shown[~shown["step"].str.startswith("99")] if only else shown
            show_table(shown, "quality", file_name="data_quality_report.csv", label="Search checks")
    with tabs[1]:
        if cleaned is None:
            st.info("Cleaned data not available.")
        else:
            miss = eda.missing_summary(cleaned, paths.reports_dir / "data_quality_report.csv")
            if int(miss.filter(like="missing").to_numpy().sum()) == 0:
                callout("ok", "**No missing values** in any column, before or after cleaning.")
            st.altair_chart(charts.missing_bar(miss), use_container_width=True)
            st.dataframe(miss, use_container_width=True, hide_index=True)
            st.caption("Missing target values are dropped (never imputed). Other numeric gaps are filled only from past values; anything left is imputed with a training-period median inside the model pipeline.")
    with tabs[2]:
        if cleaned is None:
            st.info("Cleaned data not available.")
        else:
            d = dm.duplicate_analysis(cleaned)
            kpi_row([Kpi("Rows", f"{d['rows']:,}", "Cleaned file"), Kpi("Exact duplicate rows", f"{d['exact_duplicates']:,}", "Should be 0 after cleaning", "green" if not d["exact_duplicates"] else "red"),
                     Kpi("Repeated date-store-product keys", f"{d['duplicate_keys']:,}", "Should be 0 after cleaning", "green" if not d["duplicate_keys"] else "red"), Kpi("Store-product series", f"{d['series']:,}", "Time series in the data")])
            if quality is not None:
                st.markdown("**Duplicate handling during cleaning**")
                st.dataframe(quality[quality["step"].isin(["2_duplicates", "7_duplicate_keys"])], use_container_width=True, hide_index=True)
            st.markdown("**Outliers (flagged with the 1.5 x IQR rule per store-product series)**")
            st.dataframe(eda.outlier_summary(cleaned), use_container_width=True, hide_index=True)
            st.caption("Outliers are kept: unusually high-sales days are valid demand the inventory plan has to cover.")
    with tabs[3]:
        st.markdown("**Columns that must not be used to predict same-day demand (target leakage)**")
        st.dataframe(pd.DataFrame({"column": list(LEAKAGE_COLUMNS), "why": list(LEAKAGE_COLUMNS.values())}), use_container_width=True, hide_index=True)
        st.markdown("**Other columns left out of the forecasting features**")
        st.dataframe(pd.DataFrame({"column": list(OTHER_EXCLUSIONS), "why": list(OTHER_EXCLUSIONS.values())}), use_container_width=True, hide_index=True)
        st.markdown("**How lag features avoid leakage:** each store-product series is ordered by date; lag and rolling features are looked up by calendar date "
                    "and use only days before the target day; train, validation and test periods are consecutive blocks of dates, never shuffled.")
    with tabs[4]:
        st.code(summary, language="text") if summary else st.info("No summary yet. Run:  `python -m src.preprocess`")
    with tabs[5]:
        st.code(inspection, language="text") if inspection else st.info("No inspection report yet. Run:  `python inspect_dataset.py`")
    with tabs[6]:
        status = pd.DataFrame(sup.artifact_status(paths))
        status["ready"] = status["ready"].map({True: "Ready", False: "Missing"})
        st.dataframe(status.rename(columns={"item": "Pipeline output", "ready": "Status", "command": "Command that creates it"}), use_container_width=True, hide_index=True)
