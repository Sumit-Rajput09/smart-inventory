"""Smart Inventory Analytics, Demand Forecasting and Stock-Out Prediction System - Streamlit dashboard.

Run from the project root:
    streamlit run app.py

This file only routes between pages. The pages live in src/views/, cached data access in src/app_data.py, calculations in
src/dashboard_metrics.py and the design system in src/ui_theme.py + src/ui_components.py.
The dashboard only DISPLAYS results computed from your CSV by the project modules in src/: nothing is hard-coded or invented,
and when a file or model is missing it says which command creates it.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import app_data as ad
from src import app_support as sup
from src.ui_components import inject_theme, page_header
from src.views import about_page, eda_page, evaluation_page, explorer, forecast_page, home, quality, reorder_page, report_page, sidebar, stockout_page

st.set_page_config(page_title="Smart Inventory Analytics", page_icon=":package:", layout="wide")
inject_theme()

PAGE_RENDERERS = {
    "Home / Executive Summary": home.render,
    "Dataset Explorer": explorer.render,
    "Exploratory Data Analysis": eda_page.render,
    "Demand Forecasting": forecast_page.render,
    "Stock-Out Risk": stockout_page.render,
    "Reorder Recommendations": reorder_page.render,
    "Model Evaluation": evaluation_page.render,
    "Inventory Report": report_page.render,
}
ABOUT_PAGE, QUALITY_PAGE = "Project Documentation / About", "Data Quality and Preprocessing"


def missing_artifacts_message(paths: sup.AppPaths) -> None:
    page_header("Smart Inventory Analytics", "The cleaned dataset has not been created yet.")
    st.error("The cleaned dataset has not been created yet, so there is nothing to show.")
    st.markdown("Run these commands in the VS Code terminal (project folder, virtual environment active), then refresh this page:")
    st.code("python inspect_dataset.py\npython -m src.preprocess", language="powershell")
    st.caption(f"Expected file: {paths.cleaned_csv}  |  Raw data: {paths.raw_csv}")
    status = pd.DataFrame(sup.artifact_status(paths))
    status["ready"] = status["ready"].map({True: "Ready", False: "Missing"})
    st.dataframe(status.rename(columns={"item": "Pipeline output", "ready": "Status", "command": "Command that creates it"}), use_container_width=True, hide_index=True)


def main() -> None:
    paths = sup.get_paths()
    ad.keep_widget_state()
    sidebar.brand()
    page = sidebar.navigation()
    cleaned = ad.load_cleaned_cached(str(paths.cleaned_csv), sup.mtime(paths.cleaned_csv)) if paths.cleaned_csv.exists() else None
    sidebar.status_badges(paths, cleaned)

    if page == ABOUT_PAGE:
        sidebar.disclaimer()
        about_page.render(paths)
        return
    if cleaned is None:
        st.sidebar.warning("Cleaned data not found.")
        sidebar.disclaimer()
        if page == QUALITY_PAGE:
            quality.render(paths, None)
        else:
            missing_artifacts_message(paths)
        return

    flt = sidebar.filters(cleaned)
    if page in sidebar.PAGES_WITH_INVENTORY_SETTINGS:
        sidebar.inventory_settings(cleaned, expanded=page != sup.PAGES[0])
    sidebar.disclaimer()
    view = sidebar.apply_view(cleaned, flt)
    if view is None:
        st.stop()

    if page == QUALITY_PAGE:
        quality.render(paths, cleaned)
        return
    PAGE_RENDERERS[page](ad.Ctx(paths=paths, cleaned=cleaned, view=view, filters=flt))


main()
