"""Project Documentation / About."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import app_support as sup
from src import reorder_optimizer as ro
from src.preprocess import LEAKAGE_COLUMNS, OTHER_EXCLUSIONS
from src.ui_components import callout, page_header

SYNTHETIC_NOTE = ("The dataset is **synthetic** (Kaggle). Everything shown describes this file only; it is not evidence of real retail "
                  "behaviour and does not demonstrate real business savings.")

ARCHITECTURE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.5;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#CBD5E1", fillcolor="#FFFFFF"];
  edge [color="#64748B"];
  raw [label="Raw CSV\\n(Kaggle, synthetic)", fillcolor="#F1F5F9"];
  pre [label="preprocess.py\\nclean, validate,\\nfeatures, split", fillcolor="#DBEAFE"];
  eda [label="eda.py\\nstatistics, tests", fillcolor="#DBEAFE"];
  fc [label="demand_forecasting.py\\nbaselines, LR, RF, HGB", fillcolor="#EDE9FE"];
  so [label="stockout_prediction.py\\nrule-based + simulated\\nclassifier", fillcolor="#EDE9FE"];
  ro [label="reorder_optimizer.py\\n(s, S) policy,\\nback-test", fillcolor="#DCFCE7"];
  art [label="models/ and reports/\\njoblib, CSV, figures", fillcolor="#F1F5F9"];
  met [label="dashboard_metrics.py\\nreport_generator.py", fillcolor="#FEF3C7"];
  app [label="app.py + views/\\nStreamlit dashboard", fillcolor="#0F172A", fontcolor="#FFFFFF"];
  raw -> pre; pre -> eda; pre -> fc; pre -> so; pre -> ro;
  fc -> art; so -> art; ro -> art; eda -> art;
  pre -> met; ro -> met; art -> app; met -> app;
}
"""


def render(paths: sup.AppPaths) -> None:
    page_header("Project Documentation / About", "Smart Inventory Analytics, Demand Forecasting and Stock-Out Prediction System - an MCA Data Science project.", ["Python", "Streamlit", "scikit-learn", "Synthetic dataset"])
    callout("warn", SYNTHETIC_NOTE)
    tabs = st.tabs(["Overview", "Dataset", "Architecture", "Methodology", "Formulas", "Limitations and future scope", "Install and run"])

    with tabs[0]:
        st.header("Project overview")
        st.markdown("An MCA semester project that analyses retail inventory and historical sales, forecasts product demand, estimates stock-out and overstock risk, and "
                    "suggests replenishment quantities. Everything is computed from the supplied CSV with Python, Pandas, NumPy, Scikit-learn, Altair/Matplotlib and Streamlit; no paid services or credentials are needed.")
        st.header("Problem statement")
        st.markdown("Retailers lose sales when stock runs out and tie up money when they overstock. Given daily sales and stock levels per store and product, "
                    "how much will be demanded, which items are at risk of running out, and how much should be ordered?")
        st.header("Objectives")
        st.markdown("- Clean and validate the inventory data and prevent target leakage.\n- Explore sales, inventory, price and promotion patterns with statistics and hypothesis tests.\n"
                    "- Forecast demand with simple baselines, Linear Regression, Random Forest and Hist Gradient Boosting, evaluated on a chronological test period.\n"
                    "- Estimate stock-out risk with transparent formulas, and demonstrate (clearly labelled as simulated) a stock-out classifier.\n"
                    "- Recommend reorder points, safety stock and order quantities, with configurable lead time and service level.\n- Present everything in a professional dashboard with exportable reports.")

    with tabs[1]:
        st.header("Dataset")
        st.markdown("**Retail Store Inventory Forecasting Dataset** (Kaggle, author Anirudh Chauhan): `https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset`. "
                    "It is synthetic: 73,100 rows, 5 stores x 20 products x 731 days.")
        st.dataframe(pd.DataFrame(sup.COLUMN_DICTIONARY, columns=["Column", "Meaning", "What this project found"]), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.header("System architecture")
        try:
            st.graphviz_chart(ARCHITECTURE_DOT, use_container_width=True)
        except Exception:  # noqa: BLE001 - the diagram is a nicety; never break the page
            st.code(ARCHITECTURE_DOT, language="text")
        st.header("Technology stack")
        st.dataframe(pd.DataFrame([("Language", "Python 3.10 - 3.13"), ("Dashboard", "Streamlit, Altair, Matplotlib/Seaborn"), ("Data", "Pandas, NumPy, SciPy"), ("Machine learning", "scikit-learn, Joblib"),
                                   ("Testing", "pytest and Streamlit AppTest"), ("Reports", "Markdown, CSV, optional PDF with reportlab")], columns=["Layer", "Tools"]), use_container_width=True, hide_index=True)
        st.header("Project structure")
        st.code("smart-inventory/\n|-- app.py                       page routing and layout\n|-- inspect_dataset.py           dataset inspection report\n|-- requirements.txt\n"
                "|-- data/raw, data/processed     input CSV and cleaned/feature files\n|-- src/\n|   |-- data_loader.py           safe CSV loading, column mapping\n|   |-- preprocess.py            cleaning, features, chronological split\n"
                "|   |-- eda.py                   statistics, tests, figures\n|   |-- demand_forecasting.py    baselines, LR, RF, HGB, prediction intervals\n|   |-- stockout_prediction.py   rule-based risk, simulated classifier\n"
                "|   |-- reorder_optimizer.py     replenishment, back-test, policy simulation\n|   |-- dashboard_metrics.py     KPIs, health groups, insights, tests table\n|   |-- report_generator.py      Markdown / PDF / CSV reports\n"
                "|   |-- app_data.py, app_support.py, app_charts.py   cached data access, helpers, charts\n|   |-- ui_theme.py, ui_components.py                design system\n|   |-- views/                   one module per dashboard page\n"
                "|-- models/                      saved models + metadata\n|-- reports/                     reports, tables, figures\n|-- tests/                       automated tests", language="text")

    with tabs[3]:
        st.header("Data preprocessing")
        st.markdown("Parse dates; validate numbers; drop exact duplicates and repeated keys; flag (not delete) outliers; fill gaps only from past values; never impute the target.")
        st.markdown("**Leakage control** - columns that must not predict same-day demand:")
        st.dataframe(pd.DataFrame({"column": list(LEAKAGE_COLUMNS), "why": list(LEAKAGE_COLUMNS.values())}), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame({"column": list(OTHER_EXCLUSIONS), "why": list(OTHER_EXCLUSIONS.values())}), use_container_width=True, hide_index=True)
        st.header("Forecasting methodology")
        st.markdown("""
- **Task:** predict units sold of a store-product on day *t* from information up to day *t - h* (h = horizon).
- **Features:** calendar fields, lags (1, 7, 14 days), rolling mean and standard deviation (7, 14, 28 days) looked up by calendar date from earlier days only, price and promotion fields.
- **Split:** consecutive train / validation / test blocks of dates; never shuffled. Models are fitted on train and scored on validation; the model is chosen with a simplicity rule (simplest model within 1% of the best validation MAE), refitted on train + validation, and scored once on test, on the same observations for every model.
- **Candidates:** global mean, series mean, naive and seasonal-naive, moving averages, Linear Regression, Random Forest, Hist Gradient Boosting.
- **Metrics:** MAE, RMSE, WAPE (sum of absolute errors / sum of actual; safe with zero demand), R2 and bias.
- **Prediction intervals:** empirical quantiles of held-out residuals (split-conformal style), not an assumed distribution. They assume future errors resemble past test errors.
- **Honest outcome:** on this synthetic data no model clearly beats the series mean, so the simple model is selected.""")
        st.header("Stock-out prediction methodology")
        st.markdown("- **Rule-based:** probability that demand during the lead time exceeds the inventory position, with demand ~ Normal(d x L, sigma x sqrt(L)). Levels: Low (probability at most 1 - service level), Medium (below 20%), High (20% to below 50%), Critical (50% or more).\n"
                    "- **Simulated-label classifier (experiment):** the label is hypothetical (recorded sales over the next L days exceed today's stock, assuming no delivery). Logistic Regression and Random Forest are compared with the rule on the same label. "
                    "It does **not** predict real historical stock-outs, and its scores are not calibrated probabilities.")
        st.header("Model evaluation methodology")
        st.markdown("Chronological validation and test, identical test observations for all models, a simple baseline always shown, paired bootstrap intervals against the baseline, residual analysis, and for the classifier ROC/PR curves, confusion matrices and a calibration check. "
                    "Inventory policies are back-tested on the last 90 days using parameters estimated before them.")

    with tabs[4]:
        st.header("Inventory optimization formulas")
        st.latex(r"\text{Lead-time demand} = d \times L")
        st.latex(r"\text{Safety stock} = z \times \sigma \times \sqrt{L}")
        st.latex(r"\text{Reorder point (ROP)} = \text{Lead-time demand} + \text{Safety stock}")
        st.latex(r"\text{Inventory position} = \text{Stock on hand} + \text{On-order quantity}")
        st.latex(r"\text{Target inventory} = \text{ROP} + d \times R")
        st.latex(r"\text{Suggested order quantity} = \max(0,\ \text{Target inventory} - \text{Inventory position})")
        st.markdown("""**Conditions used in the implementation**
- d and sigma come from the `lookback` days (default 90) before the snapshot day; fewer than 14 days of history means *Insufficient history* and no recommendation.
- z is the normal quantile of the service level (95% gives 1.645). Safety stock can instead be fixed units or days of demand.
- Reorder when the **inventory position is at or below the ROP**. A positive order is raised to the minimum order quantity (if set) and rounded up to whole packs.
- *Excess* means the position is above the excess factor x target inventory.
- Lead time, on-order quantities, minimum order sizes and costs are not in the data: they are settings or assumptions.""")
        with st.expander("Assumptions and limitations of the replenishment policy"):
            st.text(ro.ASSUMPTIONS)

    with tabs[5]:
        st.header("Limitations")
        st.markdown("- The data is synthetic; day-to-day sales behave like random noise, so no model beats a simple average. This is reported honestly.\n"
                    "- Lead times, quantities on order, minimum order quantities, shelf life and costs are not in the data; they are settings or assumptions.\n"
                    "- Estimated inventory value uses the dataset's price field, not an acquisition cost.\n"
                    "- The classifier uses a simulated label and is not evidence of real stock-out prediction.\n- Probabilities, intervals and quantities are estimates, not guarantees.")
        st.header("Future scope")
        st.markdown("Real sales and stock-out history, supplier lead-time data, cost and shelf-life constraints, probabilistic or hierarchical forecasting, multi-echelon replenishment, probability calibration on a dedicated period and automated data refresh.")

    with tabs[6]:
        st.header("Installation instructions")
        st.code("cd D:\\PROJECT\\smart-inventory\npython -m venv venv\n.\\venv\\Scripts\\Activate.ps1\npip install -r requirements.txt\n# optional, only for the PDF report:\npip install reportlab", language="powershell")
        st.header("Execution instructions")
        st.code("python inspect_dataset.py\npython -m src.preprocess\npython -m src.eda\npython -m src.demand_forecasting\npython -m src.stockout_prediction\npython -m src.reorder_optimizer\npython -m pytest -q\nstreamlit run app.py", language="powershell")
        st.caption("Re-run `python -m src.demand_forecasting` and `python -m src.stockout_prediction` once after upgrading to Phase 6: they add Hist Gradient Boosting and save the test predictions used by the new charts.")
        st.header("Pipeline status")
        status = pd.DataFrame(sup.artifact_status(paths))
        status["ready"] = status["ready"].map({True: "Ready", False: "Missing"})
        st.dataframe(status.rename(columns={"item": "Pipeline output", "ready": "Status", "command": "Command that creates it"}), use_container_width=True, hide_index=True)
