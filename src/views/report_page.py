"""Inventory Report: one report built from the app's real numbers, downloadable as Markdown, PDF (optional) and CSV bundle."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src import app_data as ad
from src import app_support as sup
from src import dashboard_metrics as dm
from src import demand_forecasting as dfc
from src import reorder_optimizer as ro
from src import report_generator as rg
from src import stockout_prediction as sp
from src.preprocess import GROUP_KEYS
from src.ui_components import Kpi, callout, kpi_row, page_header, section
from src.utils import DataFileError, DataValidationError
from src.views.forecast_page import _with_attributes


def _forecast_rows(ctx: ad.Ctx) -> list[dict]:
    """Per trained horizon: selected model, test metrics, baseline MAE and the total forecast for the sidebar selection."""
    out = []
    for h in ad.forecast_horizons(ctx.paths):
        mpath = dfc.model_paths(h, ctx.paths.models_dir)[0]
        try:
            _, meta = ad.load_model_cached(h, str(ctx.paths.models_dir), sup.mtime(mpath))
        except (DataFileError, DataValidationError):
            continue
        tm = meta["test_metrics"]
        comp_path = ctx.paths.reports_dir / f"forecast_model_comparison_h{h}.csv"
        base = None
        if comp_path.exists():
            comp = ad.read_csv_cached(str(comp_path), sup.mtime(comp_path))
            if "Series mean" in set(comp["model"]):
                base = float(comp.loc[comp["model"] == "Series mean", "mae_test"].iloc[0])
        row = {"horizon": h, "model": meta["model_name"], "mae": tm["mae"], "rmse": tm["rmse"], "wape": tm["wape"], "r2": tm["r2"], "baseline_mae": base, "total_forecast": None, "target_date": None}
        try:
            fc = _with_attributes(ad.forecast_cached(h, str(ctx.paths.models_dir), sup.mtime(mpath), ctx.path, ctx.stamp), ctx.cleaned)
            fc = sup.filter_recommendations(fc, ctx.filters["stores"], ctx.filters["products"], ctx.filters["categories"], ctx.filters["regions"])
            row.update(total_forecast=float(fc["forecast_units"].sum()), target_date=f"{fc['target_date'].iloc[0]:%Y-%m-%d}")
        except (DataValidationError, DataFileError):
            pass
        out.append(row)
    return out


def _evaluation_lines(ctx: ad.Ctx, params: ro.InventoryParams | None, forecast_rows: list[dict]) -> list[str]:
    lines = []
    for f in forecast_rows:
        gain = f" ({100 * (f['baseline_mae'] - f['mae']) / f['baseline_mae']:+.2f}% MAE vs series-mean baseline {f['baseline_mae']:.2f})" if f.get("baseline_mae") else ""
        lines.append(f"Forecast horizon {f['horizon']} day(s): selected {f['model']}, test MAE {f['mae']:.2f}, RMSE {f['rmse']:.2f}, WAPE {100 * f['wape']:.1f}%, R2 {f['r2']:.3f}{gain}.")
    if forecast_rows:
        lines.append("Machine-learning models do not clearly beat the series-mean baseline on this synthetic data; the simpler model is preferred.")
    for L in ad.available_classifier_lead_times(ctx.paths):
        _, meta = ad.load_classifier_cached(L, str(ctx.paths.models_dir), sup.mtime(sp.classifier_paths(L, ctx.paths.models_dir)[0]))
        t = meta["test_metrics"]
        lines.append(f"SIMULATED-label classifier ({L}-day window, {meta['model_name']}): ROC-AUC {t['roc_auc']:.3f}, precision {t['precision']:.2f}, recall {t['recall']:.2f}, F1 {t['f1']:.2f}. The label is hypothetical.")
    if params is not None:
        try:
            bt = ad.backtest_cached(ctx.path, ctx.stamp, params)
            lines.append(f"Service-level back-test ({bt['test_period'][0]} to {bt['test_period'][1]}, lead time {bt['lead_time_days_used']} d): target {bt['target_service_level']:.1%}, achieved {bt['achieved_service_level']:.1%} over {bt['windows']:,} windows.")
            sim = ad.policy_backtest_cached(ctx.path, ctx.stamp, params)["chosen"]
            lines.append(f"Policy simulation at the {sim['service_level_target']:.1%} target: fill rate {sim['fill_rate']:.1%}, stock-out frequency {sim['stockout_day_frequency']:.1%}, average stock {sim['avg_on_hand_units']:,.0f} units per store-product. "
                         "Back-testing on one synthetic file does not prove real-world performance.")
        except (DataValidationError, ValueError):
            lines.append("Inventory back-test not available for the current data and settings.")
    return lines


def build_data(ctx: ad.Ctx, n_rows: int) -> tuple[dict, dict]:
    """(report data dict, CSV tables dict) from the current filters and settings."""
    recs, params = ad.filtered_recommendations(ctx, quiet=True)
    view = ctx.view
    kpis = dm.home_kpis(view, recs)
    snapshot = str(recs["snapshot_date"].iloc[0].date()) if recs is not None and len(recs) else "n/a"
    kpis["snapshot"] = snapshot
    forecast_rows = _forecast_rows(ctx)
    data = {"generated_at": datetime.now(), "dataset": {"records": len(view), "records_total": len(ctx.cleaned), "date_min": f"{view['date'].min():%Y-%m-%d}", "date_max": f"{view['date'].max():%Y-%m-%d}",
                                                         "stores": int(view["store_id"].nunique()), "products": int(view["product_id"].nunique())},
            "filters": ad.filter_summary(ctx), "settings": ad.settings_text(params, snapshot) if params is not None else None, "kpis": kpis, "forecast": forecast_rows, "evaluation": {"lines": _evaluation_lines(ctx, params, forecast_rows)}}
    tables = {"selected_data_summary": view.groupby(["store_id", "product_id"], observed=True)["units_sold"].agg(total_units="sum", mean_units="mean", days="count").reset_index()}
    if recs is not None:
        k = ro.summarize(recs)
        data["health"] = dm.health_counts(recs)
        data["stockout"] = {"risk_counts_text": ", ".join(f"{a} {b}" for a, b in k["risk_counts"].items() if b or a != "Unknown"), "mean_probability": k["mean_stockout_probability"]}
        L = (ad.available_classifier_lead_times(ctx.paths) or [None])[0]
        if L is not None:
            _, meta = ad.load_classifier_cached(L, str(ctx.paths.models_dir), sup.mtime(sp.classifier_paths(L, ctx.paths.models_dir)[0]))
            data["stockout"]["classifier"] = {"lead_time": L, "model": meta["model_name"], **{k_: meta["test_metrics"][k_] for k_ in ("recall", "precision", "roc_auc")}}
        data["recommendations"] = dm.critical_items(recs, n_rows)
        data["insights"] = dm.generate_actionable_insights(view, recs)
        tables.update({"reorder_recommendations": recs.assign(risk_level=recs["risk_level"].astype(str)), "most_urgent_items": dm.critical_items(recs, n_rows), "inventory_health": data["health"],
                       "insights": pd.DataFrame(data["insights"])})
    tables["forecast_summary"] = pd.DataFrame(forecast_rows)
    return data, tables


def render(ctx: ad.Ctx) -> None:
    page_header("Inventory Report", "A report built from the numbers currently on screen: your filters, your inventory settings and the saved models.", ["Markdown, PDF and CSV", "Synthetic dataset"])
    callout("warn", "The report states clearly that the dataset is **synthetic** and lists the limitations. Sections whose inputs are missing say so instead of showing placeholders.")
    n_rows = st.slider("Rows of reorder recommendations in the report", 5, 50, 15, step=5, key="rep_rows")
    try:
        data, tables = build_data(ctx, n_rows)
        blocks = rg.build_report_blocks(data)
    except (DataValidationError, DataFileError, ValueError) as exc:
        st.error(f"The report could not be built: {exc}")
        return
    md = rg.to_markdown(blocks)
    stamp = datetime.now().strftime("%Y%m%d")
    section("Downloads")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Report (Markdown)", md, f"inventory_report_{stamp}.md", "text/markdown", key="rep_md", use_container_width=True)
    c2.download_button("Data tables (ZIP of CSV files)", rg.csv_bundle(tables, md), f"inventory_report_{stamp}.zip", "application/zip", key="rep_zip", use_container_width=True)
    if rg.pdf_available():
        if c3.toggle("Prepare PDF", key="rep_pdf_prep", help="Builds the PDF only when you ask."):
            try:
                c3.download_button("Report (PDF)", rg.to_pdf(blocks), f"inventory_report_{stamp}.pdf", "application/pdf", key="rep_pdf", use_container_width=True)
            except DataValidationError as exc:
                c3.error(str(exc))
    else:
        c3.caption("PDF export is optional. Install it with  `pip install reportlab`  and reload this page.")
    section("Preview")
    st.markdown(md)
