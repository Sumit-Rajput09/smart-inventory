"""Inventory report: built once as a list of blocks from real values, rendered as Markdown, PDF (optional) and a CSV bundle.

Nothing is invented: a section whose inputs are missing says so and names the command that creates them.
PDF export needs the optional `reportlab` package; Markdown and CSV exports use only the standard library and pandas.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from src.utils import DataValidationError

TITLE = "Smart Inventory Analytics, Demand Forecasting and Stock-Out Prediction System"
SYNTHETIC_STATEMENT = ("The dataset behind this report is SYNTHETIC (Kaggle retail inventory file). Every number describes that file only; "
                       "it is not evidence of real retail behaviour and does not demonstrate real business savings.")
LIMITATIONS = [
    "The data is synthetic: daily sales behave like random noise around an average, so forecasting models cannot beat a simple average by a meaningful margin.",
    "Supplier lead times, quantities on order, minimum order quantities and costs are not in the data; they are user settings or assumptions.",
    "Estimated inventory value uses the dataset's price field; it is not an acquisition cost or accounting value.",
    "Rule-based stock-out probabilities assume normally distributed, independent daily demand; they are screening scores, not exact chances.",
    "The classifier is trained on a SIMULATED label (recorded sales over the lead time exceed today's stock, assuming no delivery). It does not predict real historical stock-outs.",
    "Back-tests and simulations on one synthetic file illustrate how a policy behaves; they do not prove real-world performance.",
]

Block = tuple  # ("h1"|"h2"|"p"|"bullets"|"table"|"note", payload)


def _num(x, digits: int = 1, default: str = "n/a") -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return f"{x:,.{digits}f}"
    except (TypeError, ValueError):
        return default


def _pct(x, digits: int = 1, default: str = "n/a") -> str:
    return default if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def _table(df: pd.DataFrame, max_rows: int = 15, floats: int = 1) -> pd.DataFrame:
    t = df.head(max_rows).copy()
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].map(lambda v: _num(v, floats))
        else:
            t[c] = t[c].astype(str)
    return t


# ================================================================ content
def build_report_blocks(data: dict) -> list[Block]:
    """data keys (all optional except dataset): dataset, filters, settings, kpis, health, forecast, stockout, recommendations,
    insights, evaluation, missing. Returns a list of blocks."""
    ds = data.get("dataset")
    if not ds:
        raise DataValidationError("The report needs dataset information.")
    b: list[Block] = [("h1", TITLE), ("p", f"Inventory report generated {data.get('generated_at') or datetime.now():%Y-%m-%d %H:%M}"),
                      ("note", SYNTHETIC_STATEMENT)]

    b += [("h2", "1. Dataset information"),
          ("bullets", [f"Records in selection: {ds['records']:,} of {ds['records_total']:,}", f"Period in selection: {ds['date_min']} to {ds['date_max']}",
                       f"Stores: {ds['stores']}  |  Products: {ds['products']}", f"Source: {ds.get('source', 'Kaggle Retail Store Inventory Forecasting Dataset (synthetic)')}"])]
    b += [("h2", "2. Selected filters and settings"), ("bullets", list(data.get("filters") or ["No filters applied (all data)."]) + ([data["settings"]] if data.get("settings") else []))]

    k = data.get("kpis")
    b.append(("h2", "3. Inventory summary"))
    if k and "inventory_units" in k:
        lines = [f"Snapshot: {k['snapshot']} (stock is read at the start of this day)", f"Store-products assessed: {k['series']}",
                 f"Stock on hand: {_num(k['inventory_units'], 0)} units", f"Units sold in the selected period: {_num(k['units_sold'], 0)}",
                 f"Products requiring reorder: {k['reorder_now']} of {k['series']} (suggested {k['units_to_order']:,} units)",
                 f"Critical stock-out risk: {k['critical']}  |  Excess inventory: {k['excess']}  |  Healthy: {k['healthy']}"]
        if k.get("inventory_value") is not None:
            lines.append(f"Estimated inventory value (stock on hand x recorded price; not acquisition cost): {_num(k['inventory_value'], 0)}")
        b.append(("bullets", lines))
        if data.get("health") is not None:
            h = data["health"].copy()
            h["share"] = h["share"].map(_pct)
            b.append(("table", h.rename(columns={"group": "Inventory health", "store_products": "Store-products", "share": "Share"})))
    else:
        b.append(("p", "Inventory snapshot not available for this selection."))

    b.append(("h2", "4. Forecast summary"))
    fc = data.get("forecast") or []
    if fc:
        rows = [{"Horizon (days)": f["horizon"], "Selected model": f["model"], "Test MAE": _num(f["mae"], 2), "Test RMSE": _num(f["rmse"], 2), "Test WAPE": _pct(f["wape"]),
                 "Test R2": _num(f["r2"], 3), "Series-mean MAE": _num(f.get("baseline_mae"), 2),
                 "Forecast for selection": (f"{_num(f['total_forecast'], 0)} units on {f['target_date']}" if f.get("total_forecast") is not None else "n/a")} for f in fc]
        b.append(("table", pd.DataFrame(rows)))
        b.append(("p", "Test metrics are measured on the final chronological test period, never on data the model was fitted on. "
                       "R2 close to 0 means the model explains almost none of the day-to-day variation."))
    else:
        b.append(("p", "No trained forecasting model found. Run: python -m src.demand_forecasting"))

    b.append(("h2", "5. Stock-out risk summary"))
    so = data.get("stockout")
    if so:
        b.append(("bullets", [f"Rule-based risk levels (formula estimate): {so['risk_counts_text']}", f"Mean estimated stock-out probability: {_pct(so['mean_probability'])}",
                              "Rule-based risk is a formula, not an observed outcome."]))
        if so.get("classifier"):
            c = so["classifier"]
            b.append(("note", f"SIMULATED-LABEL EXPERIMENT ({c['lead_time']}-day window, model {c['model']}): recall {_num(c['recall'], 2)}, precision {_num(c['precision'], 2)}, "
                              f"ROC-AUC {_num(c['roc_auc'], 3)}. The label is hypothetical; these are not real stock-out results."))
    else:
        b.append(("p", "Stock-out risk not available for this selection."))

    b.append(("h2", "6. Reorder recommendations (most urgent)"))
    rec = data.get("recommendations")
    if rec is not None and len(rec):
        b.append(("table", _table(rec, 15, 1)))
        b.append(("p", "Reorder when the inventory position (stock on hand + on order) is at or below the reorder point; order up to the target inventory."))
    else:
        b.append(("p", "No recommendations available for this selection."))

    b.append(("h2", "7. Important insights"))
    ins = data.get("insights") or []
    b.append(("bullets", [f"{i['title']}: {i['detail']}" for i in ins] if ins else ["No insights could be generated for this selection."]))

    b.append(("h2", "8. Model evaluation summary"))
    ev = data.get("evaluation") or {}
    lines = list(ev.get("lines") or [])
    b.append(("bullets", lines if lines else ["No evaluation results available."]))

    b += [("h2", "9. Dataset and method limitations"), ("bullets", LIMITATIONS)]
    return b


# ================================================================ renderers
def to_markdown(blocks: list[Block]) -> str:
    out: list[str] = []
    for kind, payload in blocks:
        if kind == "h1":
            out.append(f"# {payload}\n")
        elif kind == "h2":
            out.append(f"\n## {payload}\n")
        elif kind == "p":
            out.append(f"{payload}\n")
        elif kind == "note":
            out.append(f"> **{payload}**\n")
        elif kind == "bullets":
            out.append("\n".join(f"- {x}" for x in payload) + "\n")
        elif kind == "table":
            t = payload.astype(str)
            head = "| " + " | ".join(t.columns) + " |\n| " + " | ".join("---" for _ in t.columns) + " |\n"
            out.append(head + "\n".join("| " + " | ".join(v.replace("|", "/") for v in row) + " |" for row in t.to_numpy()) + "\n")
    return "\n".join(out)


def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def _latin(text: str) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


def to_pdf(blocks: list[Block]) -> bytes:
    """PDF bytes via reportlab. Raises DataValidationError with install instructions when reportlab is missing."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise DataValidationError("PDF export needs the optional 'reportlab' package. Install it with:  pip install reportlab") from exc

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
    cell = ParagraphStyle("cell", parent=body, fontSize=7, leading=8.5)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, leading=20, alignment=0, textColor=colors.HexColor("#0F172A"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#1D4ED8"), spaceBefore=10)
    note = ParagraphStyle("note", parent=body, backColor=colors.HexColor("#FFFBEB"), borderColor=colors.HexColor("#D97706"), borderWidth=0.8, borderPadding=5, spaceBefore=4, spaceAfter=6)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm, title=_latin(TITLE))
    story = []
    for kind, payload in blocks:
        if kind == "h1":
            story.append(Paragraph(escape(_latin(payload)), h1))
        elif kind == "h2":
            story.append(Paragraph(escape(_latin(payload)), h2))
        elif kind == "p":
            story.append(Paragraph(escape(_latin(payload)), body))
        elif kind == "note":
            story.append(Paragraph("<b>" + escape(_latin(payload)) + "</b>", note))
        elif kind == "bullets":
            story.append(ListFlowable([ListItem(Paragraph(escape(_latin(x)), body), leftIndent=10) for x in payload], bulletType="bullet", start="-", leftIndent=10))
        elif kind == "table":
            t = payload.astype(str)
            data = [[Paragraph("<b>" + escape(_latin(c)) + "</b>", cell) for c in t.columns]] + [[Paragraph(escape(_latin(v)), cell) for v in row] for row in t.to_numpy()]
            width = A4[0] - 32 * mm
            tbl = Table(data, colWidths=[width / len(t.columns)] * len(t.columns), repeatRows=1)
            tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                                     ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [Spacer(1, 3), tbl, Spacer(1, 4)]
    doc.build(story)
    return buf.getvalue()


def csv_bundle(tables: dict[str, pd.DataFrame], markdown: str | None = None) -> bytes:
    """ZIP of CSV files (and the Markdown report if given). Empty/None tables are skipped."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            if df is not None and len(df):
                z.writestr(f"{name}.csv", df.to_csv(index=False))
        if markdown:
            z.writestr("inventory_report.md", markdown)
    return buf.getvalue()


def forecast_report_markdown(info: dict, table: pd.DataFrame, max_rows: int = 25) -> str:
    """Downloadable forecast report from real values. `info`: horizon, model, target_date, last_observed, selection (text), n_series,
    total_forecast, mean_forecast, interval (dict or None), metrics (dict), baseline_mae, tolerance_rule."""
    lines = [f"# Demand forecast report", "", f"> **{SYNTHETIC_STATEMENT}**", "",
             f"- Forecast horizon: {info['horizon']} day(s) after the last observed date ({info['last_observed']})",
             f"- Forecast date: {info['target_date']}", f"- Model: {info['model']} ({info.get('selection_rule', 'selected on validation data')})",
             f"- Selection: {info['selection']} ({info['n_series']} store-products)",
             f"- Total forecast units for the selection: {_num(info['total_forecast'], 1)}  (mean per store-product {_num(info['mean_forecast'], 1)})"]
    iv = info.get("interval")
    if iv:
        lines.append(f"- {iv['coverage']:.0%} empirical prediction interval for the total: {_num(iv['lower'], 0)} to {_num(iv['upper'], 0)} "
                     f"(from {iv['n_residuals']} held-out daily residuals; method: split-conformal style quantiles of actual - predicted on the test period)")
    else:
        lines.append("- Prediction interval: not available (saved test predictions missing or too few residuals).")
    m = info.get("metrics") or {}
    lines += ["", "## Accuracy on the held-out test period", f"- MAE {_num(m.get('mae'), 2)}, RMSE {_num(m.get('rmse'), 2)}, WAPE {_pct(m.get('wape'))}, R2 {_num(m.get('r2'), 3)}",
              f"- Series-mean baseline MAE on the same period: {_num(info.get('baseline_mae'), 2)}",
              "- R2 close to 0 means the model explains almost none of the day-to-day variation; this is expected for this synthetic data.", "",
              "## Assumptions", "- Price, discount and promotion of the forecast day are assumed equal to the latest observed values.",
              "- The interval assumes future errors resemble the held-out errors; it cannot capture structural change.", "",
              f"## Forecasts (first {min(max_rows, len(table))} of {len(table)} rows; the CSV has all rows)", ""]
    lines.append(to_markdown([("table", _table(table, max_rows, 1))]))
    return "\n".join(lines) + "\n"
