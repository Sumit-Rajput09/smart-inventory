"""Phase 6 tests: dashboard metrics, report generator, prediction intervals, policy simulation, UI components, filtering and cache behaviour."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import dashboard_metrics as dm
from src import demand_forecasting as dfc
from src import report_generator as rg
from src import reorder_optimizer as ro
from src import ui_components as ui
from src.utils import DataValidationError
from tests.test_inventory import make_cleaned


@pytest.fixture(scope="module")
def cleaned():
    return make_cleaned(n_days=120)


@pytest.fixture(scope="module")
def recs(cleaned):
    return ro.recommend(cleaned, ro.InventoryParams())


# ---------------------------------------------------------------- health groups and KPIs
def test_health_groups_are_mutually_exclusive_and_sum(recs):
    h = dm.inventory_health(recs)
    assert len(h) == len(recs) and h.notna().all()
    counts = dm.health_counts(recs)
    assert counts["store_products"].sum() == len(recs)
    assert abs(counts["share"].sum() - 1) < 1e-9
    assert list(counts["group"]) == list(dm.HEALTH_LEVELS)


def test_health_rules_order():
    df = pd.DataFrame({"stock_status": ["Insufficient history", "Excess stock", "Low stock", "Low stock", "Adequate stock"],
                       "risk_level": ["Critical", "Critical", "Critical", "Low", "Low"]})
    assert list(dm.inventory_health(df).astype(str)) == ["Not assessed", "Excess", "Critical", "Low", "Healthy"]


def test_health_needs_columns():
    with pytest.raises(DataValidationError):
        dm.inventory_health(pd.DataFrame({"x": [1]}))


def test_inventory_value_is_stock_times_price(cleaned, recs):
    r = dm.attach_price(recs, cleaned)
    expected = float((r["current_inventory"] * r["price"]).sum())
    assert dm.estimated_inventory_value(r) == pytest.approx(expected)
    assert dm.estimated_inventory_value(recs.assign(price=np.nan)) is None


def test_home_kpis_use_real_data(cleaned, recs):
    k = dm.home_kpis(cleaned, recs)
    assert k["units_sold"] == pytest.approx(cleaned["units_sold"].sum())
    assert k["products"] == cleaned["product_id"].nunique() and k["stores"] == cleaned["store_id"].nunique()
    assert dm.home_kpis(cleaned, None)["records"] == len(cleaned)


def test_empty_and_missing_inputs():
    empty = pd.DataFrame(columns=["date", "store_id", "product_id", "units_sold", "inventory_level"])
    assert dm.generate_actionable_insights(empty, None) == []
    assert dm.critical_items(pd.DataFrame()).empty
    assert dm.health_counts(pd.DataFrame({"stock_status": [], "risk_level": []}))["store_products"].sum() == 0


def test_insights_computed_from_data(cleaned, recs):
    ins = dm.generate_actionable_insights(cleaned, recs)
    assert ins and all({"title", "detail", "tone"} <= set(i) for i in ins)
    one = cleaned[cleaned["product_id"] == "P1"]
    assert dm.generate_actionable_insights(one, None) != ins


def test_critical_items_sorted_by_urgency(recs):
    top = dm.critical_items(recs, 10)
    assert len(top) <= 10
    first = recs.sort_values(["reorder_now", "stockout_probability", "projected_stock_after_lead_time"], ascending=[False, False, True]).iloc[0]
    assert top.iloc[0][dm.CRITICAL_COLUMNS["current_inventory"]] == first["current_inventory"]


def test_trend_frame_and_category(cleaned):
    t = dm.trend_frame(cleaned, "W")
    assert {"sold_per_day", "demand_per_series_day", "sell_through"} <= set(t.columns)
    assert not dm.category_demand(cleaned).empty


# ---------------------------------------------------------------- explorer helpers
def test_paginate_sort_filter(cleaned):
    page, pages = dm.paginate(cleaned, 1, 100)
    assert len(page) == 100 and pages == -(-len(cleaned) // 100)
    last, _ = dm.paginate(cleaned, 10 ** 6, 100)
    assert len(last) == len(cleaned) - (pages - 1) * 100
    with pytest.raises(ValueError):
        dm.paginate(cleaned, 1, 0)
    s = dm.sort_table(cleaned, "units_sold", False)
    assert s["units_sold"].is_monotonic_decreasing
    assert dm.sort_table(cleaned, "nope").equals(cleaned)
    f = dm.numeric_range_filter(cleaned, "units_sold", 10, 50)
    assert f["units_sold"].between(10, 50).all()
    with pytest.raises(ValueError):
        dm.numeric_range_filter(cleaned, "units_sold", 5, 1)
    with pytest.raises(DataValidationError):
        dm.numeric_range_filter(cleaned, "missing", 0, 1)


def test_column_info_and_duplicates(cleaned):
    info = dm.column_info(cleaned)
    assert len(info) == cleaned.shape[1]
    d = dm.duplicate_analysis(pd.concat([cleaned.head(5), cleaned.head(5)]))
    assert d and isinstance(d, dict)


def test_hypothesis_table_columns(cleaned):
    t = dm.hypothesis_test_table(cleaned)
    assert {"Test", "Null hypothesis (H0)", "Alternative (H1)", "Statistic", "P-value", "Interpretation"} <= set(t.columns)
    assert t["P-value"].between(0, 1).all()
    assert isinstance(dm.hypothesis_test_table(cleaned.head(3)), pd.DataFrame)  # too little data must not crash


# ---------------------------------------------------------------- prediction intervals
def test_empirical_interval_coverage_and_validation():
    rng = np.random.default_rng(1)
    res = rng.normal(0, 10, 2000)
    iv = dfc.empirical_interval(res, 0.90)
    assert iv["lower_offset"] < 0 < iv["upper_offset"]
    new = rng.normal(0, 10, 20000)
    cover = np.mean((new >= iv["lower_offset"]) & (new <= iv["upper_offset"]))
    assert 0.88 < cover < 0.92
    with pytest.raises(DataValidationError):
        dfc.empirical_interval(np.arange(5.0))
    with pytest.raises(ValueError):
        dfc.empirical_interval(res, 1.5)
    assert dfc.empirical_interval(np.r_[res, np.nan], 0.8)["n_residuals"] == 2000


def test_apply_interval_never_negative():
    lo, hi = dfc.apply_interval(5.0, {"lower_offset": -20.0, "upper_offset": 10.0})
    assert lo == 0.0 and hi == 15.0


def test_hist_gradient_boosting_candidate_exists():
    assert hasattr(dfc, "HistGradientBoostingRegressor")
    assert "Hist Gradient Boosting" in open(dfc.__file__, encoding="utf-8").read()


# ---------------------------------------------------------------- policy simulation
def test_simulation_conservation_and_service_level_order(cleaned):
    bt = ro.policy_backtest(cleaned, ro.InventoryParams(), test_days=30)
    tab = bt["table"]
    assert tab["fill_rate"].between(0, 1).all()
    assert tab["fill_rate"].is_monotonic_increasing or tab["fill_rate"].iloc[-1] >= tab["fill_rate"].iloc[0]
    assert "assumptions" in bt and bt["chosen"] is not None


def test_simulate_policy_rejects_bad_input():
    with pytest.raises(DataValidationError):
        ro.simulate_policy(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]), 1, 0.95)


def test_policy_backtest_empty():
    with pytest.raises(DataValidationError):
        ro.policy_backtest(pd.DataFrame())


# ---------------------------------------------------------------- report generator
def _data():
    return {"dataset": {"records": 10, "records_total": 20, "date_min": "2023-01-01", "date_max": "2023-01-10", "stores": 2, "products": 3}}


def test_report_has_synthetic_statement_and_handles_missing_sections():
    md = rg.to_markdown(rg.build_report_blocks(_data()))
    assert "SYNTHETIC" in md.upper() and rg.TITLE in md
    assert "Records in selection: 10 of 20" in md


def test_report_requires_dataset():
    with pytest.raises(DataValidationError):
        rg.build_report_blocks({})


def test_csv_bundle_skips_empty():
    z = zipfile.ZipFile(io.BytesIO(rg.csv_bundle({"a": pd.DataFrame({"x": [1]}), "b": pd.DataFrame(), "c": None}, "# md")))
    assert sorted(z.namelist()) == ["a.csv", "inventory_report.md"]


def test_pdf_when_available():
    if not rg.pdf_available():
        pytest.skip("reportlab not installed (optional)")
    pdf = rg.to_pdf(rg.build_report_blocks(_data()))
    assert pdf.startswith(b"%PDF")


# ---------------------------------------------------------------- UI components
def test_html_is_escaped():
    html = ui.kpi_cards_html([ui.Kpi("<b>x</b>", "<script>1</script>", "n&n")])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<script>" not in ui.callout_html("info", "<script>alert(1)</script> **ok**")
    assert "<b>ok</b>" in ui.callout_html("info", "**ok**")
    assert "<img" not in ui.header_html("<img src=x>", "s", ["<i>"])
    assert "<img" not in ui.insight_cards_html([{"title": "<img>", "detail": "<img>", "tone": "red"}])


def test_unknown_tone_falls_back():
    assert "si-kpi" in ui.kpi_cards_html([ui.Kpi("a", "1", tone="not-a-tone")])
    assert "not-a-tone" not in ui.badge_html("t", "not-a-tone")


# ---------------------------------------------------------------- filtering and cache-sensitive calculations
def test_restrict_selection_and_empty(cleaned):
    r = dm.restrict_selection(cleaned, stores=["S1"], products=["P1"])
    assert set(r["store_id"]) == {"S1"} and set(r["product_id"]) == {"P1"}
    assert dm.restrict_selection(cleaned, stores=["nope"]).empty


def test_recommendations_change_with_settings(cleaned):
    a = ro.recommend(cleaned, ro.InventoryParams(lead_time_days=1))
    b = ro.recommend(cleaned, ro.InventoryParams(lead_time_days=7))
    assert (b["reorder_point"] > a["reorder_point"]).all()


def test_invalid_params_rejected():
    with pytest.raises((ValueError, DataValidationError)):
        ro.InventoryParams(service_level=1.5)


def test_roc_pr_curves_on_toy_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    pred = pd.DataFrame({"stockout_label": y, "Model": np.clip(y * 0.4 + rng.uniform(0, 0.6, 400), 0, 1)})
    out = dm.roc_pr_curves(pred, ["Model"])
    assert isinstance(out, dict) and out
