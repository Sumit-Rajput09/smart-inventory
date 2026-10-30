"""Phase 5 tests: dashboard helpers and a page-by-page run of the Streamlit app with AppTest."""
import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import app_support as sup
from src import demand_forecasting as df_mod
from src import reorder_optimizer as ro
from src import stockout_prediction as sp
from src.utils import DataValidationError
from tests.test_inventory import make_cleaned

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


# ------------------------------------------------------------------ helpers
def test_get_paths_uses_env_root(tmp_path, monkeypatch):
    monkeypatch.setenv(sup.ENV_ROOT, str(tmp_path))
    p = sup.get_paths()
    assert p.root == tmp_path and p.cleaned_csv.parent.parent == tmp_path / "data" or p.cleaned_csv.is_relative_to(tmp_path)
    assert p.figures_dir == p.reports_dir / "figures"


def test_artifact_status_reports_missing(tmp_path):
    status = sup.artifact_status(sup.get_paths(tmp_path))
    assert status and not any(s.get("exists") or s.get("found") for s in status)


def test_search_dataframe_all_words_must_match():
    d = pd.DataFrame({"a": ["Red Toys", "Blue Toys", "Red Food"], "b": [1, 2, 3]})
    assert len(sup.search_dataframe(d, "red toys")) == 1
    assert len(sup.search_dataframe(d, "TOYS")) == 2
    assert len(sup.search_dataframe(d, "")) == 3
    assert sup.search_dataframe(d, "zzz").empty
    assert len(sup.search_dataframe(d, "red", columns=["b"])) == 0


def test_parse_on_order_upload_valid_and_invalid():
    ok = sup.parse_on_order_upload(b"Store ID,product id,on order\nS1,P1,10\nS2,P2,0\n")
    assert list(ok["on_order"]) == [10, 0]
    for bad in (b"store_id,product_id\nS1,P1\n", b"store_id,product_id,on_order\nS1,P1,-5\n",
                b"store_id,product_id,on_order\nS1,P1,abc\n"):
        with pytest.raises(DataValidationError):
            sup.parse_on_order_upload(bad)


def test_filter_recommendations_and_empty_error():
    recs = pd.DataFrame({"store_id": ["S1", "S2"], "product_id": ["P1", "P1"], "category": ["A", "B"], "region": ["N", "N"]})
    assert len(sup.filter_recommendations(recs)) == 2
    assert len(sup.filter_recommendations(recs, stores=["S1"])) == 1
    with pytest.raises(DataValidationError):
        sup.filter_recommendations(recs, stores=["S9"])


def test_insights_for_and_formatters():
    assert sup.insights_for(["TREND: x", "OUTLIER: y"], "TREND") == ["TREND: x"]
    assert sup.fmt_int(1234567) == "1,234,567"
    assert "%" in sup.fmt_pct(0.5) or "%" in sup.fmt_pct(50)


def test_attach_snapshot_attributes():
    cleaned = make_cleaned(n_days=30)
    recs = ro.recommend(cleaned, ro.InventoryParams())
    out = sup.attach_snapshot_attributes(recs, cleaned)
    assert {"category", "region"} <= set(out.columns) and len(out) == len(recs)


# ------------------------------------------------------------------ app runs
def _run(root, monkeypatch):
    monkeypatch.setenv(sup.ENV_ROOT, str(root))
    return AppTest.from_file(APP, default_timeout=180).run()


def test_app_without_data_shows_instructions(tmp_path, monkeypatch):
    at = _run(tmp_path, monkeypatch)
    assert not at.exception
    assert any("not been created" in e.value for e in at.error)
    assert any("python -m src.preprocess" in c.value for c in at.code)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    (root / "data" / "processed").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "reports" / "figures").mkdir(parents=True)
    cleaned = make_cleaned(n_days=330)
    cleaned.to_csv(root / "data" / "processed" / "retail_inventory_cleaned.csv", index=False)
    df_mod.run_training(cleaned, horizon=1, n_estimators=10, models_dir=root / "models", reports_dir=root / "reports")
    sp.run_classifier_experiment(cleaned, lead_time=2, n_estimators=10, models_dir=root / "models", reports_dir=root / "reports")
    ro.run_reorder_report(cleaned, ro.InventoryParams(), reports_dir=root / "reports")
    return root


@pytest.mark.parametrize("page", sup.PAGES)
def test_every_page_runs_with_data(project, monkeypatch, page):
    at = _run(project, monkeypatch)
    assert not at.exception
    at.sidebar.radio(key="nav_page").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
