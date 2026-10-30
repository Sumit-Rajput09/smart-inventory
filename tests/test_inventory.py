"""Phase 4 tests: inventory formulas, reorder quantities, recommendations, simulated stock-out classifier."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import reorder_optimizer as ro
from src import stockout_prediction as sp
from src.utils import DataFileError, DataValidationError

ROOT = Path(__file__).resolve().parents[1]


def make_cleaned(n_days=330, seed=0, stores=("S1", "S2"), products=("P1", "P2", "P3")):
    """Like the real data: sales are a random share of that day's stock (never above it)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in stores:
        for p in products:
            for i in range(n_days):
                inv = int(rng.integers(50, 500))
                rows.append({"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i), "store_id": s, "product_id": p,
                             "category": "Toys", "region": "North", "inventory_level": inv,
                             "units_sold": int(rng.uniform(0, 1) * inv), "units_ordered": 50, "demand_forecast": 100.0,
                             "price": 20.0 + float(rng.normal(0, 1)), "discount": 10, "weather": "Sunny",
                             "holiday_promo": int(rng.integers(0, 2)), "competitor_price": 21.0, "seasonality": "Winter"})
    return pd.DataFrame(rows)


def constant_series(inventory=300, demand_pattern=(100, 120), days=40, key=("S1", "P1")):
    rows = [{"date": pd.Timestamp("2023-03-01") + pd.Timedelta(days=i), "store_id": key[0], "product_id": key[1],
             "inventory_level": inventory, "units_sold": demand_pattern[i % len(demand_pattern)], "price": 10.0} for i in range(days)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ formulas
def test_z_value_and_validation():
    assert sp.z_value(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert sp.z_value(0.5) == pytest.approx(0.0)
    for bad in (0.3, 1.0, 1.2):
        with pytest.raises(ValueError):
            sp.z_value(bad)


def test_reorder_point_hand_calculation():
    ltd, ss, rop = sp.reorder_point(avg_daily_demand=100, demand_std=20, lead_time_days=4, service_level=0.95)
    assert ltd == 400
    assert ss == pytest.approx(1.6449 * 20 * 2, abs=0.01)
    assert rop == pytest.approx(ltd + ss)


def test_lead_time_must_be_positive():
    with pytest.raises(ValueError):
        sp.lead_time_demand(100, 0)


def test_stockout_probability_properties():
    assert sp.stockout_probability(400, 100, 20, 4) == pytest.approx(0.5)                # stock == expected demand
    assert sp.stockout_probability(1000, 100, 20, 4) < 1e-6
    assert sp.stockout_probability(0, 100, 20, 4) > 0.999999
    pos = np.arange(0, 800, 50)
    p = sp.stockout_probability(pos, 100, 20, 4)
    assert np.all(np.diff(p) <= 0) and ((p >= 0) & (p <= 1)).all()                      # more stock -> less risk
    assert sp.stockout_probability(300, 100, 0, 4) == 1.0                                # zero spread, demand 400 > 300
    assert sp.stockout_probability(500, 100, 0, 4) == 0.0


def test_probability_at_reorder_point_equals_one_minus_service_level():
    _, _, rop = sp.reorder_point(100, 20, 4, 0.95)
    assert sp.stockout_probability(rop, 100, 20, 4) == pytest.approx(0.05, abs=1e-6)


def test_classify_risk_thresholds():
    r = sp.classify_risk([0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.99], 0.95)
    assert list(r) == ["Low", "Low", "Medium", "High", "High", "Critical", "Critical"]
    assert list(r.categories) == sp.RISK_LEVELS


# ------------------------------------------------------------------ order quantity
def test_order_quantity_basic_and_not_triggered():
    q = ro.order_quantity(target_inventory=[500, 500], stock_on_hand=[100, 100], on_order=[0, 0], reorder_now=[True, False])
    assert list(q) == [400, 0]


def test_on_order_reduces_quantity_and_never_negative():
    q = ro.order_quantity([500, 500], [100, 100], [150, 900], [True, True])
    assert list(q) == [250, 0]


def test_min_order_quantity_and_pack_size():
    assert list(ro.order_quantity([500], [450], [0], [True], min_order_qty=100)) == [100]          # raised to MOQ
    assert list(ro.order_quantity([500], [100], [0], [True], order_multiple=24)) == [408]          # 400 -> 17 packs of 24
    assert list(ro.order_quantity([500], [450], [0], [False], min_order_qty=100)) == [0]           # no trigger, no order
    assert list(ro.order_quantity([500.2], [100], [0], [True])) == [401]                           # rounds up whole units


# ------------------------------------------------------------------ settings validation
@pytest.mark.parametrize("kwargs", [{"lead_time_days": 0}, {"service_level": 0.2}, {"service_level": 1.0}, {"review_period_days": -1},
                                    {"lookback_days": 1}, {"min_history_days": 100, "lookback_days": 50}, {"safety_stock_units": -5},
                                    {"excess_factor": 0.5}, {"order_multiple": 0}, {"min_order_qty": -1}])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        ro.InventoryParams(**kwargs)


# ------------------------------------------------------------------ demand statistics (no future information)
def test_demand_stats_uses_only_days_before_snapshot():
    df = constant_series()
    snap = df["date"].iloc[30]
    base = ro.demand_stats(df, snap, lookback_days=20, min_history_days=5)
    altered = df.copy()
    altered.loc[altered["date"] >= snap, "units_sold"] = 99999                 # the snapshot day and the future change
    again = ro.demand_stats(altered, snap, lookback_days=20, min_history_days=5)
    pd.testing.assert_frame_equal(base.drop(columns=["price"]), again.drop(columns=["price"]))
    assert base["avg_daily_demand"].iloc[0] == pytest.approx(110)


def test_demand_stats_lookback_window_and_history_flag():
    df = constant_series(days=40)
    snap = df["date"].iloc[10]
    s = ro.demand_stats(df, snap, lookback_days=90, min_history_days=14)
    assert s["history_days"].iloc[0] == 10 and not s["history_ok"].iloc[0]
    s2 = ro.demand_stats(df, df["date"].iloc[39], lookback_days=7, min_history_days=5)
    assert s2["history_days"].iloc[0] == 7


def test_demand_stats_errors():
    df = constant_series()
    with pytest.raises(DataValidationError, match="empty"):
        ro.demand_stats(df.iloc[0:0])
    with pytest.raises(DataValidationError, match="Snapshot date"):
        ro.demand_stats(df, "2030-01-01")
    with pytest.raises(DataValidationError, match="Snapshot date"):
        ro.demand_stats(df, df["date"].min())              # no history before the first day
    with pytest.raises(DataValidationError, match="Column"):
        ro.demand_stats(df.drop(columns=["inventory_level"]))


# ------------------------------------------------------------------ recommendations
def one_series_stats(inventory, d=100.0, sd=20.0, history_days=60):
    return pd.DataFrame({"store_id": ["S1"], "product_id": ["P1"], "snapshot_date": [pd.Timestamp("2023-06-01")],
                         "current_inventory": [inventory], "avg_daily_demand": [d], "demand_std": [sd],
                         "history_days": [history_days], "history_ok": [history_days >= 14]})


def test_recommendation_hand_calculation():
    params = ro.InventoryParams(lead_time_days=4, service_level=0.95, review_period_days=3)
    r = ro.compute_recommendations(one_series_stats(200), params).iloc[0]
    ss = 1.6449 * 20 * 2
    assert r["lead_time_demand"] == 400
    assert r["safety_stock"] == pytest.approx(ss, abs=0.01)
    assert r["reorder_point"] == pytest.approx(400 + ss, abs=0.01)
    assert r["target_inventory"] == pytest.approx(400 + ss + 300, abs=0.01)
    assert r["projected_stock_after_lead_time"] == pytest.approx(200 - 400)
    assert r["days_of_cover"] == pytest.approx(2.0)
    assert bool(r["reorder_now"]) and r["stock_status"] == "Low stock" and r["risk_level"] == "Critical"
    assert r["suggested_order_qty"] == int(np.ceil(400 + ss + 300 - 200))


def test_status_bands_low_adequate_excess():
    params = ro.InventoryParams(lead_time_days=1, service_level=0.95, review_period_days=1)
    rop = 100 + 1.6449 * 20
    target = rop + 100
    low = ro.compute_recommendations(one_series_stats(rop - 1), params).iloc[0]
    ok = ro.compute_recommendations(one_series_stats((rop + target) / 2), params).iloc[0]
    over = ro.compute_recommendations(one_series_stats(target + 50), params).iloc[0]
    assert (low.stock_status, ok.stock_status, over.stock_status) == ("Low stock", "Adequate stock", "Excess stock")
    assert (bool(low.reorder_now), bool(ok.reorder_now), bool(over.reorder_now)) == (True, False, False)
    assert ok.suggested_order_qty == 0 and over.recommendation == "Overstocked - do not reorder"
    assert low.risk_level != "Low" and ok.risk_level == "Low"


def test_on_order_scalar_and_table_reduce_the_order():
    params = ro.InventoryParams(lead_time_days=2)
    stats = one_series_stats(100)
    base = ro.compute_recommendations(stats, params).iloc[0]
    scalar = ro.compute_recommendations(stats, params, on_order=50).iloc[0]
    table = ro.compute_recommendations(stats, params, on_order=pd.DataFrame({"store_id": ["S1"], "product_id": ["P1"], "on_order": [50]})).iloc[0]
    assert base.suggested_order_qty - scalar.suggested_order_qty == 50 == base.suggested_order_qty - table.suggested_order_qty
    assert scalar.inventory_position == 150
    with pytest.raises(ValueError):
        ro.compute_recommendations(stats, params, on_order=-5)
    with pytest.raises(DataValidationError):
        ro.compute_recommendations(stats, params, on_order=pd.DataFrame({"x": [1]}))


def test_safety_stock_overrides():
    stats = one_series_stats(250)
    fixed = ro.compute_recommendations(stats, ro.InventoryParams(lead_time_days=2, safety_stock_units=30)).iloc[0]
    days = ro.compute_recommendations(stats, ro.InventoryParams(lead_time_days=2, safety_stock_days=0.5)).iloc[0]
    assert fixed.safety_stock == 30 and fixed.reorder_point == 230
    assert days.safety_stock == 50 and days.reorder_point == 250
    assert bool(days.reorder_now)                                  # position 250 <= ROP 250


def test_min_order_and_pack_size_in_recommendations():
    params = ro.InventoryParams(lead_time_days=1, min_order_qty=1000, order_multiple=50)
    assert ro.compute_recommendations(one_series_stats(10), params).iloc[0].suggested_order_qty == 1000


def test_insufficient_history_is_not_ordered():
    r = ro.compute_recommendations(one_series_stats(0, history_days=3), ro.InventoryParams()).iloc[0]
    assert r.stock_status == "Insufficient history" and r.suggested_order_qty == 0 and r.risk_level == "Unknown"
    assert not bool(r.reorder_now) and "history" in r.recommendation


def test_zero_demand_item_does_not_crash():
    r = ro.compute_recommendations(one_series_stats(120, d=0.0, sd=0.0), ro.InventoryParams()).iloc[0]
    assert r.suggested_order_qty == 0 and np.isinf(r.days_of_cover)


def test_recommend_end_to_end_sorted_by_urgency_and_summary():
    c = make_cleaned()
    recs = ro.recommend(c, ro.InventoryParams(lookback_days=60))
    assert len(recs) == 6 and recs["reorder_now"].iloc[0] >= recs["reorder_now"].iloc[-1]
    assert recs["stockout_probability"].iloc[0] >= recs.loc[recs.reorder_now, "stockout_probability"].min() - 1e-12
    k = ro.summarize(recs)
    assert k["series"] == 6 and k["units_to_order"] == int(recs.suggested_order_qty.sum()) and sum(k["risk_counts"].values()) == 6


def test_empty_filtered_dataset_gives_clear_error():
    c = make_cleaned()
    with pytest.raises(DataValidationError, match="empty"):
        ro.recommend(c[c.store_id == "nobody"])
    with pytest.raises(DataValidationError, match="No store-product"):
        ro.compute_recommendations(one_series_stats(10).iloc[0:0], ro.InventoryParams())


def test_lead_time_sensitivity_is_monotonic():
    s = ro.lead_time_sensitivity(make_cleaned(), ro.InventoryParams(lookback_days=60), lead_times=(0.5, 1, 2, 4))
    assert s["pct_low_stock"].is_monotonic_increasing and s["mean_reorder_point"].is_monotonic_increasing


def test_service_level_backtest_returns_sensible_values():
    c = make_cleaned(n_days=330)
    bt = ro.service_level_backtest(c, ro.InventoryParams(lead_time_days=1, service_level=0.95, lookback_days=60), test_days=60)
    assert 0.8 <= bt["achieved_service_level"] <= 1.0 and bt["windows"] > 100
    lo = ro.service_level_backtest(c, ro.InventoryParams(lead_time_days=1, service_level=0.6, lookback_days=60), test_days=60)
    assert lo["achieved_service_level"] < bt["achieved_service_level"]
    with pytest.raises(DataValidationError):
        ro.service_level_backtest(c[c.date < "2023-01-10"], ro.InventoryParams(), test_days=90)


def test_figures_build():
    c = make_cleaned()
    recs = ro.recommend(c, ro.InventoryParams(lookback_days=60))
    sens = ro.lead_time_sensitivity(c, ro.InventoryParams(lookback_days=60))
    for fig in (ro.plot_risk_overview(recs), ro.plot_position_vs_rop(recs), ro.plot_lead_time_sensitivity(sens)):
        assert len(fig.axes) == 2


def test_report_files_written(tmp_path):
    out = ro.run_reorder_report(make_cleaned(), ro.InventoryParams(lookback_days=60), reports_dir=tmp_path)
    for f in ("reorder_recommendations.csv", "reorder_summary.txt", "lead_time_sensitivity.csv"):
        assert (tmp_path / f).exists()
    assert "ASSUMPTIONS" in (tmp_path / "reorder_summary.txt").read_text(encoding="utf-8")
    assert len(out["recommendations"]) == 6


# ------------------------------------------------------------------ simulated labels and classifier
def test_label_hand_example_and_window_ends():
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5), "store_id": "S", "product_id": "P",
                       "inventory_level": [100, 100, 100, 100, 100], "units_sold": [60, 50, 10, 10, 10]})
    out = sp.add_stockout_label(df, lead_time=2)
    assert list(out["demand_next_L_days"].iloc[:4]) == [110, 60, 20, 20] and np.isnan(out["demand_next_L_days"].iloc[4])
    assert list(out[sp.LABEL_COL].iloc[:4]) == [1.0, 0.0, 0.0, 0.0] and np.isnan(out[sp.LABEL_COL].iloc[4])


def test_label_missing_day_gives_nan_not_shifted_value():
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"]), "store_id": "S", "product_id": "P",
                       "inventory_level": 100, "units_sold": [10, 20, 30]})
    out = sp.add_stockout_label(df, 2)
    assert np.isnan(out[sp.LABEL_COL].iloc[0])                    # Jan 2 is missing -> window incomplete
    assert out[sp.LABEL_COL].iloc[1] == 0.0


def test_label_rejects_bad_lead_time():
    with pytest.raises(ValueError):
        sp.add_stockout_label(make_cleaned(n_days=10), 0)
    with pytest.raises(ValueError):
        sp.add_stockout_label(make_cleaned(n_days=10), 1.5)


def test_one_day_label_is_degenerate_and_rejected():
    with pytest.raises(DataValidationError, match="only one class"):
        sp.build_classifier_frame(make_cleaned(), lead_time=1)


def test_purge_removes_rows_whose_window_reaches_next_period():
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10)})
    kept = sp.purge_before(df, pd.Timestamp("2023-01-08"), lead_time=3)       # window = date..date+2 must end before Jan 8
    assert kept["date"].max() == pd.Timestamp("2023-01-05")


def test_classification_metrics_known_confusion_matrix():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1]
    m = sp.classification_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(5 / 6) and m["f1"] == pytest.approx(0.75)
    assert m["recall_no_stockout"] == pytest.approx(5 / 6) and m["precision_no_stockout"] == pytest.approx(5 / 6)
    assert 0.9 < m["roc_auc"] <= 1


def test_metrics_with_no_predicted_positives_do_not_divide_by_zero():
    m = sp.classification_metrics([1, 0, 1, 0], [0.1, 0.1, 0.1, 0.1], 0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_threshold_for_recall_reaches_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    thr = sp.threshold_for_recall(y, p, 0.9)
    assert sp.classification_metrics(y, p, thr)["recall"] >= 0.9
    with pytest.raises(ValueError):
        sp.threshold_for_recall(y, p, 0)


@pytest.fixture(scope="module")
def clf_result():
    return sp.run_classifier_experiment(make_cleaned(n_days=330), lead_time=2, test_days=40, val_days=40, n_estimators=10, save=False)


def test_classifier_experiment_structure_and_chronology(clf_result):
    m = clf_result["metadata"]
    assert m["SIMULATED_LABEL"] is True and "NOT an observed" in m["label_definition"]
    per = m["periods"]
    assert per["train"][1] < per["validation"][0] and per["validation"][1] < per["test"][0]
    assert 0.1 < clf_result["prevalence"] < 0.9
    assert {"Rule-based reorder-point trigger", "Logistic Regression", "Random Forest", "Flag everything (baseline)"} <= set(clf_result["test_table"].model)


def test_training_labels_never_use_validation_sales(clf_result):
    """After purging, the last training row's label window ends before the validation period starts."""
    per = clf_result["metadata"]["periods"]
    last_train = pd.Timestamp(per["train"][1])
    assert last_train + pd.Timedelta(days=clf_result["lead_time"] - 1) < pd.Timestamp(per["validation"][0])


def test_classifier_features_contain_no_target_or_label(clf_result):
    feats = set(clf_result["metadata"]["feature_columns"])
    assert not {"units_sold", "demand_forecast", "units_ordered", sp.LABEL_COL, "demand_next_L_days", "rule_probability"} & feats
    assert "inventory_level" in feats


def test_classifier_recall_target_met_on_validation(clf_result):
    v = clf_result["val_table"].set_index("model")
    for n in ("Logistic Regression", "Random Forest", "Rule-based probability"):
        assert v.loc[n, "recall"] >= 0.9 - 1e-9


def test_classifier_summary_is_labelled_simulated(clf_result):
    text = sp.build_classifier_summary(clf_result)
    assert "SIMULATED" in text and "NOT" in text and "rule-based risk assessment is a formula" in text
    assert len(sp.plot_classifier_evaluation(clf_result).axes) >= 3


def test_score_snapshot_marks_output_simulated(clf_result):
    out = sp.score_snapshot(clf_result["model"], make_cleaned(n_days=330))
    assert len(out) == 6 and out["simulated_stockout_probability"].between(0, 1).all()
    assert out["note"].str.contains("SIMULATED").all()
    with pytest.raises(DataValidationError):
        sp.score_snapshot(clf_result["model"], make_cleaned(n_days=330), "2030-01-01")


def test_load_missing_classifier_message(tmp_path):
    with pytest.raises(DataFileError, match="python -m src.stockout_prediction"):
        sp.load_classifier(5, tmp_path)


def test_cli_classifier_loads_in_another_process(tmp_path):
    csv = tmp_path / "clean.csv"
    make_cleaned(n_days=330).to_csv(csv, index=False)
    cmd = [sys.executable, "-m", "src.stockout_prediction", "--input", str(csv), "--lead-time", "2", "--test-days", "40",
           "--val-days", "40", "--n-estimators", "10", "--models-dir", str(tmp_path / "m"), "--reports-dir", str(tmp_path / "r")]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr + done.stdout
    model, meta = sp.load_classifier(2, tmp_path / "m")
    assert type(model).__module__ == "src.stockout_prediction" and meta["SIMULATED_LABEL"]
    assert (tmp_path / "r" / "stockout_classifier_simulated_L2_summary.txt").exists()


def test_cli_reorder_runs(tmp_path):
    csv = tmp_path / "clean.csv"
    make_cleaned(n_days=200).to_csv(csv, index=False)
    done = subprocess.run([sys.executable, "-m", "src.reorder_optimizer", "--input", str(csv), "--lead-time", "2", "--lookback", "60",
                           "--reports-dir", str(tmp_path / "r")], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr + done.stdout
    assert (tmp_path / "r" / "reorder_recommendations.csv").exists()
    bad = subprocess.run([sys.executable, "-m", "src.reorder_optimizer", "--input", str(csv), "--lead-time", "0",
                          "--reports-dir", str(tmp_path / "r")], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert bad.returncode == 1 and "lead_time_days" in bad.stdout
