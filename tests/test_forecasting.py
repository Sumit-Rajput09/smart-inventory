"""Phase 3 tests: metrics, leakage, chronological protocol, selection rule, training, saving, future forecasts."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import demand_forecasting as df_mod
from src.preprocess import chronological_split
from src.utils import DataFileError, DataValidationError

ROOT = Path(__file__).resolve().parents[1]


def make_cleaned(n_days=330, signal=0.0, seed=0, stores=("S1", "S2"), products=("P1", "P2", "P3")):
    """Cleaned-style data. units_sold = base + signal * (previous day's deviation) + noise, per series."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in stores:
        for p in products:
            base = 100 + 10 * products.index(p)
            prev = 0.0
            for i in range(n_days):
                dev = signal * prev + rng.normal(0, 20)
                prev = dev
                sold = int(max(0, base + dev))
                rows.append({"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i), "store_id": s, "product_id": p,
                             "category": "Toys", "region": "North", "inventory_level": sold + int(rng.integers(5, 200)),
                             "units_sold": sold, "units_ordered": 50, "demand_forecast": sold + rng.normal(0, 3),
                             "price": 20.0 + float(rng.normal(0, 1)), "discount": 10, "weather": "Sunny", "holiday_promo": int(rng.integers(0, 2)),
                             "competitor_price": 21.0, "seasonality": "Winter"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def noise_result():
    return df_mod.run_training(make_cleaned(), horizon=1, test_days=40, val_days=40, n_estimators=20, save=False)


# ------------------------------------------------------------------ metrics
def test_regression_metrics_known_values():
    m = df_mod.regression_metrics([10, 20, 30], [12, 18, 33])
    assert m["mae"] == pytest.approx((2 + 2 + 3) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["wape"] == pytest.approx(7 / 60)
    assert m["bias"] == pytest.approx((2 - 2 + 3) / 3)


def test_metrics_handle_zeros_and_bad_input():
    assert np.isnan(df_mod.regression_metrics([0, 0], [1, 1])["wape"])      # no division by zero crash
    with pytest.raises(DataValidationError):
        df_mod.regression_metrics([], [])
    with pytest.raises(DataValidationError):
        df_mod.regression_metrics([1, 2], [1])


def test_seven_day_totals():
    d = pd.date_range("2023-01-01", periods=14)
    f = pd.DataFrame({"date": d, "store_id": "S", "product_id": "P", "units_sold": 10, "pred": 12})
    m = df_mod.seven_day_total_metrics(f, "pred")
    assert m["mae_7d_total"] == pytest.approx(14)         # 7*12 - 7*10 per block
    assert m["wape_7d_total"] == pytest.approx(0.2)


def test_weekly_metric_only_for_horizon_7_or_more():
    d = pd.date_range("2023-01-01", periods=14)
    f = pd.DataFrame({"date": d, "store_id": "S", "product_id": "P", "units_sold": 10, "pred": 12})
    assert np.isnan(df_mod.evaluate(f, "pred", horizon=1)["mae_7d_total"])
    assert df_mod.evaluate(f, "pred", horizon=7)["mae_7d_total"] == pytest.approx(14)
    assert df_mod.evaluate(f, "pred", horizon=1)["mae"] == pytest.approx(2)


def test_paired_bootstrap_detects_difference():
    a, b = np.full(500, 5.0), np.full(500, 10.0)
    diff, lo, hi = df_mod.paired_bootstrap_diff(a, b)
    assert diff == -5 and hi < 0


# ------------------------------------------------------------------ leakage and protocol
def test_features_never_include_leaky_columns():
    frame, cols = df_mod.prepare_model_frame(make_cleaned(), horizon=1)
    assert not {"units_sold", "demand_forecast", "units_ordered", "inventory_level", "year"} & set(cols)
    assert not frame[[c for c in cols if "_lag_" in c or c.startswith("roll_")]].isna().any().any()


def test_features_do_not_change_when_target_day_changes():
    c = make_cleaned(n_days=60)
    base, cols = df_mod.prepare_model_frame(c, horizon=1)
    c2 = c.copy()
    t = pd.Timestamp("2023-02-15")
    c2.loc[c2.date == t, "units_sold"] = 9999
    alt, _ = df_mod.prepare_model_frame(c2, horizon=1)
    key = ["store_id", "product_id"]
    a = base[base.date == t].sort_values(key)[cols].reset_index(drop=True)
    b = alt[alt.date == t].sort_values(key)[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_train_validation_test_are_chronological(noise_result):
    meta = noise_result["metadata"]["periods"]
    assert meta["train"][1] < meta["validation"][0] <= meta["validation"][1] < meta["test"][0]
    assert noise_result["test_frame"][df_mod.DATE].min() == pd.Timestamp(meta["test"][0])


def test_all_models_scored_on_same_test_rows(noise_result):
    f = noise_result["test_frame"]
    names = [n for n in noise_result["test_table"].model if n in f.columns]
    assert f[names].notna().all().all() and len(names) == len(noise_result["test_table"])


def test_predictions_are_non_negative(noise_result):
    f = noise_result["test_frame"]
    for n in noise_result["val_table"].model:
        assert (f[n] >= 0).all()


def test_series_mean_uses_only_training_data():
    c = make_cleaned(n_days=100)
    train = c[c.date < "2023-03-01"]
    m = df_mod.SeriesMean().fit(train)
    changed = c.copy()
    changed.loc[changed.date >= "2023-03-01", "units_sold"] = 5000
    pred = m.predict(changed)
    assert pred.max() < 500                                   # future values had no influence


def test_leaky_references_are_labelled_invalid(noise_result):
    t = noise_result["test_table"].set_index("model")
    assert not t.loc["Demand Forecast column", "valid"] and not t.loc["LR + same-day inventory", "valid"]
    assert t.loc["Demand Forecast column", "mae"] < t.loc["Series mean", "mae"] / 3      # leakage looks "great"
    assert noise_result["selected"] not in ("Demand Forecast column", "LR + same-day inventory")


# ------------------------------------------------------------------ selection rule
def test_noise_data_selects_simple_model(noise_result):
    assert noise_result["selected"] in ("Series mean", "Moving average 28d", "Moving average 7d")
    assert noise_result["selected"] != "Global mean"


def test_signal_data_selects_ml_and_beats_baseline():
    r = df_mod.run_training(make_cleaned(signal=0.9, seed=1), horizon=1, test_days=40, val_days=40,
                            n_estimators=30, save=False)
    t = r["test_table"].set_index("model")
    assert t.loc["Linear Regression", "mae"] < 0.9 * t.loc["Series mean", "mae"]
    assert r["selected"] in ("Linear Regression", "Random Forest", "Moving average 7d", "Naive (lag 1)")


def test_select_model_prefers_simple_within_tolerance():
    cands = [df_mod.Candidate("A", 0, True, df_mod.SeriesMean), df_mod.Candidate("B", 4, True, df_mod.SeriesMean),
             df_mod.Candidate("G", 0, False, df_mod.GlobalMean)]
    table = pd.DataFrame({"model": ["A", "B", "G"], "mae": [10.0, 9.95, 1.0]})
    assert df_mod.select_model(table, cands, tolerance=0.01) == "A"      # B only 0.5% better; G not eligible
    assert df_mod.select_model(table, cands, tolerance=0.0) == "B"
    with pytest.raises(DataValidationError):
        df_mod.select_model(table, [df_mod.Candidate("G", 0, False, df_mod.GlobalMean)])


def test_horizon_changes_available_features():
    _, cols7 = df_mod.prepare_model_frame(make_cleaned(), horizon=7)
    assert "units_sold_lag_1" not in cols7 and "units_sold_lag_7" in cols7
    cands = df_mod.build_candidates(cols7, 7, ["units_sold_lag_7", "units_sold_lag_14", "roll_mean_7", "roll_mean_28"])
    names = [c.name for c in cands]
    assert "Naive (lag 7)" in names and "Seasonal naive (lag 7)" not in names      # identical models, no duplicate
    names1 = [c.name for c in df_mod.build_candidates(cols7, 1, ["units_sold_lag_1", "units_sold_lag_7"])]
    assert "Naive (lag 1)" in names1 and "Seasonal naive (lag 7)" in names1


def test_bad_horizon_and_short_data_raise():
    with pytest.raises(ValueError):
        df_mod.prepare_model_frame(make_cleaned(), horizon=0)
    with pytest.raises(DataValidationError):
        df_mod.run_training(make_cleaned(n_days=60), horizon=1, test_days=90, val_days=90, save=False)


# ------------------------------------------------------------------ saving / loading / future forecast
def test_save_and_load_roundtrip(tmp_path):
    c = make_cleaned()
    r = df_mod.run_training(c, 1, 40, 40, n_estimators=10, models_dir=tmp_path / "m", reports_dir=tmp_path / "r")
    model, meta = df_mod.load_model(1, tmp_path / "m")
    frame, _ = df_mod.prepare_model_frame(c, 1)
    np.testing.assert_allclose(model.predict(frame), r["model"].predict(frame))
    assert meta["horizon_days"] == 1 and meta["data_is_synthetic"] and meta["model_name"] == r["selected"]
    assert df_mod.available_horizons(tmp_path / "m") == [1]
    for f in ("forecast_summary_h1.txt", "forecast_model_comparison_h1.csv", "forecast_test_predictions_h1.csv"):
        assert (tmp_path / "r" / f).exists()
    assert (tmp_path / "r" / "figures" / "forecast_actual_vs_predicted_h1.png").exists()


def test_load_missing_model_message(tmp_path):
    with pytest.raises(DataFileError, match="python -m src.demand_forecasting"):
        df_mod.load_model(3, tmp_path)


@pytest.mark.parametrize("h", [1, 7])
def test_future_forecast_shape_dates_and_values(h):
    c = make_cleaned()
    model = df_mod.run_training(c, h, 40, 40, n_estimators=10, save=False)["fitted"]["Random Forest"]
    out = df_mod.forecast_future(model, c, h)
    assert len(out) == c.groupby(["store_id", "product_id"]).ngroups
    assert (out["target_date"] == c.date.max() + pd.Timedelta(days=h)).all()
    assert out["forecast_units"].notna().all() and (out["forecast_units"] >= 0).all()


def test_model_for_wrong_horizon_is_rejected(noise_result):
    """A horizon-1 model needs lag_1, which does not exist for a 7-day-ahead forecast: fail loudly, not silently."""
    with pytest.raises(DataValidationError, match="missing feature"):
        df_mod.forecast_future(noise_result["fitted"]["Random Forest"], make_cleaned(), 7)


def test_future_forecast_ignores_nothing_but_history():
    """Forecast for last+1 must not depend on any invented future value: rebuild with a different future price assumption is fixed, history identical -> identical output."""
    c = make_cleaned(n_days=80)
    m = df_mod.SeriesMean().fit(c)
    a = df_mod.forecast_future(m, c, 1)
    b = df_mod.forecast_future(m, c.copy(), 1)
    pd.testing.assert_frame_equal(a, b)


def test_future_forecast_empty_data():
    with pytest.raises(DataValidationError):
        df_mod.build_future_frame(make_cleaned().iloc[0:0], 1)


def test_cli_trains_and_model_loads_in_another_process(tmp_path):
    """The classic pickle trap: running `python -m ...` must not save classes as __main__.X."""
    csv = tmp_path / "clean.csv"
    make_cleaned().to_csv(csv, index=False)
    cmd = [sys.executable, "-m", "src.demand_forecasting", "--input", str(csv), "--horizons", "1",
           "--test-days", "40", "--val-days", "40", "--n-estimators", "10",
           "--models-dir", str(tmp_path / "m"), "--reports-dir", str(tmp_path / "r")]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr + done.stdout
    model, meta = df_mod.load_model(1, tmp_path / "m")        # would raise if saved as __main__.SeriesMean
    assert type(model).__module__ == "src.demand_forecasting"
