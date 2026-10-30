"""Phase 2 tests: filters, aggregation, statistics, hypothesis tests, chart smoke tests."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src import eda
from src.utils import DataFileError, DataValidationError


def make_df(n_days=120, seed=0, promo_effect=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in ("S1", "S2"):
        for p in ("P1", "P2", "P3"):
            for i in range(n_days):
                promo = int(rng.integers(0, 2))
                inv = int(rng.integers(100, 500))
                sold = int(min(inv, max(0, rng.normal(100, 30) + promo_effect * promo)))
                rows.append({"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i), "store_id": s, "product_id": p,
                             "category": "Toys" if p == "P1" else "Clothing", "region": "North" if s == "S1" else "South",
                             "inventory_level": inv, "units_sold": sold, "units_ordered": int(rng.integers(20, 200)),
                             "demand_forecast": sold + rng.normal(0, 5), "price": float(rng.uniform(10, 100)),
                             "discount": int(rng.choice([0, 5, 10, 15, 20])), "holiday_promo": promo,
                             "competitor_price": float(rng.uniform(10, 100))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ loading / filters
def test_load_missing_file_message(tmp_path):
    with pytest.raises(DataFileError, match="python -m src.preprocess"):
        eda.load_cleaned(tmp_path / "missing.csv")


def test_filters_select_correct_rows():
    df = make_df()
    out = eda.apply_filters(df, stores=["S1"], products=["P1", "P2"], date_range=("2023-01-01", "2023-01-10"))
    assert set(out.store_id) == {"S1"} and set(out.product_id) == {"P1", "P2"}
    assert out.date.max() <= pd.Timestamp("2023-01-10")


def test_empty_filter_result_raises_friendly_error():
    df = make_df()
    with pytest.raises(DataValidationError, match="No records match"):
        eda.apply_filters(df, stores=["S9"])
    with pytest.raises(DataValidationError):
        eda.apply_filters(df, stores=[])          # nothing selected
    with pytest.raises(DataValidationError):
        eda.apply_filters(df, date_range=("2030-01-01", "2030-02-01"))


def test_missing_column_error():
    with pytest.raises(DataValidationError, match="inventory_level"):
        eda.plot_inventory_distribution(make_df().drop(columns=["inventory_level"]))


# ------------------------------------------------------------------ aggregation
def test_daily_total_matches_manual_sum():
    df = make_df(n_days=10)
    daily = eda.aggregate_sales(df, "D")
    first = df[df.date == df.date.min()].units_sold.sum()
    assert daily.iloc[0] == first and len(daily) == 10


def test_weekly_and_monthly_drop_incomplete_periods():
    df = make_df(n_days=120)                     # 2023-01-01 (Sunday) .. 2023-04-30
    w = eda.aggregate_sales(df, "W")
    m = eda.aggregate_sales(df, "M")
    assert (w.index.dayofweek == 6).all()        # weeks end on Sunday
    assert list(m.index.month) == [1, 2, 3, 4]    # April complete (30 days) is kept
    df2 = df[df.date <= "2023-04-15"]
    assert 4 not in eda.aggregate_sales(df2, "M").index.month   # half April dropped
    full_week_total = df[(df.date >= "2023-01-02") & (df.date <= "2023-01-08")].units_sold.sum()
    assert w.loc["2023-01-08"] == full_week_total


def test_bad_frequency_and_empty():
    with pytest.raises(ValueError):
        eda.aggregate_sales(make_df(), "Y")
    with pytest.raises(DataValidationError):
        eda.aggregate_sales(make_df().iloc[0:0], "D")


def test_top_and_slow_ranking_order():
    df = make_df()
    df.loc[df.product_id == "P1", "units_sold"] = df.loc[df.product_id == "P1", "inventory_level"] // 2 + 100
    top, slow = eda.top_and_slow(df, 1)
    assert top.product_id.iloc[0] == "P1"
    assert slow.mean_units.iloc[0] <= top.mean_units.iloc[0]


# ------------------------------------------------------------------ statistics
def test_descriptive_statistics_match_numpy():
    df = make_df()
    t = eda.descriptive_statistics(df, ["units_sold"])
    x = df.units_sold.to_numpy(float)
    assert t.loc["units_sold", "mean"] == pytest.approx(x.mean())
    assert t.loc["units_sold", "std"] == pytest.approx(x.std(ddof=1))
    assert t.loc["units_sold", "variance"] == pytest.approx(x.var(ddof=1))
    assert t.loc["units_sold", "iqr"] == pytest.approx(np.percentile(x, 75) - np.percentile(x, 25))
    assert t.loc["units_sold", "range"] == x.max() - x.min()


def test_two_group_test_detects_real_effect_and_no_effect():
    strong = eda.compare_two_groups(make_df(n_days=300, promo_effect=40))
    assert strong["welch_p"] < 0.001 and strong["diff_b_minus_a"] > 20 and strong["ci_low"] > 0
    none = eda.compare_two_groups(make_df(n_days=300, promo_effect=0.0))
    assert none["welch_p"] > 0.01 and none["ci_low"] < 0 < none["ci_high"]


def test_two_group_test_needs_two_groups():
    df = make_df()
    with pytest.raises(DataValidationError, match="exactly 2"):
        eda.compare_two_groups(df[df.holiday_promo == 1])


def test_many_group_test():
    df = make_df(n_days=200)
    r = eda.compare_many_groups(df, "units_sold", "discount")
    assert r["n_groups"] == 5 and 0 <= r["eta_squared"] <= 1
    with pytest.raises(DataValidationError):
        eda.compare_many_groups(df, "units_sold", "store_id")     # only 2 groups


def test_correlation_tests_find_forecast_leak():
    ct = eda.correlation_tests(make_df())
    assert ct.iloc[0].variable == "demand_forecast" and ct.iloc[0].pearson_r > 0.9


def test_trend_test_detects_slope():
    df = make_df(n_days=200)
    df["units_sold"] = df["units_sold"] + (df.date - df.date.min()).dt.days * 2
    r = eda.trend_test(df)
    assert r["slope_per_day"] > 10 and r["p_value"] < 0.001


def test_outlier_summary_counts_extreme_value():
    df = make_df()
    df.loc[0, "units_sold"] = 99999
    t = eda.outlier_summary(df, ["units_sold"])
    assert t.n_iqr_outliers.iloc[0] >= 1


def test_missing_summary_counts(tmp_path):
    df = make_df()
    df.loc[:4, "price"] = np.nan
    t = eda.missing_summary(df, tmp_path / "none.csv").set_index("column")
    assert t.loc["price", "missing_after_cleaning"] == 5


# ------------------------------------------------------------------ figures + insights
@pytest.mark.parametrize("name", list(eda.FIGURES))
def test_every_figure_builds(name):
    fig = eda.FIGURES[name](make_df(n_days=200))
    assert isinstance(fig, Figure) and len(fig.axes) >= 1


def test_insights_are_generated_from_data():
    strong = " ".join(eda.generate_insights(make_df(n_days=300, promo_effect=40)))
    flat = " ".join(eda.generate_insights(make_df(n_days=300, promo_effect=0)))
    assert "There is evidence of an effect" in strong
    assert "no evidence that promotions/holidays change sales" in flat
    assert "synthetic" in flat


def test_run_eda_writes_files(tmp_path):
    csv = tmp_path / "clean.csv"
    make_df(n_days=200).to_csv(csv, index=False)
    res = eda.run_eda(csv, tmp_path / "out")
    assert len(res["figures"]) == len(eda.FIGURES) and all(p.exists() for p in res["figures"])
    assert res["report"].read_text(encoding="utf-8").startswith("EDA REPORT")
