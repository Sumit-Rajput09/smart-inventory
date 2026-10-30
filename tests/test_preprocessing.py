"""Phase 1 tests: loading, cleaning, lag features, leakage guard, chronological split."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import check_required_columns, load_dataset, map_columns, read_csv_safely
from src.preprocess import (
    add_lag_features,
    assert_no_leakage,
    build_features,
    chronological_split,
    clean_dataset,
    get_feature_columns,
)
from src.utils import DataFileError, DataValidationError, SchemaError

COLS = ["date", "store_id", "product_id", "category", "region", "inventory_level", "units_sold",
        "units_ordered", "demand_forecast", "price", "discount", "weather", "holiday_promo",
        "competitor_price", "seasonality"]


def make_df(n_days=40, stores=("S1",), products=("P1", "P2")):
    rows = []
    for s in stores:
        for p in products:
            for i in range(n_days):
                rows.append([pd.Timestamp("2023-01-01") + pd.Timedelta(days=i), s, p, "Toys", "North",
                             200, 10 + i, 50, 12.0 + i, 20.0, 10, "Sunny", 0, 21.0, "Winter"])
    df = pd.DataFrame(rows, columns=COLS)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ------------------------------------------------------------ loading
def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(DataFileError, match="not found"):
        read_csv_safely(tmp_path / "nope.csv")


def test_empty_file(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("")
    with pytest.raises(DataFileError, match="empty"):
        read_csv_safely(f)


def test_header_only_file(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text("Date,Store ID\n")
    with pytest.raises(DataFileError, match="no data rows"):
        read_csv_safely(f)


def test_missing_required_column(tmp_path):
    f = tmp_path / "x.csv"
    make_df().drop(columns=["units_sold"]).to_csv(f, index=False)
    with pytest.raises(SchemaError, match="units_sold"):
        load_dataset(f)


def test_column_mapping_kaggle_names():
    raw = pd.DataFrame(columns=["Date", "Store ID", "Product ID", "Units Sold", "Inventory Level",
                                "Holiday/Promotion", "Competitor Pricing", "Mystery Col"])
    df, mapping, unknown = map_columns(raw)
    assert {"date", "store_id", "product_id", "units_sold", "inventory_level", "holiday_promo",
            "competitor_price", "mystery_col"} == set(df.columns)
    assert unknown == ["Mystery Col"]  # kept, not discarded


def test_conflicting_column_names_raise():
    raw = pd.DataFrame(columns=["Units Sold", "units_sold"])
    with pytest.raises(SchemaError):
        map_columns(raw)


def test_latin1_file_is_readable(tmp_path):
    f = tmp_path / "l.csv"
    f.write_bytes("Date,Store ID,Note\n2023-01-01,S1,caf\xe9\n".encode("latin-1"))
    assert len(read_csv_safely(f)) == 1


# ------------------------------------------------------------ cleaning
def test_exact_duplicates_removed():
    df = make_df()
    dup = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    cleaned, _ = clean_dataset(dup)
    assert len(cleaned) == len(df)


def test_invalid_dates_dropped_and_reported():
    df = make_df()
    df.loc[0, "date"] = "not a date"
    df.loc[1, "date"] = None
    cleaned, qlog = clean_dataset(df)
    assert len(cleaned) == len(df) - 2
    q = qlog.to_frame()
    assert q[(q.check == "unparseable / missing dates")].affected_rows.iloc[0] == 2


def test_missing_target_dropped_never_imputed():
    df = make_df()
    df.loc[5, "units_sold"] = np.nan
    cleaned, _ = clean_dataset(df)
    assert len(cleaned) == len(df) - 1
    assert cleaned["units_sold"].notna().all()


def test_negative_and_text_numbers_become_missing_then_causally_filled():
    df = make_df()
    df["price"] = df["price"].astype(object)
    df.loc[3, "price"] = "abc"          # non numeric
    df.loc[4, "price"] = -5             # impossible
    df.loc[2, "discount"] = 150         # impossible
    cleaned, _ = clean_dataset(df)
    p = cleaned[cleaned.product_id == "P1"].sort_values("date").reset_index(drop=True)
    assert p.loc[3, "price"] == 20.0 and p.loc[4, "price"] == 20.0   # filled from the PAST value
    assert (cleaned["price"] > 0).all()
    assert cleaned["discount"].between(0, 100).all()


def test_no_future_information_in_imputation():
    df = make_df(n_days=10, products=("P1",))
    df.loc[0, "price"] = np.nan                  # first row: nothing in the past
    df.loc[1:, "price"] = 99.0                   # the future has values
    cleaned, _ = clean_dataset(df)
    first = cleaned.sort_values("date").iloc[0]
    assert pd.isna(first["price"])               # must NOT be back-filled from the future


def test_negative_units_sold_is_dropped():
    df = make_df()
    df.loc[0, "units_sold"] = -3
    cleaned, _ = clean_dataset(df)
    assert (cleaned["units_sold"] >= 0).all() and len(cleaned) == len(df) - 1


def test_outliers_flagged_not_removed():
    df = make_df(n_days=60, products=("P1",))
    df.loc[30, "units_sold"] = 5000
    cleaned, _ = clean_dataset(df)
    assert len(cleaned) == len(df)
    assert cleaned["outlier_units_sold"].sum() >= 1


def test_everything_invalid_raises():
    df = make_df(n_days=3)
    df["date"] = "garbage"
    with pytest.raises(DataValidationError):
        clean_dataset(df)


# ------------------------------------------------------------ features
def _clean(df):
    return clean_dataset(df)[0]


def test_lag_features_are_correct_and_per_group():
    c = _clean(make_df())
    f = add_lag_features(c)
    p1 = f[f.product_id == "P1"].reset_index(drop=True)
    assert pd.isna(p1.loc[0, "units_sold_lag_1"])
    assert p1.loc[5, "units_sold_lag_1"] == p1.loc[4, "units_sold"]
    assert p1.loc[20, "units_sold_lag_7"] == p1.loc[13, "units_sold"]
    # first row of the second series must not borrow the last value of the first series
    p2 = f[f.product_id == "P2"].reset_index(drop=True)
    assert pd.isna(p2.loc[0, "units_sold_lag_1"])


def test_rolling_mean_excludes_current_day():
    c = _clean(make_df(n_days=30, products=("P1",)))
    f = add_lag_features(c).reset_index(drop=True)
    expected = c.sort_values("date").units_sold.iloc[10:17].mean()   # days t-7..t-1 for t = 17
    assert f.loc[17, "roll_mean_7"] == pytest.approx(expected)


def test_changing_todays_target_does_not_change_todays_features():
    c = _clean(make_df(n_days=30, products=("P1",)))
    base = add_lag_features(c).reset_index(drop=True)
    c2 = c.copy()
    idx = c2.index[c2["date"] == c2["date"].iloc[20]][0]
    c2.loc[idx, "units_sold"] = 99999
    alt = add_lag_features(c2).reset_index(drop=True)
    feats = [x for x in base.columns if "_lag_" in x or x.startswith("roll_")]
    pd.testing.assert_series_equal(base.loc[20, feats], alt.loc[20, feats])
    assert not base.loc[21, feats].equals(alt.loc[21, feats])   # tomorrow SHOULD see it


def test_calendar_gap_does_not_shift_lag():
    c = _clean(make_df(n_days=20, products=("P1",)))
    c = c[c["date"] != pd.Timestamp("2023-01-10")]               # remove one day
    f = add_lag_features(c, lags=(1,)).set_index("date")
    assert pd.isna(f.loc["2023-01-11", "units_sold_lag_1"])      # yesterday is missing -> NaN, not Jan 9
    assert f.loc["2023-01-12", "units_sold_lag_1"] == f.loc["2023-01-11", "units_sold"]


def test_horizon_removes_short_lags_and_delays_rolling():
    c = _clean(make_df(n_days=40, products=("P1",)))
    f = add_lag_features(c, horizon=7).reset_index(drop=True)
    assert "units_sold_lag_1" not in f.columns and "units_sold_lag_7" in f.columns
    # for t = day 24 with horizon 7 the window must cover days t-13 .. t-7 = 11 .. 17
    expected = c.sort_values("date").units_sold.iloc[11:18].mean()
    assert f.loc[24, "roll_mean_7"] == pytest.approx(expected)


def test_leakage_guard():
    with pytest.raises(DataValidationError, match="leakage"):
        assert_no_leakage(["price", "demand_forecast"])
    with pytest.raises(DataValidationError):
        assert_no_leakage(["units_sold"])
    feats = build_features(_clean(make_df()))
    cols = get_feature_columns(feats)
    assert not {"units_sold", "demand_forecast", "units_ordered"} & set(cols)


# ------------------------------------------------------------ split
def test_chronological_split_has_no_overlap_and_is_ordered():
    c = _clean(make_df(n_days=300))
    tr, va, te = chronological_split(c, test_days=30, val_days=30)
    assert tr.date.max() < va.date.min() and va.date.max() < te.date.min()
    assert len(tr) + len(va) + len(te) == len(c)
    assert te.date.nunique() == 30 and va.date.nunique() == 30


def test_split_too_short_raises():
    c = _clean(make_df(n_days=20))
    with pytest.raises(DataValidationError, match="too short"):
        chronological_split(c, test_days=30, val_days=30)


def test_empty_dataframe_split_raises():
    with pytest.raises(Exception):
        chronological_split(pd.DataFrame({"date": pd.to_datetime([])}))
