"""Demand forecasting: baselines vs Linear Regression vs Random Forest.

Run from the project root (needs `python -m src.preprocess` first):
    python -m src.demand_forecasting                      # horizons 1, 7 and 14 days
    python -m src.demand_forecasting --horizons 1 --tolerance 0

PREDICTION TASK
    Predict units_sold of one store-product on day t using only information up to day t-h
    (h = "horizon", in days). Models never see the target day or anything after it.

PROTOCOL (all chronological, never shuffled)
    train | validation | test   (default 90 days validation, 90 days test)
    1. Fit every candidate on TRAIN, score on VALIDATION. Random Forest settings are tuned here.
    2. Pick the model by validation MAE with a simplicity rule (see select_model).
    3. Refit every candidate on TRAIN+VALIDATION and score ONCE on TEST (same period for all).
    4. Save the selected model (fitted on train+validation) and its metadata.
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda import load_cleaned
from src.preprocess import (
    DATE,
    GROUP_KEYS,
    TARGET,
    assert_no_leakage,
    build_features,
    chronological_split,
    get_feature_columns,
)
from src.utils import (
    CLEANED_CSV,
    MODELS_DIR,
    REPORTS_DIR,
    DataFileError,
    DataValidationError,
    ensure_dirs,
    get_logger,
)

log = get_logger()

CATEGORICAL_FEATURES = ["store_id", "product_id"]
# 'year' cannot be extrapolated to the future (a tree has never seen 2025) and carries no repeatable pattern.
EXCLUDED_MODEL_FEATURES = {"year"}
RF_GRID = [{"min_samples_leaf": 30, "max_depth": 12}, {"min_samples_leaf": 100, "max_depth": 8}]


# ================================================================ metrics
def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, WAPE (sum|error| / sum(actual), safe when actual contains zeros), R2 and bias (mean error)."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise DataValidationError(f"Need two non-empty arrays of equal length, got {len(y)} and {len(p)}.")
    err = p - y
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "wape": float(np.abs(err).sum() / y.sum()) if y.sum() > 0 else float("nan"),
            "r2": float(r2_score(y, p)) if np.var(y) > 0 else float("nan"), "bias": float(err.mean())}


def seven_day_total_metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    """Error of 7-day TOTALS per store-product (complete 7-day blocks only).

    Replenishment cares about demand over a lead time, i.e. a sum of days, not one day.
    """
    f = frame[[DATE, *GROUP_KEYS, TARGET, pred_col]].copy()
    f["block"] = (f[DATE] - f[DATE].min()).dt.days // 7
    g = f.groupby([*GROUP_KEYS, "block"]).agg(actual=(TARGET, "sum"), pred=(pred_col, "sum"), n=(DATE, "nunique"))
    g = g[g["n"] == 7]
    if g.empty:
        return {"mae_7d_total": float("nan"), "wape_7d_total": float("nan")}
    err = (g["pred"] - g["actual"]).abs()
    return {"mae_7d_total": float(err.mean()), "wape_7d_total": float(err.sum() / g["actual"].sum())}


def evaluate(frame: pd.DataFrame, pred_col: str, horizon: int) -> dict:
    """Daily metrics always; 7-day-total metrics only when horizon >= 7.

    For horizon < 7 the daily forecasts inside a week use actual sales from that same week, which are NOT
    known when the week is planned, so summing them would give an unrealistically good weekly forecast.
    """
    weekly = seven_day_total_metrics(frame, pred_col) if horizon >= 7 else {"mae_7d_total": float("nan"), "wape_7d_total": float("nan")}
    return {**regression_metrics(frame[TARGET], frame[pred_col]), **weekly}


def paired_bootstrap_diff(abs_err_a: np.ndarray, abs_err_b: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """Mean difference (A - B) of absolute errors with a 95% bootstrap interval.

    Negative = A is better. Rows are resampled as if independent (approximate: rows of one
    store-product series are repeated over time), so read the interval as a rough guide.
    """
    d = np.asarray(abs_err_a, float) - np.asarray(abs_err_b, float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, len(d), len(d))].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def empirical_interval(residuals, coverage: float = 0.90, min_n: int = 20) -> dict:
    """Prediction interval half-widths from HELD-OUT residuals (split-conformal style, no distribution assumed).

    residuals = actual - predicted on data the model was not fitted on. With n residuals the interval
    forecast + [lo, hi] uses the empirical quantiles of the residuals with the finite-sample level
    ceil((n + 1) * coverage) / n, so it covers a new observation with about `coverage` probability IF the
    future errors look like the held-out ones (exchangeability). It says nothing about structural change.
    Raises DataValidationError when there are too few residuals to estimate quantiles honestly.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1 (for example 0.90).")
    if len(r) < min_n:
        raise DataValidationError(f"Only {len(r)} held-out residuals: need at least {min_n} for a prediction interval.")
    tail = (1 - coverage) / 2
    n = len(r)
    lo_level = min(1.0, max(0.0, np.floor((n + 1) * tail) / n))
    hi_level = min(1.0, np.ceil((n + 1) * (1 - tail)) / n)
    lo, hi = np.quantile(r, lo_level), np.quantile(r, hi_level)
    return {"lower_offset": float(lo), "upper_offset": float(hi), "coverage": float(coverage), "n_residuals": int(n)}


def apply_interval(forecast: float, interval: dict, floor: float = 0.0) -> tuple[float, float]:
    """(lower, upper) = forecast + residual quantiles, never below `floor` (demand cannot be negative)."""
    return max(floor, forecast + interval["lower_offset"]), max(floor, forecast + interval["upper_offset"])


# ================================================================ forecasters (picklable, defined at module level)
class _Forecaster:
    """Common interface: fit(df) -> self, predict(df) -> non-negative numpy array."""

    name = "forecaster"
    family = "baseline"

    def fit(self, df: pd.DataFrame):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _non_negative(a) -> np.ndarray:
        return np.clip(np.asarray(a, dtype=float), 0, None)


class GlobalMean(_Forecaster):
    """The same number for every product: the mean units sold in the training period."""

    name = "Global mean"

    def fit(self, df):
        self.mean_ = float(df[TARGET].mean())
        return self

    def predict(self, df):
        return np.full(len(df), self.mean_)


class SeriesMean(_Forecaster):
    """Each store-product's own mean units sold in the training period."""

    name = "Series mean"

    def fit(self, df):
        self.global_ = float(df[TARGET].mean())
        self.means_ = df.groupby(GROUP_KEYS)[TARGET].mean()
        return self

    def predict(self, df):
        idx = pd.MultiIndex.from_frame(df[GROUP_KEYS].astype(str))
        means = self.means_.copy()
        means.index = pd.MultiIndex.from_frame(means.index.to_frame(index=False).astype(str))
        return self._non_negative(means.reindex(idx).fillna(self.global_).to_numpy())


class ColumnBaseline(_Forecaster):
    """Predict with an existing history column (a lag or a rolling mean), falling back to the series mean."""

    def __init__(self, name: str, column: str):
        self.name, self.column = name, column

    def fit(self, df):
        self._fallback = SeriesMean().fit(df)
        return self

    def predict(self, df):
        fallback = self._fallback.predict(df)
        if self.column not in df.columns:
            raise DataValidationError(f"Column '{self.column}' needed by '{self.name}' is missing.")
        return self._non_negative(df[self.column].astype(float).fillna(pd.Series(fallback, index=df.index)).to_numpy())


class SklearnForecaster(_Forecaster):
    """scikit-learn estimator + preprocessing. Imputer/scaler/encoder are fitted on the data passed to fit()."""

    def __init__(self, name: str, estimator, numeric_cols: list[str], categorical_cols: list[str], scale: bool, family: str):
        self.name, self.family = name, family
        self.numeric_cols, self.categorical_cols = list(numeric_cols), list(categorical_cols)
        steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
        pre = ColumnTransformer(
            [("num", Pipeline(steps), self.numeric_cols),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols)],
            sparse_threshold=0)
        self.pipeline = Pipeline([("prep", pre), ("model", estimator)])

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols

    def fit(self, df):
        self.pipeline.fit(df[self.feature_names], df[TARGET].astype(float))
        return self

    def predict(self, df):
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise DataValidationError(f"Cannot predict: missing feature column(s) {missing}")
        return self._non_negative(self.pipeline.predict(df[self.feature_names]))


@dataclass
class Candidate:
    name: str
    complexity: int          # used by the simplicity rule: lower = simpler
    eligible: bool           # can this be saved as THE forecasting model?
    make: Callable[[], _Forecaster]


def build_candidates(feature_cols: list[str], horizon: int, available_columns, rf_params: dict | None = None,
                     n_estimators: int = 100, seed: int = 42) -> list[Candidate]:
    """All models compared in one experiment (naive/seasonal-naive only if their lag exists for this horizon)."""
    rf_params = rf_params or RF_GRID[0]
    cols = set(available_columns)
    out = [Candidate("Global mean", 0, False, GlobalMean),   # reference: ignores the product, so not usable per item
           Candidate("Series mean", 0, True, SeriesMean)]
    lag = f"{TARGET}_lag_{horizon}"
    if lag in cols:
        out.append(Candidate(f"Naive (lag {horizon})", 1, True, lambda: ColumnBaseline(f"Naive (lag {horizon})", lag)))
    seasonal = sorted(int(c.rsplit("_", 1)[1]) for c in cols if c.startswith(f"{TARGET}_lag_") and int(c.rsplit("_", 1)[1]) % 7 == 0)
    if seasonal and seasonal[0] != horizon:          # at horizon 7 the naive model already IS the weekly-seasonal naive
        k = seasonal[0]
        out.append(Candidate(f"Seasonal naive (lag {k})", 1, True,
                             lambda k=k: ColumnBaseline(f"Seasonal naive (lag {k})", f"{TARGET}_lag_{k}")))
    for w in (7, 28):
        if f"roll_mean_{w}" in cols:
            out.append(Candidate(f"Moving average {w}d", 2, True,
                                 lambda w=w: ColumnBaseline(f"Moving average {w}d", f"roll_mean_{w}")))
    out.append(Candidate("Linear Regression", 3, True, lambda: SklearnForecaster(
        "Linear Regression", LinearRegression(), feature_cols, CATEGORICAL_FEATURES, scale=True, family="linear")))
    out.append(Candidate("Random Forest", 4, True, lambda: SklearnForecaster(
        "Random Forest",
        RandomForestRegressor(n_estimators=n_estimators, max_features=0.33, max_samples=0.5, n_jobs=-1,
                              random_state=seed, **rf_params),
        feature_cols, CATEGORICAL_FEATURES, scale=False, family="forest")))
    # Gradient boosting is part of scikit-learn (no new dependency). Small, strongly regularised trees: the data has a very weak signal.
    out.append(Candidate("Hist Gradient Boosting", 5, True, lambda: SklearnForecaster(
        "Hist Gradient Boosting",
        HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
                                      l2_regularization=1.0, early_stopping=False, random_state=seed),
        feature_cols, CATEGORICAL_FEATURES, scale=False, family="boosting")))
    return out


# ================================================================ selection
def select_model(val_table: pd.DataFrame, candidates: list[Candidate], tolerance: float = 0.01) -> str:
    """Pick the SIMPLEST eligible model whose validation MAE is within `tolerance` (1%) of the best eligible MAE.

    Rationale: validation errors are noisy estimates. A complex model must beat a simple one clearly
    to deserve its extra complexity; otherwise the simpler model is preferred. tolerance = 0 gives
    pure "lowest validation MAE".
    """
    info = {c.name: c for c in candidates if c.eligible}
    t = val_table[val_table["model"].isin(info)].copy()
    if t.empty:
        raise DataValidationError("No eligible candidate models to select from.")
    threshold = t["mae"].min() * (1 + tolerance)
    ok = t[t["mae"] <= threshold].copy()
    ok["complexity"] = ok["model"].map(lambda n: info[n].complexity)
    return ok.sort_values(["complexity", "mae"]).iloc[0]["model"]


# ================================================================ data preparation
def prepare_model_frame(cleaned: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """Features for horizon h; drops warm-up rows whose lag/rolling history does not exist yet."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    feats = build_features(cleaned, horizon=horizon).replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in get_feature_columns(feats) if c not in EXCLUDED_MODEL_FEATURES]
    assert_no_leakage(feature_cols)
    history_cols = [c for c in feature_cols if "_lag_" in c or c.startswith("roll_") or c == "trend_7_vs_28"]
    frame = feats.dropna(subset=history_cols).reset_index(drop=True)
    if frame.empty:
        raise DataValidationError("No rows have enough history for the lag/rolling features - the data is too short.")
    return frame, feature_cols


# ================================================================ training experiment
def run_training(cleaned: pd.DataFrame, horizon: int = 1, test_days: int = 90, val_days: int = 90,
                 tolerance: float = 0.01, n_estimators: int = 100, seed: int = 42,
                 models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR, save: bool = True) -> dict:
    """Run the full experiment for one horizon and (optionally) write model + reports."""
    frame, feature_cols = prepare_model_frame(cleaned, horizon)
    train, val, test = chronological_split(frame, test_days=test_days, val_days=val_days)
    trainval = pd.concat([train, val], ignore_index=True)
    log.info("h=%d | train %d rows (%s..%s), validation %d, test %d", horizon, len(train),
             train[DATE].min().date(), train[DATE].max().date(), len(val), len(test))

    # -- 1. tune Random Forest on validation
    rf_grid_rows = []
    for params in RF_GRID:
        cands = build_candidates(feature_cols, horizon, frame.columns, params, n_estimators, seed)
        rf = next(c for c in cands if c.name == "Random Forest").make().fit(train)
        rf_grid_rows.append({**params, "val_mae": regression_metrics(val[TARGET], rf.predict(val))["mae"]})
    rf_grid = pd.DataFrame(rf_grid_rows)
    best_params = {k: int(v) for k, v in rf_grid.sort_values("val_mae").iloc[0].drop("val_mae").items()}
    candidates = build_candidates(feature_cols, horizon, frame.columns, best_params, n_estimators, seed)

    # -- 2. validation scores (fit on train)
    val_frame = val[[DATE, *GROUP_KEYS, TARGET]].copy()
    val_rows = []
    for c in candidates:
        val_frame[c.name] = c.make().fit(train).predict(val)
        val_rows.append({"model": c.name, "eligible": c.eligible, **evaluate(val_frame, c.name, horizon)})
    val_table = pd.DataFrame(val_rows)
    selected = select_model(val_table, candidates, tolerance)

    # -- 3. test scores (refit on train+validation, identical test period for every model)
    test_frame = test[[DATE, *GROUP_KEYS, TARGET]].copy()
    fitted: dict[str, _Forecaster] = {}
    test_rows = []
    for c in candidates:
        fitted[c.name] = c.make().fit(trainval)
        test_frame[c.name] = fitted[c.name].predict(test)
        test_rows.append({"model": c.name, "valid": True, **evaluate(test_frame, c.name, horizon)})

    # -- 4. reference experiments that are NOT valid forecasting models (shown to explain the exclusions)
    ref_rows = []
    if "demand_forecast" in test.columns:
        test_frame["Demand Forecast column"] = test["demand_forecast"].fillna(trainval[TARGET].mean()).clip(lower=0).to_numpy()
        ref_rows.append({"model": "Demand Forecast column", "valid": False, **evaluate(test_frame, "Demand Forecast column", horizon)})
    if "inventory_level" in test.columns:
        leaky = SklearnForecaster("LR + same-day inventory", LinearRegression(), feature_cols + ["inventory_level"],
                                  CATEGORICAL_FEATURES, scale=True, family="linear").fit(trainval)
        test_frame["LR + same-day inventory"] = leaky.predict(test)
        ref_rows.append({"model": "LR + same-day inventory", "valid": False, **evaluate(test_frame, "LR + same-day inventory", horizon)})
    test_table = pd.DataFrame(test_rows + ref_rows)

    # -- 5. is any ML model really better than the plain series mean? (paired bootstrap on test)
    base = "Series mean"
    base_err = (test_frame[base] - test_frame[TARGET]).abs().to_numpy()
    comparisons = []
    for name in ("Linear Regression", "Random Forest", "Hist Gradient Boosting"):
        diff, lo, hi = paired_bootstrap_diff((test_frame[name] - test_frame[TARGET]).abs().to_numpy(), base_err)
        comparisons.append({"model": name, "vs": base, "mae_diff": diff, "ci_low": lo, "ci_high": hi})
    comparison = pd.DataFrame(comparisons)

    importance = None
    rf_model = fitted.get("Random Forest")
    if rf_model is not None:
        names = [n.split("__", 1)[1] for n in rf_model.pipeline.named_steps["prep"].get_feature_names_out()]
        importance = pd.Series(rf_model.pipeline.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_days": horizon, "target": TARGET,
        "task": f"predict units_sold of a store-product on day t using only data up to day t-{horizon}",
        "model_name": selected, "model_family": fitted[selected].family,
        "feature_columns": feature_cols + CATEGORICAL_FEATURES,
        "selection_rule": f"simplest eligible model with validation MAE within {tolerance:.0%} of the best eligible MAE",
        "rf_best_params": best_params,
        "periods": {"train": [str(train[DATE].min().date()), str(train[DATE].max().date())],
                    "validation": [str(val[DATE].min().date()), str(val[DATE].max().date())],
                    "test": [str(test[DATE].min().date()), str(test[DATE].max().date())]},
        "fitted_on": "train + validation",
        "validation_metrics": _row(val_table, selected), "test_metrics": _row(test_table, selected),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "data_is_synthetic": True,
        "warning": "Synthetic dataset: metrics do not demonstrate real business value.",
    }
    result = {"horizon": horizon, "selected": selected, "val_table": val_table, "test_table": test_table, "rf_grid": rf_grid,
              "comparison": comparison, "importance": importance, "test_frame": test_frame, "metadata": meta, "model": fitted[selected],
              "fitted": fitted, "tolerance": tolerance}
    if save:
        write_outputs(result, models_dir, reports_dir)
    return result


def _row(table: pd.DataFrame, model: str) -> dict:
    r = table[table["model"] == model].iloc[0].to_dict()
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items() if k not in ("model", "eligible", "valid")}


# ================================================================ saving / loading
def model_paths(horizon: int, models_dir: Path = MODELS_DIR) -> tuple[Path, Path]:
    return models_dir / f"demand_model_h{horizon}.joblib", models_dir / f"demand_model_h{horizon}_metadata.json"


def load_model(horizon: int = 1, models_dir: Path = MODELS_DIR):
    """Return (model, metadata). Raises a friendly DataFileError when nothing has been trained yet."""
    mpath, jpath = model_paths(horizon, models_dir)
    if not mpath.exists() or not jpath.exists():
        raise DataFileError(f"No trained model for horizon {horizon} day(s) in {models_dir}.\n"
                            f"Train it first:  python -m src.demand_forecasting --horizons {horizon}")
    return joblib.load(mpath), json.loads(jpath.read_text(encoding="utf-8"))


def available_horizons(models_dir: Path = MODELS_DIR) -> list[int]:
    return sorted(int(p.stem.split("_h")[1]) for p in models_dir.glob("demand_model_h*.joblib"))


def write_outputs(res: dict, models_dir: Path, reports_dir: Path) -> None:
    ensure_dirs()
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    h = res["horizon"]
    mpath, jpath = model_paths(h, models_dir)
    joblib.dump(res["model"], mpath)
    jpath.write_text(json.dumps(res["metadata"], indent=2, default=str), encoding="utf-8")

    comp = res["val_table"].merge(res["test_table"], on="model", how="outer", suffixes=("_val", "_test"))
    comp.to_csv(reports_dir / f"forecast_model_comparison_h{h}.csv", index=False)
    res["test_frame"].to_csv(reports_dir / f"forecast_test_predictions_h{h}.csv", index=False)
    (reports_dir / f"forecast_summary_h{h}.txt").write_text(build_summary(res), encoding="utf-8")
    plot_model_comparison(res).savefig(fig_dir / f"forecast_model_comparison_h{h}.png", dpi=120)
    plot_actual_vs_predicted(res).savefig(fig_dir / f"forecast_actual_vs_predicted_h{h}.png", dpi=120)
    plot_residuals(res).savefig(fig_dir / f"forecast_residuals_h{h}.png", dpi=120)
    if res["importance"] is not None:
        plot_feature_importance(res).savefig(fig_dir / f"forecast_feature_importance_h{h}.png", dpi=120)
    log.info("h=%d | saved %s, %s and reports", h, mpath.name, jpath.name)


# ================================================================ future forecasts (used by the app / reorder module)
def build_future_frame(cleaned: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Prediction rows for day (last date + horizon), one per store-product.

    Features use history up to the last observed date only. Planned price, discount and promotion
    for the future day are ASSUMED equal to the latest observed values (stated in the output).
    """
    if cleaned.empty:
        raise DataValidationError("Cannot forecast from an empty dataset.")
    last = cleaned[DATE].max()
    latest = cleaned.sort_values(DATE).groupby(GROUP_KEYS, sort=False).tail(1)
    steps = []
    for k in range(1, horizon + 1):
        step = latest.copy()
        step[DATE] = last + pd.Timedelta(days=k)
        step[TARGET] = np.nan
        steps.append(step)
    full = pd.concat([cleaned, *steps], ignore_index=True)
    feats = build_features(full, horizon=horizon).replace([np.inf, -np.inf], np.nan)
    return feats[feats[DATE] == last + pd.Timedelta(days=horizon)].reset_index(drop=True)


def forecast_future(model: _Forecaster, cleaned: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast units sold on day (last observed date + horizon) for every store-product."""
    rows = build_future_frame(cleaned, horizon)
    if rows.empty:
        raise DataValidationError("No store-product series available for forecasting.")
    out = rows[GROUP_KEYS + [DATE]].rename(columns={DATE: "target_date"}).copy()
    out["forecast_units"] = model.predict(rows)
    out["horizon_days"] = horizon
    out["assumption"] = "price/discount/promotion of the target day assumed equal to the latest observed values"
    return out


# ================================================================ plots (each returns a Figure)
def _new_fig(nrows: int = 1, ncols: int = 1, figsize=(10, 5)):
    fig = Figure(figsize=figsize, layout="constrained")
    return fig, fig.subplots(nrows, ncols)


def plot_model_comparison(res: dict) -> Figure:
    val = res["val_table"].set_index("model")
    test = res["test_table"][res["test_table"]["valid"]].set_index("model")
    models = list(test.index)
    fig, axes = _new_fig(1, 2, (14, 5.5))
    x = np.arange(len(models))
    for ax, metric, label in ((axes[0], "mae", "MAE (units per store-day)"), (axes[1], "rmse", "RMSE (units per store-day)")):
        b1 = ax.bar(x - 0.2, val.loc[models, metric], 0.4, label="Validation (fit on train)", color="#4C78A8")
        b2 = ax.bar(x + 0.2, test.loc[models, metric], 0.4, label="Test (fit on train+validation)", color="#F58518")
        for bars in (b1, b2):
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_xticks(x, models, rotation=30, ha="right")
        ax.set(title=f"{label.split(' ')[0]} by model, horizon {res['horizon']} day(s)", ylabel=label)
        ax.set_ylim(0, max(val.loc[models, metric].max(), test.loc[models, metric].max()) * 1.35)
        ax.legend(loc="upper left", fontsize=9)
    return fig


def plot_actual_vs_predicted(res: dict) -> Figure:
    f, sel = res["test_frame"], res["selected"]
    fig, axes = _new_fig(1, 2, (15, 5.5))
    daily = f.groupby(DATE)[[TARGET, sel, "Random Forest"]].sum()
    axes[0].plot(daily.index, daily[TARGET], color="black", lw=1.4, label="Actual")
    axes[0].plot(daily.index, daily[sel], color="#4C78A8", lw=1.8, label=f"Selected: {sel}")
    if sel != "Random Forest":
        axes[0].plot(daily.index, daily["Random Forest"], color="#E45756", lw=1.3, ls="--", label="Random Forest")
    axes[0].set(title="Test period: total units sold per day (all store-products)", xlabel="Date", ylabel="Units sold per day")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()
    hb = axes[1].hexbin(f[TARGET], f["Random Forest"], gridsize=40, bins="log", cmap="viridis", mincnt=1)
    lim = max(f[TARGET].max(), f["Random Forest"].max())
    axes[1].plot([0, lim], [0, lim], color="#E45756", ls="--", label="Perfect forecast")
    r2 = r2_score(f[TARGET], f["Random Forest"])
    axes[1].set(title=f"Random Forest: actual vs predicted per store-day (R2 = {r2:.3f})", xlabel="Actual units sold", ylabel="Predicted units sold")
    axes[1].legend(loc="upper left")
    fig.colorbar(hb, ax=axes[1], label="Records (log scale)")
    return fig


def plot_residuals(res: dict) -> Figure:
    f, sel = res["test_frame"], res["selected"]
    fig, axes = _new_fig(1, 2, (13, 5))
    r = f[sel] - f[TARGET]
    axes[0].hist(r, bins=50, color="#4C78A8")
    axes[0].axvline(0, color="black", lw=1)
    axes[0].axvline(r.mean(), color="#E45756", ls="--", label=f"Mean error (bias) {r.mean():+.1f}")
    axes[0].set(title=f"Residuals of selected model ({sel})", xlabel="Predicted - actual (units)", ylabel="Records")
    axes[0].legend()
    hb = axes[1].hexbin(f[sel], r, gridsize=40, bins="log", cmap="viridis", mincnt=1)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(title="Residual vs predicted value", xlabel="Predicted units", ylabel="Predicted - actual")
    fig.colorbar(hb, ax=axes[1], label="Records (log)")
    return fig


def plot_feature_importance(res: dict, top: int = 15) -> Figure:
    imp = res["importance"].head(top)
    fig, ax = _new_fig(1, 1, (9, 6))
    ax.barh(imp.index[::-1], imp.to_numpy()[::-1], color="#54A24B")
    ax.set(title=f"Random Forest feature importance (impurity-based, top {len(imp)})", xlabel="Importance")
    return fig


# ================================================================ written summary
def build_summary(res: dict) -> str:
    h, sel, meta = res["horizon"], res["selected"], res["metadata"]
    fmt = lambda t: t.to_string(index=False, float_format=lambda v: f"{v:,.3f}")  # noqa: E731
    t, c = res["test_table"], res["comparison"]
    base = t[t["model"] == "Series mean"].iloc[0]
    lines = [f"DEMAND FORECASTING SUMMARY - horizon {h} day(s)", "=" * 78,
             f"Task      : {meta['task']}",
             f"Periods   : train {meta['periods']['train'][0]}..{meta['periods']['train'][1]} | validation "
             f"{meta['periods']['validation'][0]}..{meta['periods']['validation'][1]} | test {meta['periods']['test'][0]}..{meta['periods']['test'][1]}",
             f"Selected  : {sel} ({meta['selection_rule']})", "",
             "VALIDATION RESULTS (fit on train)", fmt(res["val_table"]), "",
             "RANDOM FOREST SETTINGS TRIED (validation MAE)", fmt(res["rf_grid"]), "",
             "TEST RESULTS (fit on train+validation; same test period for every model)",
             "Rows with valid=False are NOT usable forecasting models; they only show what leakage would look like.",
             fmt(t), "",
             "ML MODEL vs SERIES-MEAN BASELINE ON TEST (mean difference of absolute error, 95% paired bootstrap; negative = ML better)", fmt(c), ""]
    best_ml = t[t["model"].isin(["Linear Regression", "Random Forest", "Hist Gradient Boosting"])].sort_values("mae").iloc[0]
    row = c[c["model"] == best_ml["model"]].iloc[0]
    gain = 100 * (base["mae"] - best_ml["mae"]) / base["mae"]
    lines.append("INTERPRETATION (generated from the numbers above)")
    if row["ci_low"] <= 0 <= row["ci_high"]:
        verdict = "the interval includes 0: no evidence that it differs from the simple baseline."
    elif row["ci_high"] < 0:
        verdict = "the whole interval is below 0: it is better than the baseline on this test period."
    else:
        verdict = "the whole interval is above 0: it is WORSE than the baseline on this test period."
    lines.append(f"- Best ML model on test is {best_ml['model']} (MAE {best_ml['mae']:.2f}) vs Series mean (MAE {base['mae']:.2f}): "
                 f"{gain:+.2f}% MAE improvement; {verdict}")
    naive = t[t["model"].str.startswith("Naive")]
    if len(naive):
        n = naive.iloc[0]
        lines.append(f"- {n['model']} has MAE {n['mae']:.2f} ({n['mae'] / base['mae']:.2f}x the series mean). Copying an old day is worse because "
                     "each day's sales are independent noise around the average, so the old day adds its own noise.")
    lines.append(f"- R2 of the selected model on test is {meta['test_metrics']['r2']:.3f}: it explains almost none of the day-to-day variation."
                 if meta["test_metrics"]["r2"] is not None and meta["test_metrics"]["r2"] < 0.05 else
                 f"- R2 of the selected model on test is {meta['test_metrics']['r2']:.3f}.")
    ref = t[~t["valid"]]
    for r in ref.itertuples():
        lines.append(f"- NOT VALID: {r.model} reaches MAE {r.mae:.2f} / R2 {r.r2:.2f} only because it uses information unavailable when forecasting (leakage).")
    if h >= 7:
        sm = t[t["model"] == "Series mean"].iloc[0]
        lines.append(f"- 7-day totals (what replenishment needs): the selected model's MAE on weekly totals per store-product is "
                     f"{meta['test_metrics']['mae_7d_total']:.1f} units, a WAPE of {meta['test_metrics']['wape_7d_total']:.1%}, versus a daily WAPE of "
                     f"{meta['test_metrics']['wape']:.1%}; errors partly cancel when days are summed (series-mean weekly MAE {sm['mae_7d_total']:.1f}).")
    else:
        lines.append("- 7-day-total metrics are shown only for horizons >= 7 days: for shorter horizons the daily forecasts inside a week use sales "
                     "from that same week, which would not be known when the week is planned.")
    lines += ["",
              "LIMITATION: the dataset is SYNTHETIC. Results do not show real retail savings and cannot be generalised."]
    return "\n".join(lines) + "\n"


# ================================================================ CLI
def main() -> None:
    p = argparse.ArgumentParser(description="Train and compare demand forecasting models.")
    p.add_argument("--input", default=str(CLEANED_CSV), help="cleaned CSV from src.preprocess")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 7, 14], help="forecast horizons in days")
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--val-days", type=int, default=90)
    p.add_argument("--tolerance", type=float, default=0.01, help="simplicity tolerance on validation MAE (0 = pure best MAE)")
    p.add_argument("--n-estimators", type=int, default=100, help="trees in the Random Forest")
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    p.add_argument("--reports-dir", default=str(REPORTS_DIR))
    a = p.parse_args()
    try:
        cleaned = load_cleaned(a.input)
        for h in a.horizons:
            r = run_training(cleaned, h, a.test_days, a.val_days, a.tolerance, a.n_estimators,
                             models_dir=Path(a.models_dir), reports_dir=Path(a.reports_dir))
            t = r["test_table"].set_index("model")
            log.info("h=%d | SELECTED: %s | test MAE %.2f RMSE %.2f | Series mean MAE %.2f | Random Forest MAE %.2f | Linear MAE %.2f",
                     h, r["selected"], t.loc[r["selected"], "mae"], t.loc[r["selected"], "rmse"], t.loc["Series mean", "mae"],
                     t.loc["Random Forest", "mae"], t.loc["Linear Regression", "mae"])
    except (DataFileError, DataValidationError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    # Import the module under its real name first: classes defined while running as "__main__" would be
    # pickled as __main__.X and could not be loaded later by Streamlit.
    from src.demand_forecasting import main as _main

    _main()
