"""Stock-out risk: (A) rule-based assessment and (B) a SIMULATED-label classifier experiment.

Run the classifier experiment from the project root:
    python -m src.stockout_prediction                 # lead time 2 days
    python -m src.stockout_prediction --lead-time 3 --target-recall 0.9

DECISION MOMENT used everywhere: the start of day t. We know the stock on hand (inventory_level on day t)
and the sales history up to day t-1. An order placed now arrives after `lead time` L days, so the stock must
cover the demand of days t .. t+L-1.

(A) RULE-BASED  - pure formulas, no training, no labels:
        lead-time demand  LTD = average daily demand x L
        safety stock      SS  = z x daily std x sqrt(L)           z from the service level (95% -> 1.645)
        reorder point     ROP = LTD + SS
        stock-out probability = P(demand during L days > inventory position), demand ~ Normal(d*L, std*sqrt(L))
    This probability is a model-based ESTIMATE, not an observed outcome.

(B) CLASSIFIER  - the dataset has NO real stock-out records. To demonstrate a classifier we build a
    SIMULATED label: "the recorded sales of days t..t+L-1 add up to MORE than the stock on hand on day t,
    assuming nothing is delivered in between". It uses real recorded sales but describes a hypothetical
    situation, so every output is labelled SIMULATED and must not be presented as real stock-out history.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.demand_forecasting import prepare_model_frame
from src.eda import load_cleaned
from src.preprocess import DATE, GROUP_KEYS, TARGET, chronological_split
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

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
HIGH_RISK_PROB = 0.20       # documented thresholds of the rule-based risk levels
CRITICAL_RISK_PROB = 0.50
LABEL_COL = "stockout_label"


# ================================================================ (A) rule-based formulas
def z_value(service_level: float) -> float:
    """Normal quantile for a cycle service level, e.g. 0.95 -> 1.645."""
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be at least 0.5 and below 1 (for example 0.95).")
    return float(stats.norm.ppf(service_level))


def lead_time_demand(avg_daily_demand, lead_time_days):
    """Expected demand while waiting for an order: average daily demand x lead time."""
    if np.any(np.asarray(lead_time_days) <= 0):
        raise ValueError("lead_time_days must be positive.")
    return avg_daily_demand * lead_time_days


def safety_stock(demand_std, lead_time_days, service_level: float = 0.95):
    """Buffer against demand variation: z x daily std x sqrt(lead time).

    Assumes independent days and a fixed lead time (supplier delay is not modelled).
    """
    return z_value(service_level) * demand_std * np.sqrt(lead_time_days)


def reorder_point(avg_daily_demand, demand_std, lead_time_days, service_level: float = 0.95):
    """Return (lead_time_demand, safety_stock, reorder_point). ROP = LTD + SS."""
    ltd = lead_time_demand(avg_daily_demand, lead_time_days)
    ss = safety_stock(demand_std, lead_time_days, service_level)
    return ltd, ss, ltd + ss


def stockout_probability(inventory_position, avg_daily_demand, demand_std, lead_time_days):
    """Estimated probability that demand during the lead time exceeds the stock available.

    Demand over L days is taken as Normal(mean = d*L, sd = std*sqrt(L)). If the spread is zero the
    answer is 1 when expected demand exceeds the stock, else 0. Real daily sales here are right-skewed
    and bounded, so treat the number as a screening score, not an exact probability.
    """
    pos = np.asarray(inventory_position, dtype=float)
    mean = np.asarray(avg_daily_demand, dtype=float) * lead_time_days
    sd = np.asarray(demand_std, dtype=float) * np.sqrt(lead_time_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (pos - mean) / sd
    prob = np.where(sd > 0, 1.0 - stats.norm.cdf(z), (mean > pos).astype(float))
    return np.clip(prob, 0.0, 1.0)


def classify_risk(prob, service_level: float = 0.95) -> pd.Categorical:
    """Documented rule: Low <= (1 - service level) < Medium < 20% <= High < 50% <= Critical."""
    p = np.asarray(prob, dtype=float)
    level = np.select([p <= (1 - service_level) + 1e-12, p < HIGH_RISK_PROB, p < CRITICAL_RISK_PROB],
                      ["Low", "Medium", "High"], default="Critical")
    return pd.Categorical(level, categories=RISK_LEVELS, ordered=True)


RISK_RULES_TEXT = (
    "Risk levels (from the estimated probability that lead-time demand exceeds the inventory position):\n"
    "  Low      : probability <= 1 - service level (the service target is met; equals 'above the reorder point')\n"
    "  Medium   : above that, below 20%\n"
    "  High     : 20% to below 50%\n"
    "  Critical : 50% or more (expected demand during the lead time already exceeds the stock)"
)


# ================================================================ (B) simulated labels
def add_stockout_label(frame: pd.DataFrame, lead_time: int) -> pd.DataFrame:
    """SIMULATED label: 1 if recorded sales of days t..t+L-1 exceed inventory_level on day t.

    Window days are looked up by calendar date, so a missing day gives NaN (row is dropped later) instead
    of silently shifting. Rows in the last L-1 days of a series have no complete window and also get NaN.
    """
    if int(lead_time) != lead_time or lead_time < 1:
        raise ValueError("lead_time must be a whole number of days >= 1 for the simulated label.")
    lead_time = int(lead_time)
    frame = frame.sort_values([*GROUP_KEYS, DATE], kind="mergesort").reset_index(drop=True)
    total = np.full(len(frame), np.nan)
    sold, dates = frame[TARGET].to_numpy(float), frame[DATE].to_numpy()
    for _, idx in frame.groupby(GROUP_KEYS, sort=False).indices.items():
        s = pd.Series(sold[idx], index=pd.DatetimeIndex(dates[idx]))
        total[idx] = sum(s.reindex(s.index + pd.Timedelta(days=k)).to_numpy() for k in range(lead_time))
    frame["demand_next_L_days"] = total
    frame[LABEL_COL] = np.where(np.isnan(total), np.nan, (total > frame["inventory_level"].to_numpy(float)).astype(float))
    return frame


def build_classifier_frame(cleaned: pd.DataFrame, lead_time: int) -> tuple[pd.DataFrame, list[str]]:
    """Leakage-free features (history up to t-1 + stock on day t) and the simulated label."""
    frame, feature_cols = prepare_model_frame(cleaned, horizon=1)
    frame = add_stockout_label(frame, lead_time)
    frame["inventory_to_mean28"] = (frame["inventory_level"] / frame["roll_mean_28"]).replace([np.inf, -np.inf], np.nan)
    frame["rule_probability"] = stockout_probability(frame["inventory_level"], frame["roll_mean_28"],
                                                     frame["roll_std_28"].fillna(0), lead_time)
    frame = frame[frame[LABEL_COL].notna()].reset_index(drop=True)
    classes = frame[LABEL_COL].nunique()
    if classes < 2:
        only = int(frame[LABEL_COL].iloc[0]) if len(frame) else None
        raise DataValidationError(
            f"The simulated label has only one class (always {only}) for lead_time={lead_time}. "
            "In this dataset units sold never exceed the stock on hand, so a 1-day window can never be a stock-out; "
            "use a longer lead time (2-4 days) for the experiment.")
    return frame, feature_cols + ["inventory_level", "inventory_to_mean28"]


def purge_before(df: pd.DataFrame, boundary: pd.Timestamp, lead_time: int) -> pd.DataFrame:
    """Drop rows whose label window (date .. date+L-1) reaches `boundary` or later.

    Without this, labels of the last training days would be built from sales in the validation period.
    """
    return df[df[DATE] + pd.Timedelta(days=int(lead_time) - 1) < boundary]


# ================================================================ metrics and thresholds
def classification_metrics(y_true, proba, threshold: float) -> dict:
    """Precision, recall, F1, F2 (recall-weighted), specificity, accuracy, AUCs, confusion matrix, per-class figures."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    both = len(np.unique(y)) == 2
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": div(2 * precision * recall, precision + recall),
            "f2": div(5 * precision * recall, 4 * precision + recall),
            "specificity": div(tn, tn + fp), "accuracy": div(tp + tn, len(y)),
            "balanced_accuracy": (div(tp, tp + fn) + div(tn, tn + fp)) / 2,
            "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
            "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
            "precision_no_stockout": div(tn, tn + fn), "recall_no_stockout": div(tn, tn + fp),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def threshold_for_recall(y_true, proba, target_recall: float = 0.90) -> float:
    """Highest threshold that still catches at least `target_recall` of the stock-out class (on validation data)."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1].")
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


# ================================================================ classifier object (picklable, module level)
class StockoutClassifier:
    """Imputer (+ scaler) + classifier. Everything is fitted on the training rows only. SIMULATED label."""

    label_is_simulated = True

    def __init__(self, name: str, estimator, feature_names: list[str], scale: bool, lead_time: int):
        self.name, self.feature_names, self.lead_time, self.threshold = name, list(feature_names), int(lead_time), 0.5
        steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else []) + [("model", estimator)]
        self.pipeline = Pipeline(steps)

    def fit(self, df: pd.DataFrame):
        y = df[LABEL_COL].astype(int)
        if y.nunique() < 2:
            raise DataValidationError("Training data contains only one class; cannot train a classifier.")
        self.pipeline.fit(df[self.feature_names], y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise DataValidationError(f"Cannot predict: missing feature column(s) {missing}")
        return self.pipeline.predict_proba(df[self.feature_names])[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)


# ================================================================ experiment
def run_classifier_experiment(cleaned: pd.DataFrame, lead_time: int = 2, service_level: float = 0.95,
                              target_recall: float = 0.90, test_days: int = 90, val_days: int = 90,
                              n_estimators: int = 100, seed: int = 42, models_dir: Path = MODELS_DIR,
                              reports_dir: Path = REPORTS_DIR, save: bool = True) -> dict:
    """SIMULATED stock-out classification: Logistic Regression and Random Forest vs the rule-based score."""
    frame, feats = build_classifier_frame(cleaned, lead_time)
    train, val, test = chronological_split(frame, test_days=test_days, val_days=val_days)
    val_start, test_start = val[DATE].min(), test[DATE].min()
    train_p = purge_before(train, val_start, lead_time)
    val_p = purge_before(val, test_start, lead_time)
    trainval_p = purge_before(pd.concat([train, val], ignore_index=True), test_start, lead_time)
    for name, part in (("training", train_p), ("validation", val_p), ("test", test)):
        if part[LABEL_COL].nunique() < 2:
            raise DataValidationError(f"The {name} period contains only one class after purging; use more data or a different lead time.")
    log.info("L=%d | simulated label prevalence: train %.1f%%, validation %.1f%%, test %.1f%%", lead_time,
             100 * train_p[LABEL_COL].mean(), 100 * val_p[LABEL_COL].mean(), 100 * test[LABEL_COL].mean())

    def make(name):
        if name == "Logistic Regression":
            return StockoutClassifier(name, LogisticRegression(max_iter=2000, class_weight="balanced"), feats, True, lead_time)
        return StockoutClassifier(name, RandomForestClassifier(n_estimators=n_estimators, max_depth=8, min_samples_leaf=50,
                                                               max_features=0.5, class_weight="balanced_subsample",
                                                               n_jobs=-1, random_state=seed), feats, False, lead_time)

    names = ["Logistic Regression", "Random Forest"]
    # -- validation: fit on purged train, choose the operating threshold that reaches the target recall
    val_rows, thresholds, val_scores = [], {}, {}
    val_scores["Rule-based probability"] = val_p["rule_probability"].to_numpy()
    for n in names:
        val_scores[n] = make(n).fit(train_p).predict_proba(val_p)
    rule_trigger = 1 - service_level                                    # operational rule: reorder when position <= ROP
    thresholds["Rule-based reorder-point trigger"] = rule_trigger
    thresholds["Rule-based probability"] = threshold_for_recall(val_p[LABEL_COL], val_scores["Rule-based probability"], target_recall)
    for n in names:
        thresholds[n] = threshold_for_recall(val_p[LABEL_COL], val_scores[n], target_recall)
    val_scores["Rule-based reorder-point trigger"] = val_scores["Rule-based probability"]
    for n, s in val_scores.items():
        val_rows.append({"model": n, **classification_metrics(val_p[LABEL_COL], s, thresholds[n])})
    val_table = pd.DataFrame(val_rows)

    # -- test: refit on purged train+validation, evaluate once
    test_frame = test[[DATE, *GROUP_KEYS, "inventory_level", "demand_next_L_days", LABEL_COL]].copy()
    test_frame["Rule-based probability"] = test["rule_probability"].to_numpy()
    test_frame["Rule-based reorder-point trigger"] = test["rule_probability"].to_numpy()
    fitted = {}
    for n in names:
        fitted[n] = make(n).fit(trainval_p)
        fitted[n].threshold = thresholds[n]
        test_frame[n] = fitted[n].predict_proba(test)
    test_rows = [{"model": n, **classification_metrics(test[LABEL_COL], test_frame[n], thresholds[n])} for n in
                 ["Rule-based reorder-point trigger", "Rule-based probability", *names]]
    prev = float(test[LABEL_COL].mean())
    test_rows.append({"model": "Flag everything (baseline)", **classification_metrics(test[LABEL_COL], np.ones(len(test)), 0.5)})
    test_table = pd.DataFrame(test_rows)

    best = val_table[val_table["model"].isin(names)].sort_values("pr_auc", ascending=False).iloc[0]["model"]
    importance = None
    if best == "Random Forest":
        importance = pd.Series(fitted[best].pipeline.named_steps["model"].feature_importances_, index=feats).sort_values(ascending=False)
    elif best == "Logistic Regression":
        importance = pd.Series(np.abs(fitted[best].pipeline.named_steps["model"].coef_[0]), index=feats).sort_values(ascending=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "SIMULATED_LABEL": True,
        "label_definition": f"1 if the recorded units sold on days t..t+{lead_time - 1} sum to more than inventory_level on day t "
                            "(hypothetical: assumes no delivery arrives). It is NOT an observed stock-out.",
        "decision_moment": "start of day t: stock on hand = inventory_level(t); history up to day t-1",
        "lead_time_days": int(lead_time), "model_name": best, "threshold": thresholds[best], "target_recall_on_validation": target_recall,
        "feature_columns": feats, "label_prevalence_test": prev,
        "periods": {"train": [str(train_p[DATE].min().date()), str(train_p[DATE].max().date())],
                    "validation": [str(val_p[DATE].min().date()), str(val_p[DATE].max().date())],
                    "test": [str(test[DATE].min().date()), str(test[DATE].max().date())]},
        "purge": f"rows whose label window reaches the next period were removed ({lead_time - 1} day(s))",
        "test_metrics": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in
                         test_table[test_table.model == best].iloc[0].drop("model").to_dict().items()},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "warning": "Synthetic dataset AND simulated label. Not evidence of real stock-out performance.",
    }
    res = {"lead_time": int(lead_time), "service_level": service_level, "target_recall": target_recall, "val_table": val_table,
           "test_table": test_table, "best": best, "model": fitted[best], "fitted": fitted, "thresholds": thresholds,
           "importance": importance, "test_frame": test_frame, "metadata": meta, "prevalence": prev}
    if save:
        write_classifier_outputs(res, models_dir, reports_dir)
    return res


# ================================================================ saving / loading / scoring
def classifier_paths(lead_time: int, models_dir: Path = MODELS_DIR) -> tuple[Path, Path]:
    return (models_dir / f"stockout_classifier_simulated_L{lead_time}.joblib",
            models_dir / f"stockout_classifier_simulated_L{lead_time}_metadata.json")


def load_classifier(lead_time: int = 2, models_dir: Path = MODELS_DIR):
    mpath, jpath = classifier_paths(lead_time, models_dir)
    if not mpath.exists() or not jpath.exists():
        raise DataFileError(f"No simulated stock-out classifier for lead time {lead_time} day(s) in {models_dir}.\n"
                            f"Train it first:  python -m src.stockout_prediction --lead-time {lead_time}")
    return joblib.load(mpath), json.loads(jpath.read_text(encoding="utf-8"))


def score_snapshot(model: StockoutClassifier, cleaned: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """SIMULATED-label classifier probabilities for every store-product at the snapshot date (default: last date)."""
    from src.preprocess import build_features

    if cleaned.empty:
        raise DataValidationError("No data to score.")
    snap = pd.Timestamp(snapshot_date) if snapshot_date is not None else cleaned[DATE].max()
    feats = build_features(cleaned, horizon=1).replace([np.inf, -np.inf], np.nan)
    feats["inventory_to_mean28"] = (feats["inventory_level"] / feats["roll_mean_28"]).replace([np.inf, -np.inf], np.nan)
    rows = feats[feats[DATE] == snap]
    if rows.empty:
        raise DataValidationError(f"No records on {snap.date()}.")
    out = rows[GROUP_KEYS].copy()
    out["snapshot_date"] = snap
    out["simulated_stockout_probability"] = model.predict_proba(rows)
    out["simulated_flag"] = out["simulated_stockout_probability"] >= model.threshold
    out["note"] = f"SIMULATED label ({model.lead_time}-day window); not a real stock-out outcome"
    return out.reset_index(drop=True)


def write_classifier_outputs(res: dict, models_dir: Path, reports_dir: Path) -> None:
    ensure_dirs()
    models_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    L = res["lead_time"]
    mpath, jpath = classifier_paths(L, models_dir)
    joblib.dump(res["model"], mpath)
    jpath.write_text(json.dumps(res["metadata"], indent=2, default=str), encoding="utf-8")
    res["val_table"].merge(res["test_table"], on="model", how="outer", suffixes=("_val", "_test")).to_csv(
        reports_dir / f"stockout_classifier_simulated_L{L}_comparison.csv", index=False)
    (reports_dir / f"stockout_classifier_simulated_L{L}_summary.txt").write_text(build_classifier_summary(res), encoding="utf-8")
    plot_classifier_evaluation(res).savefig(reports_dir / "figures" / f"stockout_classifier_simulated_L{L}.png", dpi=120)
    # test-period scores of every model: lets the dashboard draw ROC / precision-recall / calibration curves from real numbers
    res["test_frame"].to_csv(reports_dir / f"stockout_test_predictions_simulated_L{L}.csv", index=False)
    log.info("L=%d | saved %s, %s, summary, comparison and figure", L, mpath.name, jpath.name)


# ================================================================ plots / summary
def plot_classifier_evaluation(res: dict) -> Figure:
    f = res["test_frame"]
    y = f[LABEL_COL].astype(int).to_numpy()
    fig = Figure(figsize=(17, 5.5), layout="constrained")
    ax = fig.subplots(1, 3)
    for name, color in (("Rule-based probability", "#7F7F7F"), ("Logistic Regression", "#4C78A8"), ("Random Forest", "#E45756")):
        prec, rec, _ = precision_recall_curve(y, f[name])
        ax[0].plot(rec, prec, color=color, lw=1.8, label=f"{name} (PR-AUC {average_precision_score(y, f[name]):.3f})")
        r = res["test_table"].set_index("model").loc[name]
        ax[0].scatter([r["recall"]], [r["precision"]], color=color, s=60, zorder=3)
    ax[0].axhline(y.mean(), color="black", ls=":", label=f"Chance = prevalence {y.mean():.2f}")
    ax[0].set(title="SIMULATED label: precision-recall (test); dots = chosen operating point", xlabel="Recall (stock-out class)",
              ylabel="Precision", ylim=(0, 1.02), xlim=(0, 1.02))
    ax[0].legend(loc="lower left", fontsize=9)
    import seaborn as sns

    t = res["test_table"].set_index("model")
    for a, name in ((ax[1], res["best"]), (ax[2], "Rule-based reorder-point trigger")):
        r = t.loc[name]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]]).astype(int)
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=a, xticklabels=["No stock-out", "Stock-out"],
                    yticklabels=["No stock-out", "Stock-out"])
        a.set(title=f"{name}\nthreshold {r['threshold']:.3f}: precision {r['precision']:.2f}, recall {r['recall']:.2f}",
              xlabel="Predicted", ylabel="Actual (simulated label)")
    fig.suptitle(f"SIMULATED stock-out classification, lead time {res['lead_time']} days (synthetic data, hypothetical label)", fontsize=13)
    return fig


def build_classifier_summary(res: dict) -> str:
    fmt = lambda t: t.to_string(index=False, float_format=lambda v: f"{v:,.3f}")  # noqa: E731
    cols = ["model", "threshold", "precision", "recall", "f1", "f2", "specificity", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
    m = res["metadata"]
    t = res["test_table"].set_index("model")
    best, rule = t.loc[res["best"]], t.loc["Rule-based reorder-point trigger"]
    L = res["lead_time"]
    lines = [f"SIMULATED STOCK-OUT CLASSIFIER - lead time {L} day(s)", "=" * 78,
             "!! THE LABEL IS SIMULATED. The dataset has no real stock-out records. !!",
             f"Label: {m['label_definition']}",
             f"Decision moment: {m['decision_moment']}",
             f"Purging: {m['purge']} (prevents labels of one period using sales of the next).",
             f"Test period {m['periods']['test'][0]}..{m['periods']['test'][1]}: {100 * res['prevalence']:.1f}% of rows are simulated stock-outs.", "",
             f"Operating thresholds: ML models and the rule probability use the highest threshold that still catches >= {res['target_recall']:.0%} of "
             "stock-outs on VALIDATION data; the rule-based trigger uses its documented threshold (probability >= 1 - service level, i.e. 'at or below the reorder point').", "",
             "VALIDATION (fit on train)", fmt(res["val_table"][cols]), "",
             "TEST (fit on train+validation)", fmt(res["test_table"][cols]), "",
             f"Selected ML model (highest validation PR-AUC): {res['best']}", "",
             "INTERPRETATION (generated from the numbers above)"]
    lines.append(f"- Stock-out class: {res['best']} reaches recall {best['recall']:.2f} with precision {best['precision']:.2f} (F1 {best['f1']:.2f}); "
                 f"the reorder-point trigger reaches recall {rule['recall']:.2f} with precision {rule['precision']:.2f}.")
    lines.append(f"- Missed stock-outs (false negatives) on test: {int(best['fn'])} for {res['best']} vs {int(rule['fn'])} for the rule trigger; "
                 f"false alarms: {int(best['fp'])} vs {int(rule['fp'])}.")
    better = best["roc_auc"] - t.loc["Rule-based probability", "roc_auc"]
    lines.append(f"- Ranking quality (ROC-AUC): {res['best']} {best['roc_auc']:.3f} vs rule-based probability {t.loc['Rule-based probability', 'roc_auc']:.3f} "
                 f"({better:+.3f}). " + ("The learned model separates risky from safe rows better than the formula."
                                         if better > 0.01 else "The learned model is not clearly better than the formula."))
    lines.append(f"- Baseline 'flag everything' has recall 1.00 but precision only {t.loc['Flag everything (baseline)', 'precision']:.2f}: a useful model must "
                 "keep recall high while raising precision above that.")
    lines += ["", "CAUTIONS",
              "- The label is hypothetical (no deliveries assumed, real lost sales never observed) and the data is synthetic. In this file sales are a random share of the day's stock, "
              "so the learned model may exploit a data-generation pattern that real retail demand would not show.",
              "- Do not present these scores as real historical stock-out performance. The rule-based risk assessment is a formula, not this classifier."]
    return "\n".join(lines) + "\n"


# ================================================================ CLI
def main() -> None:
    p = argparse.ArgumentParser(description="Simulated-label stock-out classifier experiment.")
    p.add_argument("--input", default=str(CLEANED_CSV))
    p.add_argument("--lead-time", type=int, default=2, help="lead time in whole days (label window)")
    p.add_argument("--service-level", type=float, default=0.95)
    p.add_argument("--target-recall", type=float, default=0.90)
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--val-days", type=int, default=90)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    p.add_argument("--reports-dir", default=str(REPORTS_DIR))
    a = p.parse_args()
    try:
        r = run_classifier_experiment(load_cleaned(a.input), a.lead_time, a.service_level, a.target_recall, a.test_days,
                                      a.val_days, a.n_estimators, models_dir=Path(a.models_dir), reports_dir=Path(a.reports_dir))
    except (DataFileError, DataValidationError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    t = r["test_table"].set_index("model")
    for name in ("Rule-based reorder-point trigger", "Logistic Regression", "Random Forest"):
        log.info("SIMULATED | %-34s recall %.2f precision %.2f F1 %.2f ROC-AUC %.3f", name, t.loc[name, "recall"],
                 t.loc[name, "precision"], t.loc[name, "f1"], t.loc[name, "roc_auc"])
    log.info("Best ML model on validation PR-AUC: %s", r["best"])


if __name__ == "__main__":
    # import under the real module name so pickled classes are 'src.stockout_prediction.X', not '__main__.X'
    from src.stockout_prediction import main as _main

    _main()
