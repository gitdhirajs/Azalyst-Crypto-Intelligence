"""Main ML trainer for the scanner snapshot model."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier

from src.config import (
    FEATURES_DIR,
    LABEL_HORIZON_MINUTES,
    LOGS_DIR,
    MIN_SAMPLES_TO_TRAIN,
    ML_RANDOM_STATE,
    ML_TEST_SIZE,
    MODELS_DIR,
)


FEATURE_COLS = [
    "rsi_1m",
    "rsi_5m",
    "rsi_15m",
    "rsi_1h",
    "rsi_4h",
    "rsi_1d",
    "rsi_mean",
    "rsi_std",
    "rsi_min",
    "rsi_max",
    "rsi_range",
    "rsi_divergence_5m_4h",
    "rsi_divergence_15m_1h",
    "rsi_divergence_1m_1h",
    "n_tf_oversold",
    "n_tf_overbought",
    "oi_change_pct_1h",
    "oi_rising",
    "oi_spike",
    "oi_contracts",
    "price_change_pct_24h",
    "price_pct_change",
    "price_rolling_std_6",
    "oi_pct_change",
    "funding_rate",
    "rsi_5m_slope",
]
LABEL_COL = "label"
MODEL_PARAMS = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "random_state": ML_RANDOM_STATE,
    "tree_method": "hist",
}


def _get_latest_features() -> pd.DataFrame:
    path = FEATURES_DIR / "latest_features.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "scan_time" in df.columns:
        df["scan_time"] = pd.to_datetime(df["scan_time"], utc=True)
    return df


def _prepare_data(df: pd.DataFrame):
    clean = df.dropna(subset=[LABEL_COL]).copy().reset_index(drop=True)
    available = [column for column in FEATURE_COLS if column in clean.columns]
    if clean.empty or not available:
        return clean, pd.DataFrame(), pd.Series(dtype=int), [], pd.Series(dtype=float)

    X = clean[available].apply(pd.to_numeric, errors="coerce")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    y = clean[LABEL_COL].astype(int)
    return clean, X, y, available, medians


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _fallback_random_split(y: pd.Series):
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=ML_TEST_SIZE,
        stratify=y if y.nunique() > 1 else None,
        random_state=ML_RANDOM_STATE,
    )
    return train_idx, test_idx, "random_row_fallback"


def _time_split(clean: pd.DataFrame, y: pd.Series):
    if "scan_time" not in clean.columns or len(clean) < 10:
        return _fallback_random_split(y)

    ordered_idx = clean.sort_values("scan_time").index.to_numpy()
    cut = int(round(len(ordered_idx) * (1 - ML_TEST_SIZE)))
    cut = max(1, min(cut, len(ordered_idx) - 1))
    train_idx = ordered_idx[:cut]
    test_idx = ordered_idx[cut:]

    if len(train_idx) == 0 or len(test_idx) == 0:
        return _fallback_random_split(y)

    return train_idx, test_idx, "time_ordered_holdout"


def _fit_and_score(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[XGBClassifier, dict]:
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)

    y_test = y.iloc[test_idx]
    y_pred = model.predict(X.iloc[test_idx])
    y_proba = model.predict_proba(X.iloc[test_idx])[:, 1]

    return model, {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": (
            round(_safe_auc(y_test, y_proba), 4)
            if _safe_auc(y_test, y_proba) is not None
            else None
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def _bucket_analysis(df: pd.DataFrame) -> dict:
    analyses = {}
    specs = {
        "rsi_1h": (
            [-np.inf, 35, 50, 65, 80, np.inf],
            ["<=35", "35-50", "50-65", "65-80", "80+"],
        ),
        "oi_change_pct_1h": (
            [-np.inf, -5, -1, 1, 5, np.inf],
            ["<=-5", "-5 to -1", "-1 to 1", "1 to 5", "5+"],
        ),
        "price_change_pct_24h": (
            [-np.inf, -10, -3, 0, 3, 10, np.inf],
            ["<=-10", "-10 to -3", "-3 to 0", "0 to 3", "3 to 10", "10+"],
        ),
        "funding_rate": (
            [-np.inf, -0.001, 0, 0.001, np.inf],
            ["<=-0.10%", "-0.10% to 0", "0 to 0.10%", "0.10%+"],
        ),
    }

    for feature_name, (bins, labels) in specs.items():
        if feature_name not in df.columns:
            continue

        work = df[[feature_name, LABEL_COL]].dropna().copy()
        if work.empty:
            continue

        work["bucket"] = pd.cut(
            work[feature_name],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        grouped = (
            work.dropna(subset=["bucket"])
            .groupby("bucket", observed=False)[LABEL_COL]
            .agg(["count", "mean"])
            .reset_index()
        )
        grouped = grouped[grouped["count"] >= 8].copy()
        if grouped.empty:
            continue

        grouped["bucket"] = grouped["bucket"].astype(str)
        grouped["win_rate"] = grouped["mean"].round(4)
        grouped = grouped.drop(columns=["mean"]).sort_values(
            ["win_rate", "count"], ascending=[False, False]
        )
        analyses[feature_name] = {
            "top_bullish_ranges": grouped.head(5).to_dict(orient="records"),
            "top_bearish_ranges": grouped.sort_values(
                ["win_rate", "count"], ascending=[True, False]
            )
            .head(5)
            .to_dict(orient="records"),
        }

    if {"rsi_1h", "oi_change_pct_1h", "price_change_pct_24h"}.issubset(df.columns):
        cluster_df = df[["rsi_1h", "oi_change_pct_1h", "price_change_pct_24h", LABEL_COL]].dropna()
        if not cluster_df.empty:
            cluster_df = cluster_df.copy()
            cluster_df["rsi_band"] = pd.cut(
                cluster_df["rsi_1h"],
                bins=[-np.inf, 35, 50, 65, 80, np.inf],
                labels=["<=35", "35-50", "50-65", "65-80", "80+"],
                include_lowest=True,
            )
            cluster_df["oi_band"] = pd.cut(
                cluster_df["oi_change_pct_1h"],
                bins=[-np.inf, -5, -1, 1, 5, np.inf],
                labels=["<=-5", "-5 to -1", "-1 to 1", "1 to 5", "5+"],
                include_lowest=True,
            )
            cluster_df["price_band"] = pd.cut(
                cluster_df["price_change_pct_24h"],
                bins=[-np.inf, -10, -3, 0, 3, 10, np.inf],
                labels=["<=-10", "-10 to -3", "-3 to 0", "0 to 3", "3 to 10", "10+"],
                include_lowest=True,
            )

            clusters = (
                cluster_df.groupby(
                    ["rsi_band", "oi_band", "price_band"], observed=False
                )[LABEL_COL]
                .agg(["count", "mean"])
                .reset_index()
            )
            clusters = clusters[clusters["count"] >= 6].copy()
            if not clusters.empty:
                clusters["cluster"] = clusters.apply(
                    lambda row: (
                        f"RSI {row['rsi_band']} | "
                        f"OI {row['oi_band']} | "
                        f"24h {row['price_band']}"
                    ),
                    axis=1,
                )
                clusters["win_rate"] = clusters["mean"].round(4)
                clusters = clusters.sort_values(
                    ["win_rate", "count"], ascending=[False, False]
                )
                analyses["scanner_clusters"] = {
                    "top_gainer_clusters": clusters[["cluster", "count", "win_rate"]]
                    .head(7)
                    .to_dict(orient="records"),
                    "top_loser_clusters": clusters.sort_values(
                        ["win_rate", "count"], ascending=[True, False]
                    )[["cluster", "count", "win_rate"]]
                    .head(7)
                    .to_dict(orient="records"),
                }

    return analyses


def _symbol_generalization(clean: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict | None:
    if "symbol" not in clean.columns or clean["symbol"].nunique() < 10:
        return None

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=clean["symbol"]))
    _, metrics = _fit_and_score(X, y, train_idx, test_idx)
    return {
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
    }


def train_model(force: bool = False) -> dict:
    """Train or retrain the main scanner model."""
    df = _get_latest_features()
    if df.empty:
        return {"status": "no_data"}

    clean, X, y, used_cols, medians = _prepare_data(df)
    n_samples = len(clean)
    n_symbols = clean["symbol"].nunique() if "symbol" in clean.columns else 0

    if n_samples == 0:
        return {"status": "no_labeled_data"}

    if y.nunique() < 2:
        return {"status": "single_class", "n_samples": n_samples, "n_symbols": n_symbols}

    if n_samples < MIN_SAMPLES_TO_TRAIN and not force:
        return {
            "status": "not_enough_data",
            "n_samples": n_samples,
            "n_symbols": n_symbols,
            "need": MIN_SAMPLES_TO_TRAIN,
        }

    train_idx, test_idx, split_name = _time_split(clean, y)
    eval_model, metrics = _fit_and_score(X, y, train_idx, test_idx)

    final_model = XGBClassifier(**MODEL_PARAMS)
    final_model.fit(X, y, verbose=False)

    importances = dict(zip(used_cols, final_model.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10]

    baseline_accuracy = round(float(max(y.mean(), 1 - y.mean())), 4)
    report = {
        "status": "trained",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": "main_scanner_xgboost",
        "evaluation_split": split_name,
        "label_horizon_minutes": LABEL_HORIZON_MINUTES,
        "n_samples": int(n_samples),
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "n_symbols": int(n_symbols),
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "baseline_accuracy": baseline_accuracy,
        "label_distribution": y.value_counts().to_dict(),
        "top_features": top_features,
        "bucket_analysis": _bucket_analysis(clean),
        "symbol_generalization": _symbol_generalization(clean, X, y),
        "classification_report": metrics["classification_report"],
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    joblib.dump(final_model, MODELS_DIR / f"model_{ts}.joblib")
    joblib.dump(final_model, MODELS_DIR / "latest_model.joblib")
    joblib.dump(used_cols, MODELS_DIR / "feature_columns.joblib")
    joblib.dump(medians.to_dict(), MODELS_DIR / "feature_medians.joblib")

    report_path = LOGS_DIR / f"train_report_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    with open(LOGS_DIR / "latest_train_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    history_path = LOGS_DIR / "training_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, default=str) + "\n")

    return report


def predict_current(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Predict probabilities for the current scan snapshot."""
    model_path = MODELS_DIR / "latest_model.joblib"
    cols_path = MODELS_DIR / "feature_columns.joblib"
    medians_path = MODELS_DIR / "feature_medians.joblib"

    if not model_path.exists() or not cols_path.exists():
        scan_df["ml_prediction"] = None
        scan_df["ml_probability"] = None
        return scan_df

    model = joblib.load(model_path)
    used_cols = joblib.load(cols_path)
    medians = joblib.load(medians_path) if medians_path.exists() else {}

    available = [column for column in used_cols if column in scan_df.columns]
    X = scan_df[available].copy().apply(pd.to_numeric, errors="coerce")

    for column in used_cols:
        if column not in X.columns:
            X[column] = medians.get(column, 0.0)

    X = X[used_cols]
    for column in used_cols:
        X[column] = X[column].fillna(medians.get(column, 0.0))

    scan_df["ml_prediction"] = model.predict(X)
    scan_df["ml_probability"] = model.predict_proba(X)[:, 1]
    return scan_df
