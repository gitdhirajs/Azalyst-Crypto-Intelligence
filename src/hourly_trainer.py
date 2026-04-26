"""Hourly candle-pattern ML pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.collector import compute_rsi, fetch_klines, fetch_oi_history, get_active_symbols
from src.config import (
    HOURLY_CONTINUATION_THRESHOLD_PCT,
    HOURLY_DIR,
    HOURLY_FORWARD_CANDLES,
    HOURLY_KLINE_LIMIT,
    HOURLY_MIN_SAMPLES_TO_TRAIN,
    HOURLY_REPORT_TOP_N,
    HOURLY_SYMBOL_LIMIT,
    LOGS_DIR,
    ML_RANDOM_STATE,
    ML_TEST_SIZE,
    MODELS_DIR,
    REPORTS_DIR,
)


FEATURE_COLS = [
    "candle_return_pct",
    "body_pct",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "body_to_range",
    "close_position",
    "direction",
    "volume_change_pct_1h",
    "volume_ratio_6h",
    "volume_ratio_12h",
    "volume_zscore_6h",
    "return_3h",
    "return_6h",
    "volatility_6h",
    "rsi_1h",
    "rsi_mean_3h",
    "rsi_mean_6h",
    "rsi_persist_above_60_6h",
    "rsi_persist_50_65_6h",
    "oi_change_pct_1h",
    "oi_change_pct_3h",
    "oi_change_pct_6h",
    "oi_ratio_6h",
    "oi_zscore_6h",
    "price_oi_alignment",
    "price_volume_alignment",
    "body_vs_avg_12h",
    "range_vs_avg_12h",
    "current_price_change_pct_24h",
]
LABEL_COL = "continuation_label"
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "random_state": ML_RANDOM_STATE,
    "tree_method": "hist",
}


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=2).mean()
    rolling_std = series.rolling(window, min_periods=2).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def _forward_extreme(series: pd.Series, lookahead: int, kind: str) -> pd.Series:
    shifted = [series.shift(-step) for step in range(1, lookahead + 1)]
    frame = pd.concat(shifted, axis=1)
    if kind == "max":
        return frame.max(axis=1)
    return frame.min(axis=1)


def collect_hourly_market_data(
    symbol_limit: int | None = None,
    kline_limit: int | None = None,
) -> pd.DataFrame:
    """Collect recent 1h candle data and hourly OI history for active symbols."""
    limit = kline_limit or HOURLY_KLINE_LIMIT
    top_n = symbol_limit or HOURLY_SYMBOL_LIMIT

    symbols, bulk_tickers = get_active_symbols()
    if top_n > 0:
        symbols = symbols[:top_n]

    frames = []
    for symbol in symbols:
        ticker = bulk_tickers.get(symbol, {})
        provider = ticker.get("_provider")

        candles = fetch_klines(symbol, "1h", limit=limit, provider=provider)
        if candles is None or candles.empty:
            continue

        candles = candles[
            ["open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume"]
        ].copy()
        candles["symbol"] = symbol
        candles["rsi_1h"] = compute_rsi(candles["close"])

        oi_hist = fetch_oi_history(symbol, period="1h", limit=limit, provider=provider)
        if oi_hist is not None and not oi_hist.empty:
            oi_hist = oi_hist.rename(
                columns={
                    "timestamp": "oi_time",
                    "sumOpenInterest": "oi_contracts",
                    "sumOpenInterestValue": "oi_value_now",
                }
            )[["oi_time", "oi_contracts", "oi_value_now"]]
            candles = pd.merge_asof(
                candles.sort_values("close_time"),
                oi_hist.sort_values("oi_time"),
                left_on="close_time",
                right_on="oi_time",
                direction="backward",
            )
        else:
            candles["oi_contracts"] = np.nan
            candles["oi_value_now"] = np.nan

        candles["current_price_change_pct_24h"] = ticker.get("price_change_pct_24h")
        candles["current_volume_24h"] = ticker.get("volume_24h")
        candles["market_provider"] = provider
        frames.append(candles)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    ts = pd.Timestamp.now(tz="utc").strftime("%Y%m%d_%H%M%S")
    df.to_csv(HOURLY_DIR / f"hourly_market_{ts}.csv", index=False)
    df.to_csv(HOURLY_DIR / "latest_hourly_market.csv", index=False)
    return df


def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build hourly candle-pattern features and continuation labels."""
    if df.empty:
        return df

    parts = []
    for _, group in df.groupby("symbol", sort=False):
        group = group.sort_values("close_time").copy()
        candle_range = (group["high"] - group["low"]).replace(0, np.nan)

        group["candle_return_pct"] = (group["close"] - group["open"]) / group["open"] * 100
        group["body_pct"] = (group["close"] - group["open"]).abs() / group["open"] * 100
        group["range_pct"] = candle_range / group["open"] * 100
        group["upper_wick_pct"] = (
            group["high"] - group[["open", "close"]].max(axis=1)
        ) / group["open"] * 100
        group["lower_wick_pct"] = (
            group[["open", "close"]].min(axis=1) - group["low"]
        ) / group["open"] * 100
        group["body_to_range"] = group["body_pct"] / group["range_pct"].replace(0, np.nan)
        group["close_position"] = (group["close"] - group["low"]) / candle_range
        group["direction"] = np.where(group["candle_return_pct"] >= 0, 1, -1)

        group["volume_change_pct_1h"] = group["volume"].pct_change() * 100
        group["volume_ratio_6h"] = group["volume"] / group["volume"].rolling(6, min_periods=2).mean()
        group["volume_ratio_12h"] = group["volume"] / group["volume"].rolling(12, min_periods=3).mean()
        group["volume_zscore_6h"] = _rolling_zscore(group["volume"], 6)

        group["return_3h"] = group["close"].pct_change(3) * 100
        group["return_6h"] = group["close"].pct_change(6) * 100
        group["volatility_6h"] = group["candle_return_pct"].rolling(6, min_periods=2).std()

        group["rsi_mean_3h"] = group["rsi_1h"].rolling(3, min_periods=1).mean()
        group["rsi_mean_6h"] = group["rsi_1h"].rolling(6, min_periods=1).mean()
        group["rsi_persist_above_60_6h"] = (group["rsi_1h"] >= 60).rolling(6, min_periods=1).mean()
        group["rsi_persist_50_65_6h"] = (
            ((group["rsi_1h"] >= 50) & (group["rsi_1h"] <= 65)).rolling(6, min_periods=1).mean()
        )

        if "oi_value_now" in group.columns:
            group["oi_change_pct_1h"] = group["oi_value_now"].pct_change() * 100
            group["oi_change_pct_3h"] = group["oi_value_now"].pct_change(3) * 100
            group["oi_change_pct_6h"] = group["oi_value_now"].pct_change(6) * 100
            group["oi_ratio_6h"] = group["oi_value_now"] / group["oi_value_now"].rolling(
                6, min_periods=2
            ).mean()
            group["oi_zscore_6h"] = _rolling_zscore(group["oi_value_now"], 6)
        else:
            group["oi_change_pct_1h"] = np.nan
            group["oi_change_pct_3h"] = np.nan
            group["oi_change_pct_6h"] = np.nan
            group["oi_ratio_6h"] = np.nan
            group["oi_zscore_6h"] = np.nan

        group["price_oi_alignment"] = (
            np.sign(group["candle_return_pct"]).fillna(0)
            == np.sign(group["oi_change_pct_1h"]).fillna(0)
        ).astype(int)
        group["price_volume_alignment"] = (
            np.sign(group["candle_return_pct"]).fillna(0)
            == np.sign(group["volume_change_pct_1h"]).fillna(0)
        ).astype(int)
        group["body_vs_avg_12h"] = group["body_pct"] / group["body_pct"].rolling(
            12, min_periods=3
        ).mean()
        group["range_vs_avg_12h"] = group["range_pct"] / group["range_pct"].rolling(
            12, min_periods=3
        ).mean()

        future_high = _forward_extreme(group["high"], HOURLY_FORWARD_CANDLES, "max")
        future_low = _forward_extreme(group["low"], HOURLY_FORWARD_CANDLES, "min")
        group["future_upside_pct"] = (future_high - group["close"]) / group["close"] * 100
        group["future_downside_pct"] = (future_low - group["close"]) / group["close"] * 100

        future_available = future_high.notna() & future_low.notna()
        bullish_now = group["candle_return_pct"] >= 0
        continuation = np.where(
            bullish_now,
            group["future_upside_pct"] >= HOURLY_CONTINUATION_THRESHOLD_PCT,
            group["future_downside_pct"] <= -HOURLY_CONTINUATION_THRESHOLD_PCT,
        )
        group[LABEL_COL] = np.where(future_available, continuation.astype(int), np.nan)
        parts.append(group)

    return pd.concat(parts, ignore_index=True)


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


def _time_split(clean: pd.DataFrame):
    ordered_idx = clean.sort_values("close_time").index.to_numpy()
    cut = int(round(len(ordered_idx) * (1 - ML_TEST_SIZE)))
    cut = max(1, min(cut, len(ordered_idx) - 1))
    return ordered_idx[:cut], ordered_idx[cut:]


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _fit_and_score(X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray):
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
    y_test = y.iloc[test_idx]
    y_pred = model.predict(X.iloc[test_idx])
    y_proba = model.predict_proba(X.iloc[test_idx])[:, 1]

    auc = _safe_auc(y_test, y_proba)
    return model, {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
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
        "body_pct": (
            [-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf],
            ["<=0.5", "0.5-1.0", "1.0-2.0", "2.0-4.0", "4.0+"],
        ),
        "volume_ratio_6h": (
            [-np.inf, 1.0, 1.5, 2.0, 3.0, np.inf],
            ["<=1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0+"],
        ),
        "oi_change_pct_1h": (
            [-np.inf, -10, -3, 0, 3, 10, np.inf],
            ["<=-10", "-10 to -3", "-3 to 0", "0 to 3", "3 to 10", "10+"],
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
        grouped = grouped[grouped["count"] >= 10].copy()
        if grouped.empty:
            continue

        grouped["bucket"] = grouped["bucket"].astype(str)
        grouped["continuation_rate"] = grouped["mean"].round(4)
        grouped = grouped.drop(columns=["mean"]).sort_values(
            ["continuation_rate", "count"], ascending=[False, False]
        )
        analyses[feature_name] = {
            "top_continuation_ranges": grouped.head(5).to_dict(orient="records"),
            "weak_continuation_ranges": grouped.sort_values(
                ["continuation_rate", "count"], ascending=[True, False]
            )
            .head(5)
            .to_dict(orient="records"),
        }

    if {"rsi_1h", "volume_ratio_6h", "oi_change_pct_1h"}.issubset(df.columns):
        cluster_df = df[["rsi_1h", "volume_ratio_6h", "oi_change_pct_1h", LABEL_COL]].dropna()
        if not cluster_df.empty:
            cluster_df = cluster_df.copy()
            cluster_df["rsi_band"] = pd.cut(
                cluster_df["rsi_1h"],
                bins=[-np.inf, 35, 50, 65, 80, np.inf],
                labels=["<=35", "35-50", "50-65", "65-80", "80+"],
                include_lowest=True,
            )
            cluster_df["volume_band"] = pd.cut(
                cluster_df["volume_ratio_6h"],
                bins=[-np.inf, 1.0, 1.5, 2.0, 3.0, np.inf],
                labels=["<=1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0+"],
                include_lowest=True,
            )
            cluster_df["oi_band"] = pd.cut(
                cluster_df["oi_change_pct_1h"],
                bins=[-np.inf, -10, -3, 0, 3, 10, np.inf],
                labels=["<=-10", "-10 to -3", "-3 to 0", "0 to 3", "3 to 10", "10+"],
                include_lowest=True,
            )

            clusters = (
                cluster_df.groupby(
                    ["rsi_band", "volume_band", "oi_band"], observed=False
                )[LABEL_COL]
                .agg(["count", "mean"])
                .reset_index()
            )
            clusters = clusters[clusters["count"] >= 8].copy()
            if not clusters.empty:
                clusters["cluster"] = clusters.apply(
                    lambda row: (
                        f"RSI {row['rsi_band']} | "
                        f"Vol {row['volume_band']} | "
                        f"OI {row['oi_band']}"
                    ),
                    axis=1,
                )
                clusters["continuation_rate"] = clusters["mean"].round(4)
                analyses["pattern_clusters"] = {
                    "top_gainer_clusters": clusters.sort_values(
                        ["continuation_rate", "count"], ascending=[False, False]
                    )[["cluster", "count", "continuation_rate"]]
                    .head(7)
                    .to_dict(orient="records"),
                    "top_failure_clusters": clusters.sort_values(
                        ["continuation_rate", "count"], ascending=[True, False]
                    )[["cluster", "count", "continuation_rate"]]
                    .head(7)
                    .to_dict(orient="records"),
                }

    return analyses


def _align_live_features(
    df: pd.DataFrame,
    used_cols: list[str],
    medians: dict,
) -> pd.DataFrame:
    X = df[[column for column in used_cols if column in df.columns]].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    for column in used_cols:
        if column not in X.columns:
            X[column] = medians.get(column, 0.0)
    X = X[used_cols]
    for column in used_cols:
        X[column] = X[column].fillna(medians.get(column, 0.0))
    return X


def train_hourly_model(
    force: bool = False,
    symbol_limit: int | None = None,
    kline_limit: int | None = None,
) -> dict:
    """Train the hourly candle-pattern continuation model."""
    market_df = collect_hourly_market_data(symbol_limit=symbol_limit, kline_limit=kline_limit)
    if market_df.empty:
        return {"status": "no_data"}

    feature_df = build_hourly_features(market_df)
    ts = pd.Timestamp.now(tz="utc").strftime("%Y%m%d_%H%M%S")
    feature_df.to_csv(HOURLY_DIR / f"hourly_features_{ts}.csv", index=False)
    feature_df.to_csv(HOURLY_DIR / "latest_hourly_features.csv", index=False)

    clean, X, y, used_cols, medians = _prepare_data(feature_df)
    n_samples = len(clean)
    n_symbols = clean["symbol"].nunique() if "symbol" in clean.columns else 0

    if n_samples == 0:
        return {"status": "no_labeled_data"}
    if y.nunique() < 2:
        return {"status": "single_class", "n_samples": n_samples, "n_symbols": n_symbols}
    if n_samples < HOURLY_MIN_SAMPLES_TO_TRAIN and not force:
        return {
            "status": "not_enough_data",
            "n_samples": n_samples,
            "n_symbols": n_symbols,
            "need": HOURLY_MIN_SAMPLES_TO_TRAIN,
        }

    train_idx, test_idx = _time_split(clean)
    _, metrics = _fit_and_score(X, y, train_idx, test_idx)

    final_model = XGBClassifier(**MODEL_PARAMS)
    final_model.fit(X, y, verbose=False)

    importances = dict(zip(used_cols, final_model.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10]

    latest_rows = feature_df.sort_values("close_time").groupby("symbol", as_index=False).tail(1)
    live_X = _align_live_features(latest_rows, used_cols, medians.to_dict())
    latest_rows = latest_rows.copy()
    latest_rows["continuation_probability"] = final_model.predict_proba(live_X)[:, 1]
    latest_rows["continuation_prediction"] = final_model.predict(live_X)
    latest_rows = latest_rows.sort_values("continuation_probability", ascending=False)

    live_columns = [
        "symbol",
        "close_time",
        "close",
        "candle_return_pct",
        "body_pct",
        "volume_ratio_6h",
        "rsi_1h",
        "oi_change_pct_1h",
        "continuation_probability",
        "continuation_prediction",
    ]
    latest_rows[live_columns].head(HOURLY_REPORT_TOP_N).to_csv(
        REPORTS_DIR / "hourly_live_signals.csv",
        index=False,
    )
    latest_rows[live_columns].head(HOURLY_REPORT_TOP_N).to_json(
        REPORTS_DIR / "hourly_live_signals.json",
        orient="records",
        indent=2,
        date_format="iso",
    )

    report = {
        "status": "trained",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": "hourly_candle_pattern_xgboost",
        "continuation_window_hours": HOURLY_FORWARD_CANDLES,
        "continuation_threshold_pct": HOURLY_CONTINUATION_THRESHOLD_PCT,
        "n_samples": int(n_samples),
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "n_symbols": int(n_symbols),
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
        "label_distribution": y.value_counts().to_dict(),
        "top_features": top_features,
        "bucket_analysis": _bucket_analysis(clean),
        "classification_report": metrics["classification_report"],
        "top_live_signals": latest_rows[live_columns]
        .head(10)
        .to_dict(orient="records"),
    }

    joblib.dump(final_model, MODELS_DIR / f"hourly_model_{ts}.joblib")
    joblib.dump(final_model, MODELS_DIR / "latest_hourly_model.joblib")
    joblib.dump(used_cols, MODELS_DIR / "hourly_feature_columns.joblib")
    joblib.dump(medians.to_dict(), MODELS_DIR / "hourly_feature_medians.joblib")

    report_path = LOGS_DIR / f"hourly_train_report_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    with open(LOGS_DIR / "latest_hourly_train_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    history_path = LOGS_DIR / "hourly_training_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, default=str) + "\n")

    return report
