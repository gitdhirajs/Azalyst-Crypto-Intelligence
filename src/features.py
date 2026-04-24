"""Feature engineering and labeling for the main scanner model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_DIR,
    LABEL_HORIZON_MINUTES,
    LABEL_LOOKAHEAD_TOLERANCE_MINUTES,
    PRICE_RISE_THRESHOLD_PCT,
    RAW_DIR,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
)


def load_raw_scans() -> pd.DataFrame:
    """Load all raw scan CSVs into one ordered DataFrame."""
    files = sorted(RAW_DIR.glob("scans_*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = [pd.read_csv(file_path) for file_path in files]
    df = pd.concat(dfs, ignore_index=True)
    df["scan_time"] = pd.to_datetime(df["scan_time"], utc=True)
    return df.sort_values(["symbol", "scan_time"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from raw scan rows."""
    feat = df.copy()
    rsi_cols = [col for col in feat.columns if col.startswith("rsi_")]

    if rsi_cols:
        feat["rsi_mean"] = feat[rsi_cols].mean(axis=1)
        feat["rsi_std"] = feat[rsi_cols].std(axis=1)
        feat["rsi_min"] = feat[rsi_cols].min(axis=1)
        feat["rsi_max"] = feat[rsi_cols].max(axis=1)
        feat["rsi_range"] = feat["rsi_max"] - feat["rsi_min"]

        if {"rsi_5m", "rsi_4h"}.issubset(feat.columns):
            feat["rsi_divergence_5m_4h"] = feat["rsi_5m"] - feat["rsi_4h"]
        if {"rsi_15m", "rsi_1h"}.issubset(feat.columns):
            feat["rsi_divergence_15m_1h"] = feat["rsi_15m"] - feat["rsi_1h"]
        if {"rsi_1m", "rsi_1h"}.issubset(feat.columns):
            feat["rsi_divergence_1m_1h"] = feat["rsi_1m"] - feat["rsi_1h"]

        feat["n_tf_oversold"] = (feat[rsi_cols] < RSI_OVERSOLD).sum(axis=1)
        feat["n_tf_overbought"] = (feat[rsi_cols] > RSI_OVERBOUGHT).sum(axis=1)

    if "oi_change_pct_1h" in feat.columns:
        feat["oi_rising"] = (feat["oi_change_pct_1h"] > 0).astype(int)
        feat["oi_spike"] = (feat["oi_change_pct_1h"].abs() > 3).astype(int)

    if "price_change_pct_24h" in feat.columns:
        feat["price_momentum_24h"] = feat["price_change_pct_24h"]

    rolling_parts = []
    for _, group in feat.groupby("symbol", sort=False):
        group = group.sort_values("scan_time").copy()
        if "price" in group.columns:
            group["price_pct_change"] = group["price"].pct_change() * 100
            group["price_rolling_std_6"] = (
                group["price_pct_change"].rolling(6, min_periods=1).std()
            )
        if "oi_contracts" in group.columns:
            group["oi_pct_change"] = group["oi_contracts"].pct_change() * 100
        if "rsi_5m" in group.columns:
            group["rsi_5m_slope"] = group["rsi_5m"].diff()
        rolling_parts.append(group)

    return pd.concat(rolling_parts, ignore_index=True)


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """Label rows using an actual time horizon instead of row count."""
    horizon = pd.Timedelta(minutes=LABEL_HORIZON_MINUTES)
    tolerance = pd.Timedelta(minutes=LABEL_LOOKAHEAD_TOLERANCE_MINUTES)
    labeled_parts = []

    for _, group in df.groupby("symbol", sort=False):
        group = group.sort_values("scan_time").reset_index(drop=True).copy()
        group["target_time"] = group["scan_time"] + horizon

        future_lookup = group[["scan_time", "price"]].rename(
            columns={
                "scan_time": "future_scan_time",
                "price": "future_price",
            }
        )

        merged = pd.merge_asof(
            group.sort_values("target_time"),
            future_lookup.sort_values("future_scan_time"),
            left_on="target_time",
            right_on="future_scan_time",
            direction="forward",
            tolerance=tolerance,
        )

        merged = merged.sort_values("scan_time").reset_index(drop=True)
        merged["future_return_pct"] = (
            (merged["future_price"] - merged["price"]) / merged["price"] * 100
        )
        merged["label"] = np.where(
            merged["future_return_pct"].isna(),
            np.nan,
            (merged["future_return_pct"] >= PRICE_RISE_THRESHOLD_PCT).astype(int),
        )
        labeled_parts.append(merged)

    return pd.concat(labeled_parts, ignore_index=True)


def run_feature_pipeline() -> pd.DataFrame:
    """Run the full feature pipeline and persist the result."""
    raw = load_raw_scans()
    if raw.empty:
        return raw

    featured = build_features(raw)
    labeled = label_data(featured)

    ts = pd.Timestamp.now(tz="utc").strftime("%Y%m%d_%H%M%S")
    out_path = FEATURES_DIR / f"features_{ts}.csv"
    labeled.to_csv(out_path, index=False)
    labeled.to_csv(FEATURES_DIR / "latest_features.csv", index=False)
    return labeled
