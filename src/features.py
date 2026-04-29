"""Feature engineering and labeling for the main scanner model — v2.1.

Diff vs v2.0:
  • Adds nine cg_* columns derived from Azalyst aggregates:
        cg_funding_oi_weighted_bps, cg_funding_spread_bps,
        cg_top_ls_ratio, cg_global_ls_ratio, cg_top_minus_global_ls,
        cg_taker_buy_sell_ratio,
        cg_liq_pull_up, cg_liq_pull_down, cg_liq_pull_ratio
  • If COINGLASS_API_KEY is unset, all cg_* values are NaN and trainer's
    median imputation handles them — model degrades gracefully to v2.0.
  • Original public surface (run_feature_pipeline, build_features,
    label_data, load_raw_scans) preserved.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_DIR, LABEL_HORIZON_MINUTES, LABEL_LOOKAHEAD_TOLERANCE_MINUTES,
    PRICE_RISE_THRESHOLD_PCT, RAW_DIR, RSI_OVERBOUGHT, RSI_OVERSOLD,
)

log = logging.getLogger("azalyst.features")


def load_raw_scans() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("scans_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    df["scan_time"] = pd.to_datetime(df["scan_time"], utc=True)
    return df.sort_values(["symbol", "scan_time"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    rsi_cols = [c for c in feat.columns if c.startswith("rsi_")]

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
            group["price_rolling_std_6"] = group["price_pct_change"].rolling(6, min_periods=1).std()
        if "oi_contracts" in group.columns:
            group["oi_pct_change"] = group["oi_contracts"].pct_change() * 100
        if "rsi_5m" in group.columns:
            group["rsi_5m_slope"] = group["rsi_5m"].diff()
        rolling_parts.append(group)

    out = pd.concat(rolling_parts, ignore_index=True)
    out = _enrich_with_azalyst(out)
    return out


def _enrich_with_azalyst(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach cg_* features per symbol. Cached per-symbol to avoid hitting
    rate limits when feature builder runs for thousands of rows.
    """
    if df.empty or "symbol" not in df.columns:
        return df

    # Initialize columns as NaN so downstream code is consistent
    for col in [
        "cg_funding_oi_weighted_bps", "cg_funding_spread_bps",
        "cg_top_ls_ratio", "cg_global_ls_ratio", "cg_top_minus_global_ls",
        "cg_taker_buy_sell_ratio",
        "cg_liq_pull_up", "cg_liq_pull_down", "cg_liq_pull_ratio",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    if not df.empty and "symbol" in df.columns:
        # No env var check needed — derived_data uses only free public APIs.
        try:
            from src.derived_data import DerivedDataClient
        except Exception as exc:
            log.debug("Derived data import failed: %s", exc)
            return df

        client = DerivedDataClient()

        cache: Dict[str, Dict[str, float]] = {}
        unique_symbols = df["symbol"].unique().tolist()
        # Cap to top-N per cycle (avoid hammering free public APIs)
        cap = int(os.getenv("FEATURES_DERIVED_CAP", os.getenv("FEATURES_COINGLASS_CAP", "20")))
        targeted = unique_symbols[:cap]

        for sym in targeted:
            base = sym[:-4] if sym.endswith("USDT") else sym
            row: Dict[str, float] = {}

            # Funding
            fund = client.funding_aggregated(base)
            if fund is not None:
                row["cg_funding_oi_weighted_bps"] = fund.oi_weighted_funding * 10000.0
                row["cg_funding_spread_bps"] = fund.spread_bps

            # Long/short
            ls_hist = client.longshort_history(base, "1h", limit=2)
            if ls_hist:
                latest = ls_hist[-1]
                if latest.top_account_ratio is not None:
                    row["cg_top_ls_ratio"] = latest.top_account_ratio
                if latest.global_account_ratio is not None:
                    row["cg_global_ls_ratio"] = latest.global_account_ratio
                if (latest.top_account_ratio is not None
                        and latest.global_account_ratio is not None):
                    row["cg_top_minus_global_ls"] = (latest.top_account_ratio
                                                     - latest.global_account_ratio)
                if latest.taker_buy_sell_ratio is not None:
                    row["cg_taker_buy_sell_ratio"] = latest.taker_buy_sell_ratio

            # Liquidation pull
            heat = client.liquidation_heatmap(base, "1d")
            if heat is not None and heat.last_price:
                ref = heat.last_price
                pull_up = sum(
                    lvl.notional_usdt / max(abs(lvl.price - ref) / ref * 100, 0.5)
                    for lvl in heat.levels_above(ref, 5.0)
                )
                pull_down = sum(
                    lvl.notional_usdt / max(abs(lvl.price - ref) / ref * 100, 0.5)
                    for lvl in heat.levels_below(ref, 5.0)
                )
                row["cg_liq_pull_up"] = pull_up
                row["cg_liq_pull_down"] = pull_down
                row["cg_liq_pull_ratio"] = (pull_up / max(pull_down, 1)) if pull_down > 0 else 0.0

            cache[sym] = row

        # Broadcast cached values to all rows of each symbol
        for sym, vals in cache.items():
            mask = df["symbol"] == sym
            for col, val in vals.items():
                df.loc[mask, col] = val

        log.info("Enriched %d / %d symbols with derived data (free APIs).",
                 len(cache), len(unique_symbols))
    return df


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """Time-horizon labeling. Unchanged from v2.0."""
    horizon = pd.Timedelta(minutes=LABEL_HORIZON_MINUTES)
    tolerance = pd.Timedelta(minutes=LABEL_LOOKAHEAD_TOLERANCE_MINUTES)
    parts = []
    for _, group in df.groupby("symbol", sort=False):
        group = group.sort_values("scan_time").reset_index(drop=True).copy()
        group["target_time"] = group["scan_time"] + horizon
        future_lookup = group[["scan_time", "price"]].rename(
            columns={"scan_time": "future_scan_time", "price": "future_price"})
        merged = pd.merge_asof(
            group.sort_values("target_time"),
            future_lookup.sort_values("future_scan_time"),
            left_on="target_time", right_on="future_scan_time",
            direction="forward", tolerance=tolerance,
        )
        merged = merged.sort_values("scan_time").reset_index(drop=True)
        merged["future_return_pct"] = (
            (merged["future_price"] - merged["price"]) / merged["price"] * 100
        )
        merged["label"] = np.where(
            merged["future_return_pct"].isna(), np.nan,
            (merged["future_return_pct"] >= PRICE_RISE_THRESHOLD_PCT).astype(int),
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def run_feature_pipeline() -> pd.DataFrame:
    raw = load_raw_scans()
    if raw.empty:
        return raw
    featured = build_features(raw)
    labeled = label_data(featured)
    ts = pd.Timestamp.now(tz="utc").strftime("%Y%m%d_%H%M%S")
    labeled.to_csv(FEATURES_DIR / f"features_{ts}.csv", index=False)
    labeled.to_csv(FEATURES_DIR / "latest_features.csv", index=False)
    return labeled
