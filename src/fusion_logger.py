"""
src/fusion_logger.py — Logs fused signals with outcomes for dynamic weight learning.
"""

import json
import logging
from datetime import timedelta
from typing import Dict, List

import pandas as pd

from src.config import FEATURES_DIR, LOGS_DIR

log = logging.getLogger("azalyst.fusion_logger")

def log_fused_outcomes(previous_fused_signals: List[Dict]) -> None:
    if not previous_fused_signals:
        return
    feat_path = FEATURES_DIR / "latest_features.csv"
    if not feat_path.exists():
        log.warning("No latest_features.csv found; skipping fusion outcome logging.")
        return
    try:
        features = pd.read_csv(feat_path)
        if "scan_time" in features.columns:
            features["scan_time"] = pd.to_datetime(features["scan_time"], utc=True)
    except Exception as e:
        log.error("Failed to read features: %s", e)
        return
    history_path = LOGS_DIR / "fused_history.jsonl"
    matched = 0
    for signal in previous_fused_signals:
        symbol = signal.get("symbol")
        timestamp_str = signal.get("generated_at") or signal.get("timestamp")
        if not symbol or not timestamp_str:
            continue
        try:
            sig_time = pd.to_datetime(timestamp_str, utc=True)
        except Exception:
            continue
        mask = (features["symbol"] == symbol) & ((features["scan_time"] - sig_time).abs() <= timedelta(minutes=5))
        matched_rows = features[mask]
        if matched_rows.empty:
            continue
        matched_rows = matched_rows.copy()
        matched_rows["time_diff"] = (matched_rows["scan_time"] - sig_time).abs()
        row = matched_rows.loc[matched_rows["time_diff"].idxmin()]
        future_return = row.get("future_return_pct")
        if future_return is None or pd.isna(future_return):
            continue
        outcome = 1 if float(future_return) >= 0.5 else 0
        engines = ["liq_proximity", "ml_main", "ls_extreme", "funding_extreme", "basis", "oi_delta"]
        strengths = {}
        for card in signal.get("reasons") or []:
            engine_name = card.get("engine")
            if engine_name in engines:
                strengths[engine_name] = card.get("strength", 0.0)
        if "ml_main" not in strengths and signal.get("ml_probability") is not None:
            ml_prob = signal["ml_probability"]
            strengths["ml_main"] = (ml_prob - 0.5) * 200
        for e in engines:
            strengths.setdefault(e, 0.0)
        entry = {"symbol": symbol, "direction": signal.get("direction"), "consensus_tier": signal.get("consensus_tier"), "timestamp": timestamp_str, "outcome": outcome, **strengths}
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        matched += 1
    log.info("Fusion outcome logging: matched %d/%d signals.", matched, len(previous_fused_signals))
