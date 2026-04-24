"""
Data Collector v2.0 — Dynamically discovers ALL Bybit USDT perpetual futures,
filters by volume/OI, then pulls klines, RSI, and Open Interest.
All public endpoints, no API key needed.
"""
import re
import time
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from src.config import (
    BYBIT_V5_BASE, TIMEFRAMES, KLINE_LIMIT, RSI_PERIOD,
    MIN_VOLUME_24H_USDT, MIN_OI_USDT, EXCLUDED_SYMBOLS,
    MAX_SYMBOLS_PER_SCAN, REQUEST_DELAY, BATCH_DELAY
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoScanner/2.0"})
REQUEST_ERRORS: list[dict] = []


def _extract_status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return int(status_code)

    match = re.search(r"\b(\d{3})\b", str(error))
    if match:
        return int(match.group(1))
    return None


def _record_request_error(endpoint: str, error: Exception, **details) -> None:
    entry = {
        "endpoint": endpoint,
        "status_code": _extract_status_code(error),
        "message": str(error),
    }
    entry.update({key: value for key, value in details.items() if value is not None})
    REQUEST_ERRORS.append(entry)


def reset_request_errors() -> None:
    REQUEST_ERRORS.clear()


def get_request_error_summary(max_examples: int = 8) -> dict:
    status_counts = Counter(
        str(item["status_code"])
        for item in REQUEST_ERRORS
        if item.get("status_code") is not None
    )
    endpoint_counts = Counter(item["endpoint"] for item in REQUEST_ERRORS)

    return {
        "count": len(REQUEST_ERRORS),
        "status_counts": dict(status_counts),
        "endpoint_counts": dict(endpoint_counts),
        "examples": REQUEST_ERRORS[:max_examples],
    }


# ──────────────────────────────────────────────────────────
#  Dynamic symbol discovery — ALL Bybit USDT perps
# ──────────────────────────────────────────────────────────
def fetch_all_futures_symbols() -> List[str]:
    """
    Fetch ALL active USDT-margined perpetual futures pairs from Bybit.
    Returns a sorted list like ['1000BONKUSDT', 'AAVEUSDT', 'BTCUSDT', ...].
    """
    url = f"{BYBIT_V5_BASE}/market/instruments-info"
    params = {"category": "linear", "limit": 1000}
    try:
        r = SESSION.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        symbols = []
        for s in data.get("result", {}).get("list", []):
            if (s.get("status") == "Trading"
                    and s.get("contractType") == "LinearPerpetual"
                    and s.get("quoteCoin") == "USDT"
                    and s["symbol"] not in EXCLUDED_SYMBOLS):
                symbols.append(s["symbol"])
        return sorted(symbols)
    except Exception as e:
        _record_request_error("exchange_info", e)
        print(f"  [ERR] Failed to fetch exchange info: {e}")
        return []


def fetch_all_tickers_bulk() -> dict:
    """
    Fetch 24h ticker data for ALL futures pairs in ONE API call.
    Returns dict: { 'BTCUSDT': {price, volume_24h, ...}, ... }
    """
    url = f"{BYBIT_V5_BASE}/market/tickers"
    params = {"category": "linear"}
    try:
        r = SESSION.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        tickers = {}
        for d in data.get("result", {}).get("list", []):
            tickers[d["symbol"]] = {
                "price": float(d["lastPrice"]),
                "price_change_pct_24h": float(d["price24hPcnt"]),
                "high_24h": float(d["highPrice24h"]),
                "low_24h": float(d["lowPrice24h"]),
                "volume_24h": float(d["turnover24h"]),
            }
        return tickers
    except Exception as e:
        _record_request_error("bulk_ticker", e)
        print(f"  [ERR] bulk ticker fetch: {e}")
        return {}


def filter_symbols_by_volume(symbols: List[str], tickers: dict) -> List[str]:
    """
    Filter symbols by minimum 24h volume. Returns sorted by volume desc.
    """
    valid = []
    for sym in symbols:
        t = tickers.get(sym)
        if t and t["volume_24h"] >= MIN_VOLUME_24H_USDT:
            valid.append((sym, t["volume_24h"]))

    valid.sort(key=lambda x: x[1], reverse=True)
    filtered = [s[0] for s in valid]

    if MAX_SYMBOLS_PER_SCAN > 0:
        filtered = filtered[:MAX_SYMBOLS_PER_SCAN]

    return filtered


def get_active_symbols() -> tuple:
    """
    Master function: fetch all Bybit USDT perps, filter by volume.
    Returns (filtered_symbols_list, bulk_tickers_dict).

    The bulk tickers are returned so scanner.py can pass preloaded
    ticker data into scan_symbol() and avoid redundant per-symbol calls.
    """
    all_symbols = fetch_all_futures_symbols()
    if not all_symbols:
        print("  [WARN] Exchange info unavailable, using fallback list")
        return _FALLBACK_SYMBOLS, {}

    tickers = fetch_all_tickers_bulk()
    if not tickers:
        return all_symbols, {}

    filtered = filter_symbols_by_volume(all_symbols, tickers)
    return filtered, tickers


# Fallback if Bybit exchange info endpoint fails
_FALLBACK_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
    "ATOMUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "NEARUSDT", "FILUSDT", "SUIUSDT", "PEPEUSDT", "TRXUSDT",
]


# ──────────────────────────────────────────────────────────
#  Klines (candlestick data)
# ──────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> Optional[pd.DataFrame]:
    """Fetch futures klines from Bybit."""
    # Map common intervals to Bybit format
    interval_map = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D", "1w": "W", "1M": "M"
    }
    bybit_interval = interval_map.get(interval, interval)
    
    url = f"{BYBIT_V5_BASE}/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": bybit_interval, "limit": limit}
    try:
        r = SESSION.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        klines_data = result.get("list", [])
        if not klines_data:
            return None
        df = pd.DataFrame(klines_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "turnover", "trades"
        ])
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.rename(columns={"turnover": "quote_volume"}, inplace=True)
        df["close_time"] = df["open_time"]
        return df
    except Exception as e:
        _record_request_error("klines", e, symbol=symbol, interval=interval)
        print(f"  [ERR] klines {symbol} {interval}: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  RSI calculation (Wilder's smoothing)
# ──────────────────────────────────────────────────────────
def compute_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI using exponential (Wilder) smoothing."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_rsi_for_symbol(symbol: str, interval: str) -> Optional[float]:
    """Return the *latest* RSI value for a symbol on a given timeframe."""
    df = fetch_klines(symbol, interval)
    if df is None or len(df) < RSI_PERIOD + 1:
        return None
    rsi_series = compute_rsi(df["close"])
    return round(rsi_series.iloc[-1], 2)


# ──────────────────────────────────────────────────────────
#  Multi-timeframe RSI snapshot
# ──────────────────────────────────────────────────────────
def get_multi_tf_rsi(symbol: str) -> dict:
    """Return dict  { 'rsi_1m': 45.2, 'rsi_5m': 62.1, ... }  for all configured TFs."""
    result = {}
    for label, interval in TIMEFRAMES.items():
        rsi_val = get_rsi_for_symbol(symbol, interval)
        result[f"rsi_{label}"] = rsi_val
        time.sleep(REQUEST_DELAY)
    return result


# ──────────────────────────────────────────────────────────
#  Open Interest
# ──────────────────────────────────────────────────────────
def fetch_open_interest(symbol: str) -> Optional[dict]:
    """Current open interest (contracts + value)."""
    url = f"{BYBIT_V5_BASE}/market/open-interest"
    params = {"category": "linear", "symbol": symbol}
    try:
        r = SESSION.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        list_data = result.get("list", [])
        if not list_data:
            return None
        oi_data = list_data[0]
        return {
            "oi_contracts": float(oi_data["openInterest"]),
            "oi_time": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        _record_request_error("open_interest", e, symbol=symbol)
        print(f"  [ERR] OI {symbol}: {e}")
        return None


def fetch_oi_history(symbol: str, period: str = "5m", limit: int = 30) -> Optional[pd.DataFrame]:
    """Open interest statistics (with value in USDT)."""
    # Map period to Bybit interval format
    period_map = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "4h": "240", "1d": "D"
    }
    bybit_period = period_map.get(period, period)
    
    url = f"{BYBIT_V5_BASE}/market/open-interest"
    params = {"category": "linear", "symbol": symbol, "intervalTime": bybit_period, "limit": limit}
    try:
        r = SESSION.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        list_data = result.get("list", [])
        if not list_data:
            return None
        df = pd.DataFrame(list_data)
        df["openInterest"] = df["openInterest"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # Create sumOpenInterestValue as openInterest * price approximation
        df["sumOpenInterestValue"] = df["openInterest"]
        df.rename(columns={"openInterest": "sumOpenInterest"}, inplace=True)
        return df
    except Exception as e:
        _record_request_error("open_interest_history", e, symbol=symbol, period=period)
        print(f"  [ERR] OI history {symbol}: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  Price snapshot (single symbol fallback)
# ──────────────────────────────────────────────────────────
def fetch_ticker(symbol: str) -> Optional[dict]:
    """24h ticker with price + volume."""
    url = f"{BYBIT_V5_BASE}/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    try:
        r = SESSION.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        list_data = result.get("list", [])
        if not list_data:
            return None
        d = list_data[0]
        return {
            "price": float(d["lastPrice"]),
            "price_change_pct_24h": float(d["price24hPcnt"]),
            "high_24h": float(d["highPrice24h"]),
            "low_24h": float(d["lowPrice24h"]),
            "volume_24h": float(d["turnover24h"]),
        }
    except Exception as e:
        _record_request_error("ticker", e, symbol=symbol)
        print(f"  [ERR] ticker {symbol}: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  Funding rate
# ──────────────────────────────────────────────────────────
def fetch_funding_rate(symbol: str) -> Optional[float]:
    """Latest funding rate."""
    url = f"{BYBIT_V5_BASE}/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    try:
        r = SESSION.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        list_data = result.get("list", [])
        if not list_data:
            return None
        d = list_data[0]
        # Bybit provides fundingRate in ticker endpoint for linear perps
        funding_rate = d.get("fundingRate")
        if funding_rate is not None:
            return float(funding_rate)
        return None
    except Exception as e:
        _record_request_error("funding_rate", e, symbol=symbol)
        print(f"  [ERR] funding {symbol}: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  Full scan for one symbol
# ──────────────────────────────────────────────────────────
def scan_symbol(
    symbol: str,
    preloaded_ticker: dict = None,
    scan_time: Optional[datetime] = None,
) -> Optional[dict]:
    """
    Collect ALL data points for a single symbol.
    If preloaded_ticker is provided (from bulk fetch), skip individual ticker call.
    """
    effective_scan_time = scan_time or datetime.now(timezone.utc)
    row = {"symbol": symbol, "scan_time": effective_scan_time.isoformat()}

    # Use preloaded ticker data if available (saves API calls)
    if preloaded_ticker:
        row.update(preloaded_ticker)
    else:
        ticker = fetch_ticker(symbol)
        if ticker is None:
            return None
        row.update(ticker)

    # Multi-TF RSI
    rsi_data = get_multi_tf_rsi(symbol)
    row.update(rsi_data)

    # Open Interest
    oi = fetch_open_interest(symbol)
    if oi:
        row["oi_contracts"] = oi["oi_contracts"]

    # OI history for change detection
    oi_hist = fetch_oi_history(symbol, period="5m", limit=12)
    if oi_hist is not None and len(oi_hist) >= 2:
        row["oi_value_now"] = oi_hist["sumOpenInterestValue"].iloc[-1]
        row["oi_value_1h_ago"] = oi_hist["sumOpenInterestValue"].iloc[0]
        row["oi_change_pct_1h"] = round(
            (row["oi_value_now"] - row["oi_value_1h_ago"]) / row["oi_value_1h_ago"] * 100, 3
        )

    # Funding rate
    funding = fetch_funding_rate(symbol)
    if funding is not None:
        row["funding_rate"] = funding

    return row
