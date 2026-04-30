"""
src/collector.py — v2.1 GitHub-Actions-friendly collector.

Why this exists:
  Binance Futures returns HTTP 451 to GitHub-hosted runner IPs in many
  regions. Bybit does the same. The previous v2.0 scanner relied on
  Binance public endpoints and consequently never collected real data
  on Actions — only when run locally.

Solution:
  Use exchanges that DO accept GitHub Actions IPs:
    1. KuCoin Futures (PRIMARY)  — XBT/ETH/etc, USDT-margined perps
    2. Bitget Futures (BACKUP)   — clean v2 API, full OI / funding / L/S
    3. OKX SWAP (TERTIARY)       — also unblocked on AWS/GCP IPs

  Binance + Bybit kept as optional fallback, only used when explicitly
  enabled (e.g. local development), via env var ALLOW_BLOCKED_EXCHANGES=1.

Contract / output schema is IDENTICAL to v2.0 — every other module
(features.py, trainer.py, pipeline.py, hourly_trainer.py) works
unchanged. Only the data source changes.

KuCoin symbol quirks normalized for you:
  • Bitcoin futures are XBTUSDTM on KuCoin (not BTCUSDT)
  • All KuCoin USDT perps have "M" suffix (perpetual marker)
  • The collector exposes the canonical 'BTCUSDT' name everywhere
    so downstream code is unchanged
"""

import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.config import (
    BATCH_DELAY, EXCLUDED_SYMBOLS, KLINE_LIMIT, MAX_SYMBOLS_PER_SCAN,
    MIN_VOLUME_24H_USDT, REQUEST_DELAY, RSI_PERIOD, TIMEFRAMES,
)


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AzalystCryptoScanner/2.1"})

KUCOIN_FUTURES = "https://api-futures.kucoin.com"
BITGET_BASE = "https://api.bitget.com"
OKX_BASE = "https://www.okx.com"

# Optional fallback (almost always 451 from GitHub runners)
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
ALLOW_BLOCKED = os.getenv("ALLOW_BLOCKED_EXCHANGES", "").strip() in ("1", "true", "yes")

KUCOIN_GRANULARITY = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
BITGET_GRANULARITY = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}
OKX_BAR = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}

REQUEST_ERRORS: list = []

class CircuitBreaker:
    def __init__(self, fail_threshold=5, cooldown=120):
        self.fails = {}
        self.threshold = fail_threshold
        self.cooldown = cooldown

    def is_open(self, endpoint):
        if endpoint not in self.fails: return False
        count, last = self.fails[endpoint]
        if count >= self.threshold and time.time() - last < self.cooldown:
            return True
        if time.time() - last > self.cooldown:
            self.fails[endpoint] = (0, time.time())
        return False

    def record_fail(self, endpoint):
        count, _ = self.fails.get(endpoint, (0, 0))
        self.fails[endpoint] = (count + 1, time.time())

cb_breaker = CircuitBreaker()

def _request_with_cb(url, params=None, timeout=10, cache_key=None, cache_ttl=0):
    """HTTP GET wrapper with Circuit Breaker and basic TTL caching."""
    if cb_breaker.is_open(url):
        return None
    
    # TTL Cache
    if cache_key:
        if not hasattr(_request_with_cb, "_cache"): _request_with_cb._cache = {}
        cached = _request_with_cb._cache.get(cache_key)
        if cached and time.time() - cached["ts"] < cache_ttl:
            return cached["data"]

    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        
        if cache_key and data:
            _request_with_cb._cache[cache_key] = {"ts": time.time(), "data": data}
        return data
    except Exception as e:
        cb_breaker.record_fail(url)
        _record_request_error(url, e)
        return None


def _extract_status_code(error):
    response = getattr(error, "response", None)
    sc = getattr(response, "status_code", None)
    if sc is not None:
        return int(sc)
    m = re.search(r"\b(\d{3})\b", str(error))
    return int(m.group(1)) if m else None


def _record_request_error(endpoint, error, **details):
    entry = {"endpoint": endpoint, "status_code": _extract_status_code(error), "message": str(error)}
    entry.update({k: v for k, v in details.items() if v is not None})
    REQUEST_ERRORS.append(entry)


def reset_request_errors():
    REQUEST_ERRORS.clear()


def get_request_error_summary(max_examples=8):
    status_counts = Counter(str(i["status_code"]) for i in REQUEST_ERRORS if i.get("status_code") is not None)
    endpoint_counts = Counter(i["endpoint"] for i in REQUEST_ERRORS)
    return {
        "count": len(REQUEST_ERRORS),
        "status_counts": dict(status_counts),
        "endpoint_counts": dict(endpoint_counts),
        "examples": REQUEST_ERRORS[:max_examples],
    }


# ──────────────────────────────────────────────────────────────────────────
# Symbol naming helpers (canonical = "BTCUSDT" form everywhere downstream)
# ──────────────────────────────────────────────────────────────────────────
def to_kucoin_symbol(canonical: str) -> str:
    """BTCUSDT → XBTUSDTM, ETHUSDT → ETHUSDTM."""
    if not canonical.endswith("USDT"):
        return canonical
    base = canonical[:-4]
    if base == "BTC":
        base = "XBT"
    return f"{base}USDTM"


def from_kucoin_symbol(kucoin_sym: str) -> str:
    """XBTUSDTM → BTCUSDT."""
    if kucoin_sym.endswith("USDTM"):
        base = kucoin_sym[:-5]
        if base == "XBT":
            base = "BTC"
        return f"{base}USDT"
    return kucoin_sym


def to_bitget_symbol(canonical: str) -> str:
    return canonical


def to_okx_inst(canonical: str) -> str:
    """BTCUSDT → BTC-USDT-SWAP."""
    if canonical.endswith("USDT"):
        return f"{canonical[:-4]}-USDT-SWAP"
    return canonical


# ──────────────────────────────────────────────────────────────────────────
# Symbol discovery — KuCoin first (gives prices+OI+funding in one shot),
# Bitget fallback, then a small hardcoded list.
# ──────────────────────────────────────────────────────────────────────────
def fetch_kucoin_active_contracts() -> Tuple[List[str], Dict[str, dict]]:
    url = f"{KUCOIN_FUTURES}/api/v1/contracts/active"
    try:
        r = SESSION.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get("data") or []
        canonical_syms: List[str] = []
        tickers: Dict[str, dict] = {}

        for c in data:
            quote = (c.get("quoteCurrency") or "").upper()
            if quote != "USDT":
                continue
            if (c.get("type") or "").upper() != "FFWCSX":
                continue
            if c.get("status") != "Open":
                continue
            kc_sym = c.get("symbol")
            if not kc_sym:
                continue
            canonical = from_kucoin_symbol(kc_sym)
            if canonical in EXCLUDED_SYMBOLS:
                continue
            try:
                mark = float(c.get("markPrice") or c.get("lastTradePrice") or 0)
                turnover_24h = float(c.get("turnoverOf24h") or 0)
                vol_change = float(c.get("priceChgPct") or 0) * 100
                high = float(c.get("highPrice") or 0)
                low = float(c.get("lowPrice") or 0)
                oi_contracts = float(c.get("openInterest") or 0)
                multiplier = float(c.get("multiplier") or 1)
                funding = float(c.get("fundingFeeRate") or 0)
            except (TypeError, ValueError):
                continue
            if turnover_24h < MIN_VOLUME_24H_USDT:
                continue
            tickers[canonical] = {
                "price": mark,
                "price_change_pct_24h": vol_change,
                "high_24h": high,
                "low_24h": low,
                "volume_24h": turnover_24h,
                "_kucoin_symbol": kc_sym,
                "_oi_contracts": oi_contracts,
                "_oi_value_usdt": oi_contracts * abs(multiplier) * mark,
                "_funding_rate": funding,
                "_multiplier": multiplier,
            }
            canonical_syms.append(canonical)

        canonical_syms.sort(key=lambda s: -tickers[s]["volume_24h"])
        if MAX_SYMBOLS_PER_SCAN > 0:
            canonical_syms = canonical_syms[:MAX_SYMBOLS_PER_SCAN]
        return canonical_syms, tickers
    except Exception as e:
        _record_request_error("kucoin_active_contracts", e)
        return [], {}


def fetch_bitget_tickers_bulk() -> Tuple[List[str], Dict[str, dict]]:
    url = f"{BITGET_BASE}/api/v2/mix/market/tickers"
    try:
        r = SESSION.get(url, params={"productType": "USDT-FUTURES"}, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code", "")) != "00000":
            return [], {}
        rows = payload.get("data") or []
        canonical_syms: List[str] = []
        tickers: Dict[str, dict] = {}

        for row in rows:
            sym = row.get("symbol")
            if not sym or not sym.endswith("USDT"):
                continue
            if sym in EXCLUDED_SYMBOLS:
                continue
            try:
                last = float(row.get("lastPr") or 0)
                chg_pct = float(row.get("change24h") or 0) * 100
                high = float(row.get("high24h") or 0)
                low = float(row.get("low24h") or 0)
                quote_vol = float(row.get("quoteVolume") or row.get("usdtVolume") or 0)
                if quote_vol == 0 and row.get("baseVolume") and last > 0:
                    quote_vol = float(row["baseVolume"]) * last
                funding = float(row.get("fundingRate") or 0)
                oi_usdt = float(row.get("holdingAmount") or 0) * last
            except (TypeError, ValueError):
                continue
            if quote_vol < MIN_VOLUME_24H_USDT:
                continue
            tickers[sym] = {
                "price": last,
                "price_change_pct_24h": chg_pct,
                "high_24h": high,
                "low_24h": low,
                "volume_24h": quote_vol,
                "_funding_rate": funding,
                "_oi_value_usdt": oi_usdt,
            }
            canonical_syms.append(sym)

        canonical_syms.sort(key=lambda s: -tickers[s]["volume_24h"])
        if MAX_SYMBOLS_PER_SCAN > 0:
            canonical_syms = canonical_syms[:MAX_SYMBOLS_PER_SCAN]
        return canonical_syms, tickers
    except Exception as e:
        _record_request_error("bitget_tickers", e)
        return [], {}


def get_active_symbols() -> Tuple[List[str], Dict[str, dict]]:
    syms, tickers = fetch_kucoin_active_contracts()
    if syms:
        return syms, tickers
    syms, tickers = fetch_bitget_tickers_bulk()
    if syms:
        return syms, tickers
    if ALLOW_BLOCKED:
        return _binance_fallback_discovery()
    return _FALLBACK_SYMBOLS, {}


_FALLBACK_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "ATOMUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "NEARUSDT", "FILUSDT", "SUIUSDT", "PEPEUSDT",
]


def _binance_fallback_discovery() -> Tuple[List[str], Dict[str, dict]]:
    try:
        ei = SESSION.get(f"{BINANCE_FUTURES_BASE}/fapi/v1/exchangeInfo", timeout=15)
        ei.raise_for_status()
        symbols = [
            s["symbol"] for s in ei.json().get("symbols", [])
            if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT" and s["symbol"] not in EXCLUDED_SYMBOLS
        ]
        if not symbols:
            return _FALLBACK_SYMBOLS, {}
        tk = SESSION.get(f"{BINANCE_FUTURES_BASE}/fapi/v1/ticker/24hr", timeout=15)
        tk.raise_for_status()
        tickers = {}
        for d in tk.json():
            if d["symbol"] in symbols and float(d["quoteVolume"]) >= MIN_VOLUME_24H_USDT:
                tickers[d["symbol"]] = {
                    "price": float(d["lastPrice"]),
                    "price_change_pct_24h": float(d["priceChangePercent"]),
                    "high_24h": float(d["highPrice"]),
                    "low_24h": float(d["lowPrice"]),
                    "volume_24h": float(d["quoteVolume"]),
                }
        out = [s for s in symbols if s in tickers]
        out.sort(key=lambda s: -tickers[s]["volume_24h"])
        if MAX_SYMBOLS_PER_SCAN > 0:
            out = out[:MAX_SYMBOLS_PER_SCAN]
        return out, tickers
    except Exception as e:
        _record_request_error("binance_fallback", e)
        return _FALLBACK_SYMBOLS, {}


# ──────────────────────────────────────────────────────────────────────────
# Klines — KuCoin → Bitget → OKX → (optionally Binance)
# ──────────────────────────────────────────────────────────────────────────
def fetch_klines(
    symbol: str,
    interval: str,
    limit: int = KLINE_LIMIT,
    provider: str | None = None,
) -> Optional[pd.DataFrame]:
    """Fetch klines while preserving compatibility with provider-aware callers."""
    fetchers = {
        "kucoin": _fetch_klines_kucoin,
        "bitget": _fetch_klines_bitget,
        "okx": _fetch_klines_okx,
    }

    ordered = []
    if provider in fetchers:
        ordered.append(provider)
    ordered.extend(name for name in ("kucoin", "bitget", "okx") if name not in ordered)

    for name in ordered:
        df = fetchers[name](symbol, interval, limit)
        if df is not None and not df.empty:
            return df

    if ALLOW_BLOCKED:
        return _fetch_klines_binance(symbol, interval, limit)
    return None


def _fetch_klines_kucoin(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    granularity = KUCOIN_GRANULARITY.get(interval)
    if not granularity:
        return None
    kc_sym = to_kucoin_symbol(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - granularity * 60 * 1000 * (limit + 5)
    try:
        data = _request_with_cb(f"{KUCOIN_FUTURES}/api/v1/kline/query", params={
            "symbol": kc_sym, "granularity": granularity,
            "from": start_ms, "to": end_ms,
        }, timeout=10, cache_key=f"klines_ku_{kc_sym}_{granularity}", cache_ttl=30)
        
        rows = data.get("data") if data else []
        if not rows:
            return None
        # KuCoin returns [timestamp, open, close, high, low, volume, turnover]
        # Previous versions had 6 columns; v2.1 handles the new turnover column.
        cols = ["open_time", "open", "close", "high", "low", "volume", "turnover"]
        df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        df["close_time"] = df["open_time"] + pd.Timedelta(minutes=granularity) - pd.Timedelta(milliseconds=1)
        df["quote_volume"] = df["volume"] * df["close"]
        df["_source"] = "kucoin"
        return df.tail(limit).reset_index(drop=True)
    except Exception as e:
        _record_request_error("kucoin_klines", e, symbol=symbol, interval=interval)
        return None


def _fetch_klines_bitget(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    granularity = BITGET_GRANULARITY.get(interval)
    if not granularity:
        return None
    try:
        payload = _request_with_cb(f"{BITGET_BASE}/api/v2/mix/market/candles", params={
            "symbol": symbol, "productType": "USDT-FUTURES",
            "granularity": granularity, "limit": str(min(limit, 200)),
        }, timeout=10, cache_key=f"klines_bg_{symbol}_{granularity}", cache_ttl=30)
        
        if not payload or str(payload.get("code", "")) != "00000":
            return None
        rows = payload.get("data") or []
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume", "quote_volume",
        ])
        for c in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        df["close_time"] = df["open_time"] + _interval_delta(interval) - pd.Timedelta(milliseconds=1)
        df["_source"] = "bitget"
        return df.tail(limit).reset_index(drop=True)
    except Exception as e:
        _record_request_error("bitget_klines", e, symbol=symbol, interval=interval)
        return None


def _fetch_klines_okx(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    bar = OKX_BAR.get(interval)
    if not bar:
        return None
    inst = to_okx_inst(symbol)
    try:
        payload = _request_with_cb(f"{OKX_BASE}/api/v5/market/candles",
                        params={"instId": inst, "bar": bar, "limit": str(min(limit, 300))}, 
                        timeout=10, cache_key=f"klines_okx_{inst}_{bar}", cache_ttl=30)
        if not payload or str(payload.get("code", "")) != "0":
            return None
        rows = list(reversed(payload.get("data") or []))
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "vol_ccy", "quote_volume", "confirm",
        ])
        for c in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        df["close_time"] = df["open_time"] + _interval_delta(interval) - pd.Timedelta(milliseconds=1)
        df["_source"] = "okx"
        return df[["open_time", "open", "high", "low", "close", "volume",
                   "close_time", "quote_volume", "_source"]].tail(limit).reset_index(drop=True)
    except Exception as e:
        _record_request_error("okx_klines", e, symbol=symbol, interval=interval)
        return None


def _fetch_klines_binance(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        r = SESSION.get(f"{BINANCE_FUTURES_BASE}/fapi/v1/klines",
                        params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        for c in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df["_source"] = "binance"
        return df
    except Exception as e:
        _record_request_error("binance_klines", e, symbol=symbol, interval=interval)
        return None


def _interval_delta(interval: str) -> pd.Timedelta:
    return {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }.get(interval, pd.Timedelta(hours=1))


# ──────────────────────────────────────────────────────────────────────────
# RSI (unchanged)
# ──────────────────────────────────────────────────────────────────────────
def compute_rsi(closes, period=RSI_PERIOD):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def get_rsi_for_symbol(symbol, interval):
    df = fetch_klines(symbol, interval)
    if df is None or len(df) < RSI_PERIOD + 1:
        return None
    return round(compute_rsi(df["close"]).iloc[-1], 2)


def get_multi_tf_rsi(symbol):
    out = {}
    for label, interval in TIMEFRAMES.items():
        out[f"rsi_{label}"] = get_rsi_for_symbol(symbol, interval)
        time.sleep(REQUEST_DELAY)
    return out


# ──────────────────────────────────────────────────────────────────────────
# OI snapshot — KuCoin direct, Bitget fallback
# ──────────────────────────────────────────────────────────────────────────
def fetch_open_interest(symbol: str) -> Optional[dict]:
    kc_sym = to_kucoin_symbol(symbol)
    try:
        r = SESSION.get(f"{KUCOIN_FUTURES}/api/v1/contracts/{kc_sym}", timeout=10)
        r.raise_for_status()
        d = (r.json() or {}).get("data") or {}
        oi = d.get("openInterest")
        if oi is not None:
            return {"oi_contracts": float(oi),
                    "oi_time": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        _record_request_error("kucoin_oi", e, symbol=symbol)

    try:
        r = SESSION.get(f"{BITGET_BASE}/api/v2/mix/market/open-interest",
                        params={"symbol": symbol, "productType": "USDT-FUTURES"}, timeout=10)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code", "")) == "00000":
            d = payload.get("data") or {}
            entries = d.get("openInterestList") or []
            if entries:
                return {"oi_contracts": float(entries[0].get("size") or 0),
                        "oi_time": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        _record_request_error("bitget_oi", e, symbol=symbol)

    return None


def fetch_oi_history(
    symbol: str,
    period: str = "5m",
    limit: int = 30,
    provider: str | None = None,
) -> Optional[pd.DataFrame]:
    """
    OI history with same column shape as v2.0 features expect.
    Sourced from OKX rubik (works on Actions, returns USD-valued OI directly).
    """
    _ = provider  # kept for compatibility with hourly-trainer call sites
    inst = to_okx_inst(symbol)
    url = f"{OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-volume"
    try:
        r = SESSION.get(url, params={
            "instId": inst, "period": _okx_period(period), "limit": str(min(limit, 100)),
        }, timeout=10)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code", "")) != "0":
            return None
        rows = payload.get("data") or []
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["timestamp", "sumOpenInterestValue", "vol_usd"])
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
        df["sumOpenInterest"] = df["sumOpenInterestValue"]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        _record_request_error("okx_oi_history", e, symbol=symbol, period=period)
        return None


def _okx_period(p: str) -> str:
    return {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H",
            "2h": "2H", "4h": "4H", "1d": "1D"}.get(p, "5m")


# ──────────────────────────────────────────────────────────────────────────
# Funding rate — KuCoin → Bitget → OKX
# ──────────────────────────────────────────────────────────────────────────
def fetch_funding_rate(symbol: str) -> Optional[float]:
    kc_sym = to_kucoin_symbol(symbol)
    try:
        r = SESSION.get(f"{KUCOIN_FUTURES}/api/v1/funding-rate/{kc_sym}/current", timeout=10)
        r.raise_for_status()
        d = (r.json() or {}).get("data") or {}
        if d.get("value") is not None:
            return float(d["value"])
    except Exception as e:
        _record_request_error("kucoin_funding", e, symbol=symbol)

    try:
        r = SESSION.get(f"{BITGET_BASE}/api/v2/mix/market/current-fund-rate",
                        params={"symbol": symbol, "productType": "USDT-FUTURES"}, timeout=10)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code", "")) == "00000":
            data = payload.get("data") or []
            row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
            if row and row.get("fundingRate") is not None:
                return float(row["fundingRate"])
    except Exception as e:
        _record_request_error("bitget_funding", e, symbol=symbol)

    try:
        r = SESSION.get(f"{OKX_BASE}/api/v5/public/funding-rate",
                        params={"instId": to_okx_inst(symbol)}, timeout=10)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code", "")) == "0":
            rows = payload.get("data") or []
            if rows and rows[0].get("fundingRate") is not None:
                return float(rows[0]["fundingRate"])
    except Exception as e:
        _record_request_error("okx_funding", e, symbol=symbol)

    return None


# ──────────────────────────────────────────────────────────────────────────
# Single-symbol full scan (matches v2.0 schema)
# ──────────────────────────────────────────────────────────────────────────
def fetch_ticker(symbol: str) -> Optional[dict]:
    kc_sym = to_kucoin_symbol(symbol)
    try:
        r = SESSION.get(f"{KUCOIN_FUTURES}/api/v1/contracts/{kc_sym}", timeout=10)
        r.raise_for_status()
        d = (r.json() or {}).get("data") or {}
        if d:
            mark = float(d.get("markPrice") or 0)
            return {
                "price": mark,
                "price_change_pct_24h": float(d.get("priceChgPct") or 0) * 100,
                "high_24h": float(d.get("highPrice") or 0),
                "low_24h": float(d.get("lowPrice") or 0),
                "volume_24h": float(d.get("turnoverOf24h") or 0),
            }
    except Exception as e:
        _record_request_error("kucoin_ticker", e, symbol=symbol)
    return None


def scan_symbol(
    symbol: str,
    preloaded_ticker: dict = None,
    scan_time: Optional[datetime] = None,
) -> Optional[dict]:
    """Output schema EXACTLY matches v2.0 — no other modules need changes."""
    eff = scan_time or datetime.now(timezone.utc)
    row = {"symbol": symbol, "scan_time": eff.isoformat()}

    if preloaded_ticker:
        public = {"price", "price_change_pct_24h", "high_24h", "low_24h", "volume_24h"}
        row.update({k: v for k, v in preloaded_ticker.items() if k in public})
    else:
        t = fetch_ticker(symbol)
        if t is None:
            return None
        row.update(t)

    row.update(get_multi_tf_rsi(symbol))

    # OI snapshot — preloaded if available
    if preloaded_ticker and preloaded_ticker.get("_oi_value_usdt"):
        row["oi_contracts"] = preloaded_ticker["_oi_value_usdt"]
    else:
        oi = fetch_open_interest(symbol)
        if oi:
            row["oi_contracts"] = oi["oi_contracts"]

    # OI history → 1h change
    oi_hist = fetch_oi_history(symbol, period="5m", limit=12)
    if oi_hist is not None and len(oi_hist) >= 2:
        row["oi_value_now"] = oi_hist["sumOpenInterestValue"].iloc[-1]
        row["oi_value_1h_ago"] = oi_hist["sumOpenInterestValue"].iloc[0]
        if row["oi_value_1h_ago"] != 0:
            row["oi_change_pct_1h"] = round(
                (row["oi_value_now"] - row["oi_value_1h_ago"]) / row["oi_value_1h_ago"] * 100, 3
            )

    if preloaded_ticker and preloaded_ticker.get("_funding_rate") is not None:
        row["funding_rate"] = preloaded_ticker["_funding_rate"]
    else:
        fr = fetch_funding_rate(symbol)
        if fr is not None:
            row["funding_rate"] = fr

    return row
