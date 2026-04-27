"""Market-data collection using KuCoin Futures with Bitget fallback."""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import (
    BITGET_BASE,
    EXCLUDED_SYMBOLS,
    KLINE_LIMIT,
    KUCOIN_FUTURES_BASE,
    KUCOIN_UNIFIED_BASE,
    MARKET_DATA_PROVIDERS,
    MAX_SYMBOLS_PER_SCAN,
    MIN_OI_USDT,
    MIN_VOLUME_24H_USDT,
    REQUEST_DELAY,
    RSI_PERIOD,
    TIMEFRAMES,
)


SESSION = requests.Session()
SESSION.headers.update(
    {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        ),
    }
)

REQUEST_ERRORS: list[dict] = []
_FALLBACK_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "NEARUSDT",
    "FILUSDT",
    "SUIUSDT",
    "PEPEUSDT",
    "TRXUSDT",
]
_KUCOIN_TO_INTERNAL_BASE = {"XBT": "BTC"}
_INTERNAL_TO_KUCOIN_BASE = {value: key for key, value in _KUCOIN_TO_INTERNAL_BASE.items()}
_KUCOIN_KLINE_GRANULARITY = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}
_BITGET_KLINE_GRANULARITY = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}
_KUCOIN_OI_INTERVALS = {
    "1m": "5min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "1day",
}


def _float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return int(status_code)

    message = str(error)
    for pattern in [
        r"status code[:=]?\s*(\d{3})",
        r"\bHTTP\s+(\d{3})\b",
        r"\b(\d{3})\b",
    ]:
        match = re.search(pattern, message, flags=re.IGNORECASE)
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
    provider_counts = Counter(item["provider"] for item in REQUEST_ERRORS if item.get("provider"))

    return {
        "count": len(REQUEST_ERRORS),
        "status_counts": dict(status_counts),
        "endpoint_counts": dict(endpoint_counts),
        "provider_counts": dict(provider_counts),
        "examples": REQUEST_ERRORS[:max_examples],
    }


def _provider_order(preferred: str | None = None) -> list[str]:
    ordered: list[str] = []
    if preferred:
        preferred = preferred.strip().lower()
        if preferred in MARKET_DATA_PROVIDERS:
            ordered.append(preferred)
    for provider in MARKET_DATA_PROVIDERS:
        if provider not in ordered:
            ordered.append(provider)
    return ordered


def _annotate_bulk_tickers(provider: str, tickers: dict[str, dict]) -> dict[str, dict]:
    return {
        symbol: {"_provider": provider, **row}
        for symbol, row in tickers.items()
    }


def _provider_markers(provider: str, symbols: list[str]) -> dict[str, dict]:
    return {
        symbol: {
            "_provider": provider,
            "_provider_symbol": _to_provider_symbol(symbol, provider),
        }
        for symbol in symbols
    }


def _apply_scan_cap(symbols: list[str]) -> list[str]:
    if MAX_SYMBOLS_PER_SCAN > 0:
        return symbols[:MAX_SYMBOLS_PER_SCAN]
    return symbols


def _fallback_symbols_for(provider: str, universe: list[str] | None = None) -> list[str]:
    if universe:
        allowed = set(universe)
        subset = [symbol for symbol in _FALLBACK_SYMBOLS if symbol in allowed]
        if subset:
            return _apply_scan_cap(subset)
    return _apply_scan_cap(list(_FALLBACK_SYMBOLS))


def _normalize_symbol(provider: str, provider_symbol: str) -> str:
    if provider == "kucoin":
        symbol = provider_symbol[:-1] if provider_symbol.endswith("M") else provider_symbol
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            base = _KUCOIN_TO_INTERNAL_BASE.get(base, base)
            return f"{base}USDT"
    return provider_symbol


def _to_provider_symbol(symbol: str, provider: str) -> str:
    if provider == "kucoin":
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            base = _INTERNAL_TO_KUCOIN_BASE.get(base, base)
            return f"{base}USDTM"
    return symbol


def _http_get_json(
    url: str,
    params: dict,
    timeout: int,
    endpoint: str,
    provider: str,
    **details,
):
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = SESSION.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            _record_request_error(
                endpoint,
                exc,
                provider=provider,
                attempt=attempt,
                **details,
            )
            if attempt < 3:
                backoff = min(2 ** attempt, 30) + random.uniform(0, 0.5)
                time.sleep(backoff)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{provider} {endpoint} failed without a captured error")


def _kucoin_require_ok(payload: dict, endpoint: str) -> list | dict:
    code = str(payload.get("code"))
    if code != "200000":
        raise RuntimeError(f"KuCoin {endpoint} returned code={code}")
    return payload.get("data") or []


def _bitget_require_ok(payload: dict, endpoint: str) -> list | dict:
    code = str(payload.get("code"))
    if code != "00000":
        raise RuntimeError(f"Bitget {endpoint} returned code={code}: {payload.get('msg')}")
    return payload.get("data") or []


def _kucoin_contract_to_row(item: dict) -> dict | None:
    provider_symbol = item.get("symbol")
    if not provider_symbol:
        return None
    internal_symbol = _normalize_symbol("kucoin", provider_symbol)
    if internal_symbol in EXCLUDED_SYMBOLS:
        return None

    mark_price = _float_or_none(item.get("markPrice")) or _float_or_none(item.get("lastTradePrice"))
    last_trade_price = _float_or_none(item.get("lastTradePrice")) or mark_price
    open_interest = _float_or_none(item.get("openInterest"))
    multiplier = _float_or_none(item.get("multiplier"))
    oi_value_now = None
    if open_interest is not None and multiplier is not None and mark_price is not None:
        oi_value_now = open_interest * multiplier * mark_price

    return {
        "_provider_symbol": provider_symbol,
        "_contract_multiplier": multiplier,
        "price": mark_price or last_trade_price,
        "mark_price": mark_price,
        "funding_rate": _float_or_none(item.get("fundingFeeRate")),
        "oi_contracts": open_interest,
        "oi_value_now": oi_value_now,
        "price_change_pct_24h": (_float_or_none(item.get("priceChgPct")) or 0.0) * 100,
        "high_24h": _float_or_none(item.get("highPrice")),
        "low_24h": _float_or_none(item.get("lowPrice")),
        "volume_24h": _float_or_none(item.get("turnoverOf24h"))
        or _float_or_none(item.get("volumeOf24h")),
    }


@lru_cache(maxsize=1)
def _bitget_contract_metadata() -> dict[str, dict]:
    url = f"{BITGET_BASE}/api/v2/mix/market/contracts"
    payload = _http_get_json(
        url,
        {"productType": "USDT-FUTURES"},
        timeout=15,
        endpoint="contracts",
        provider="bitget",
    )
    items = _bitget_require_ok(payload, "contracts")
    metadata = {}
    for item in items:
        if (
            item.get("quoteCoin") == "USDT"
            and item.get("symbolType") == "perpetual"
            and item.get("symbolStatus") == "normal"
            and item.get("symbol") not in EXCLUDED_SYMBOLS
        ):
            metadata[item["symbol"]] = item
    return metadata


def _bitget_ticker_to_row(item: dict, contract: dict | None = None) -> dict | None:
    symbol = item.get("symbol")
    if not symbol or symbol in EXCLUDED_SYMBOLS:
        return None

    mark_price = _float_or_none(item.get("markPrice")) or _float_or_none(item.get("lastPr"))
    last_price = _float_or_none(item.get("lastPr")) or mark_price
    holding_amount = _float_or_none(item.get("holdingAmount"))
    oi_value_now = None
    if holding_amount is not None and mark_price is not None:
        oi_value_now = holding_amount * mark_price

    return {
        "_provider_symbol": symbol,
        "_contract_multiplier": _float_or_none((contract or {}).get("sizeMultiplier")),
        "price": mark_price or last_price,
        "mark_price": mark_price,
        "funding_rate": _float_or_none(item.get("fundingRate")),
        "oi_contracts": holding_amount,
        "oi_value_now": oi_value_now,
        "price_change_pct_24h": (_float_or_none(item.get("change24h")) or 0.0) * 100,
        "high_24h": _float_or_none(item.get("high24h")),
        "low_24h": _float_or_none(item.get("low24h")),
        "volume_24h": _float_or_none(item.get("usdtVolume"))
        or _float_or_none(item.get("quoteVolume")),
    }


def _kucoin_fetch_all_tickers_bulk() -> dict[str, dict]:
    url = f"{KUCOIN_FUTURES_BASE}/api/v1/contracts/active"
    try:
        payload = _http_get_json(
            url,
            {},
            timeout=15,
            endpoint="contracts_active",
            provider="kucoin",
        )
        items = _kucoin_require_ok(payload, "contracts_active")
        tickers: dict[str, dict] = {}
        for item in items:
            if (
                item.get("quoteCurrency") != "USDT"
                or item.get("settleCurrency") != "USDT"
                or item.get("marketStage") != "NORMAL"
                or not str(item.get("symbol", "")).endswith("M")
            ):
                continue
            row = _kucoin_contract_to_row(item)
            if row is None:
                continue
            tickers[_normalize_symbol("kucoin", item["symbol"])] = row
        return tickers
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("contracts_active", exc, provider="kucoin")
        print(f"  [ERR] kucoin contracts active: {exc}")
        return {}


def _bitget_fetch_all_tickers_bulk() -> dict[str, dict]:
    url = f"{BITGET_BASE}/api/v2/mix/market/tickers"
    try:
        contracts = _bitget_contract_metadata()
        payload = _http_get_json(
            url,
            {"productType": "USDT-FUTURES"},
            timeout=15,
            endpoint="bulk_ticker",
            provider="bitget",
        )
        items = _bitget_require_ok(payload, "bulk_ticker")
        tickers: dict[str, dict] = {}
        for item in items:
            symbol = item.get("symbol")
            if symbol not in contracts:
                continue
            row = _bitget_ticker_to_row(item, contracts.get(symbol))
            if row is None:
                continue
            tickers[symbol] = row
        return tickers
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("bulk_ticker", exc, provider="bitget")
        print(f"  [ERR] bitget bulk ticker: {exc}")
        return {}


def _kucoin_fetch_ticker(symbol: str) -> Optional[dict]:
    provider_symbol = _to_provider_symbol(symbol, "kucoin")
    tickers = _kucoin_fetch_all_tickers_bulk()
    row = tickers.get(symbol)
    if row:
        return row
    for internal_symbol, candidate in tickers.items():
        if candidate.get("_provider_symbol") == provider_symbol:
            return {"_provider_symbol": provider_symbol, **candidate, "_symbol": internal_symbol}
    return None


def _bitget_fetch_ticker(symbol: str) -> Optional[dict]:
    tickers = _bitget_fetch_all_tickers_bulk()
    return tickers.get(symbol)


def _format_ohlcv_frame(
    rows: list[list],
    interval_ms: int,
    quote_volume_name: str = "quote_volume",
) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df.shape[1] < 7:
        return None
    df = df.iloc[:, :7]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", quote_volume_name]
    for column in ["open", "high", "low", "close", "volume", quote_volume_name]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    df["close_time"] = df["open_time"] + pd.to_timedelta(interval_ms - 1, unit="ms")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    df = df.rename(columns={quote_volume_name: "quote_volume"})
    return df.sort_values("open_time").reset_index(drop=True)


def _kucoin_fetch_klines(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    granularity = _KUCOIN_KLINE_GRANULARITY.get(interval)
    if granularity is None:
        return None
    interval_ms = granularity * 60 * 1000
    provider_symbol = _to_provider_symbol(symbol, "kucoin")
    now_ms = int(time.time() * 1000)
    window_ms = interval_ms * max(limit + 5, limit * 2)
    url = f"{KUCOIN_FUTURES_BASE}/api/v1/kline/query"
    params = {
        "symbol": provider_symbol,
        "granularity": granularity,
        "from": now_ms - window_ms,
        "to": now_ms,
    }
    try:
        payload = _http_get_json(
            url,
            params,
            timeout=12,
            endpoint="klines",
            provider="kucoin",
            symbol=symbol,
            interval=interval,
        )
        rows = _kucoin_require_ok(payload, "klines")
        frame = _format_ohlcv_frame(rows, interval_ms=interval_ms)
        if frame is None or frame.empty:
            return None
        return frame.tail(limit).reset_index(drop=True)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("klines", exc, provider="kucoin", symbol=symbol, interval=interval)
        print(f"  [ERR] kucoin klines {symbol} {interval}: {exc}")
        return None


def _bitget_fetch_klines(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    granularity = _BITGET_KLINE_GRANULARITY.get(interval)
    if granularity is None:
        return None
    interval_ms = _KUCOIN_KLINE_GRANULARITY[interval] * 60 * 1000
    url = f"{BITGET_BASE}/api/v2/mix/market/candles"
    params = {
        "symbol": symbol,
        "productType": "USDT-FUTURES",
        "granularity": granularity,
        "limit": limit,
    }
    try:
        payload = _http_get_json(
            url,
            params,
            timeout=12,
            endpoint="klines",
            provider="bitget",
            symbol=symbol,
            interval=interval,
        )
        rows = _bitget_require_ok(payload, "klines")
        frame = _format_ohlcv_frame(rows, interval_ms=interval_ms)
        if frame is None or frame.empty:
            return None
        return frame.tail(limit).reset_index(drop=True)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("klines", exc, provider="bitget", symbol=symbol, interval=interval)
        print(f"  [ERR] bitget klines {symbol} {interval}: {exc}")
        return None


def _kucoin_fetch_open_interest(symbol: str) -> Optional[dict]:
    ticker = _kucoin_fetch_ticker(symbol)
    if not ticker:
        return None
    open_interest = _float_or_none(ticker.get("oi_contracts"))
    oi_value_now = _float_or_none(ticker.get("oi_value_now"))
    if open_interest is None:
        return None
    return {
        "oi_contracts": open_interest,
        "oi_value_now": oi_value_now,
        "oi_time": datetime.now(timezone.utc).isoformat(),
    }


def _bitget_fetch_open_interest(symbol: str) -> Optional[dict]:
    url = f"{BITGET_BASE}/api/v2/mix/market/open-interest"
    try:
        payload = _http_get_json(
            url,
            {"symbol": symbol, "productType": "USDT-FUTURES"},
            timeout=10,
            endpoint="open_interest",
            provider="bitget",
            symbol=symbol,
        )
        data = _bitget_require_ok(payload, "open_interest")
        rows = data.get("openInterestList") or []
        if not rows:
            return None
        oi_contracts = _float_or_none(rows[0].get("size"))
        ticker = _bitget_fetch_ticker(symbol) or {}
        mark_price = _float_or_none(ticker.get("mark_price"))
        oi_value_now = oi_contracts * mark_price if oi_contracts is not None and mark_price is not None else None
        return {
            "oi_contracts": oi_contracts,
            "oi_value_now": oi_value_now,
            "oi_time": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("open_interest", exc, provider="bitget", symbol=symbol)
        print(f"  [ERR] bitget open interest {symbol}: {exc}")
        return None


def _kucoin_fetch_oi_history(symbol: str, period: str, limit: int) -> Optional[pd.DataFrame]:
    interval = _KUCOIN_OI_INTERVALS.get(period)
    if interval is None:
        return None

    ticker = _kucoin_fetch_ticker(symbol) or {}
    multiplier = _float_or_none(ticker.get("_contract_multiplier"))
    mark_price = _float_or_none(ticker.get("mark_price"))
    provider_symbol = ticker.get("_provider_symbol") or _to_provider_symbol(symbol, "kucoin")
    url = f"{KUCOIN_UNIFIED_BASE}/api/ua/v1/market/open-interest"
    params = {
        "symbol": provider_symbol,
        "interval": interval,
        "pageSize": limit,
    }
    try:
        payload = _http_get_json(
            url,
            params,
            timeout=12,
            endpoint="open_interest_history",
            provider="kucoin",
            symbol=symbol,
            period=period,
        )
        rows = _kucoin_require_ok(payload, "open_interest_history")
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["sumOpenInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
        if multiplier is not None and mark_price is not None:
            df["sumOpenInterestValue"] = df["sumOpenInterest"] * multiplier * mark_price
        else:
            df["sumOpenInterestValue"] = df["sumOpenInterest"]
        return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].sort_values(
            "timestamp"
        ).reset_index(drop=True)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error(
                "open_interest_history",
                exc,
                provider="kucoin",
                symbol=symbol,
                period=period,
            )
        print(f"  [ERR] kucoin OI history {symbol}: {exc}")
        return None


def _bitget_fetch_oi_history(symbol: str, period: str, limit: int) -> Optional[pd.DataFrame]:
    _ = (symbol, period, limit)
    return None


def _kucoin_fetch_funding_rate(symbol: str) -> Optional[float]:
    ticker = _kucoin_fetch_ticker(symbol)
    return _float_or_none((ticker or {}).get("funding_rate"))


def _bitget_fetch_funding_rate(symbol: str) -> Optional[float]:
    ticker = _bitget_fetch_ticker(symbol)
    return _float_or_none((ticker or {}).get("funding_rate"))


def filter_symbols_by_volume(symbols: list[str], tickers: dict[str, dict]) -> list[str]:
    """Filter symbols by minimum 24h volume and optional OI value."""
    valid = []
    for symbol in symbols:
        ticker = tickers.get(symbol)
        if not ticker or ticker.get("volume_24h") is None:
            continue
        if ticker["volume_24h"] < MIN_VOLUME_24H_USDT:
            continue
        if MIN_OI_USDT > 0 and (ticker.get("oi_value_now") or 0) < MIN_OI_USDT:
            continue
        valid.append((symbol, ticker["volume_24h"]))

    valid.sort(key=lambda item: item[1], reverse=True)
    return _apply_scan_cap([symbol for symbol, _ in valid])


def get_active_symbols() -> tuple[list[str], dict[str, dict]]:
    """
    Discover active USDT perpetual futures using the configured provider order.

    Returns `(symbols, bulk_tickers)` where bulk ticker rows include hidden
    `_provider` and `_provider_symbol` keys so downstream requests can keep the
    provider-specific contract format while the dashboard stays on `BTCUSDT`.
    """
    for provider in _provider_order():
        if provider == "kucoin":
            tickers = _kucoin_fetch_all_tickers_bulk()
        elif provider == "bitget":
            tickers = _bitget_fetch_all_tickers_bulk()
        else:
            continue

        all_symbols = sorted(tickers)
        if tickers:
            filtered = filter_symbols_by_volume(all_symbols, tickers)
            if filtered:
                return filtered, _annotate_bulk_tickers(provider, tickers)

        if all_symbols:
            fallback = _fallback_symbols_for(provider, universe=all_symbols)
            print(f"  [WARN] {provider} volume filter returned no symbols, using fallback subset")
            return fallback, _provider_markers(provider, fallback)

    print("  [WARN] All providers unavailable, using fallback symbol list")
    provider = _provider_order()[0] if _provider_order() else "kucoin"
    fallback = _fallback_symbols_for(provider)
    return fallback, _provider_markers(provider, fallback)


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int = KLINE_LIMIT,
    provider: str | None = None,
) -> Optional[pd.DataFrame]:
    """Fetch futures klines from the preferred provider, then configured fallbacks."""
    for candidate in _provider_order(provider):
        if candidate == "kucoin":
            df = _kucoin_fetch_klines(symbol, interval, limit)
        elif candidate == "bitget":
            df = _bitget_fetch_klines(symbol, interval, limit)
        else:
            continue
        if df is not None and not df.empty:
            return df
    return None


def compute_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI using Wilder-style exponential smoothing."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def get_rsi_for_symbol(symbol: str, interval: str, provider: str | None = None) -> Optional[float]:
    """Return the latest RSI value for a symbol on a given timeframe."""
    df = fetch_klines(symbol, interval, provider=provider)
    if df is None or len(df) < RSI_PERIOD + 1:
        return None
    rsi_series = compute_rsi(df["close"])
    return round(rsi_series.iloc[-1], 2)


def get_multi_tf_rsi(symbol: str, provider: str | None = None) -> dict:
    """Return RSI snapshots for all configured timeframes."""
    result = {}
    for label, interval in TIMEFRAMES.items():
        result[f"rsi_{label}"] = get_rsi_for_symbol(symbol, interval, provider=provider)
        time.sleep(REQUEST_DELAY)
    return result


def fetch_open_interest(symbol: str, provider: str | None = None) -> Optional[dict]:
    """Fetch current open interest from the preferred provider chain."""
    for candidate in _provider_order(provider):
        if candidate == "kucoin":
            row = _kucoin_fetch_open_interest(symbol)
        elif candidate == "bitget":
            row = _bitget_fetch_open_interest(symbol)
        else:
            continue
        if row:
            return row
    return None


def fetch_oi_history(
    symbol: str,
    period: str = "5m",
    limit: int = 30,
    provider: str | None = None,
) -> Optional[pd.DataFrame]:
    """Fetch open-interest history from the preferred provider chain."""
    for candidate in _provider_order(provider):
        if candidate == "kucoin":
            df = _kucoin_fetch_oi_history(symbol, period, limit)
        elif candidate == "bitget":
            df = _bitget_fetch_oi_history(symbol, period, limit)
        else:
            continue
        if df is not None and not df.empty:
            return df
    return None


def fetch_ticker(symbol: str, provider: str | None = None) -> Optional[dict]:
    """Fetch a single ticker snapshot from the preferred provider chain."""
    for candidate in _provider_order(provider):
        if candidate == "kucoin":
            row = _kucoin_fetch_ticker(symbol)
        elif candidate == "bitget":
            row = _bitget_fetch_ticker(symbol)
        else:
            continue
        if row:
            return {"_provider": candidate, **row}
    return None


def fetch_funding_rate(symbol: str, provider: str | None = None) -> Optional[float]:
    """Fetch the current funding rate from the preferred provider chain."""
    for candidate in _provider_order(provider):
        if candidate == "kucoin":
            funding_rate = _kucoin_fetch_funding_rate(symbol)
        elif candidate == "bitget":
            funding_rate = _bitget_fetch_funding_rate(symbol)
        else:
            continue
        if funding_rate is not None:
            return funding_rate
    return None


def scan_symbol(
    symbol: str,
    preloaded_ticker: dict | None = None,
    scan_time: Optional[datetime] = None,
) -> Optional[dict]:
    """Collect the scanner snapshot for a single symbol."""
    effective_scan_time = scan_time or datetime.now(timezone.utc)
    provider = (preloaded_ticker or {}).get("_provider")
    row = {"symbol": symbol, "scan_time": effective_scan_time.isoformat()}
    if provider:
        row["market_provider"] = provider

    if preloaded_ticker and preloaded_ticker.get("price") is not None:
        row.update({key: value for key, value in preloaded_ticker.items() if not key.startswith("_")})
    else:
        ticker = fetch_ticker(symbol, provider=provider)
        if ticker is None:
            return None
        provider = ticker.get("_provider", provider)
        if provider:
            row["market_provider"] = provider
        row.update({key: value for key, value in ticker.items() if not key.startswith("_")})

    row.update(get_multi_tf_rsi(symbol, provider=provider))

    if "oi_contracts" not in row:
        open_interest = fetch_open_interest(symbol, provider=provider)
        if open_interest:
            row["oi_contracts"] = open_interest["oi_contracts"]
            if open_interest.get("oi_value_now") is not None:
                row["oi_value_now"] = open_interest["oi_value_now"]

    oi_hist = fetch_oi_history(symbol, period="5m", limit=12, provider=provider)
    if oi_hist is not None and len(oi_hist) >= 2:
        if "sumOpenInterest" in oi_hist.columns:
            row["oi_contracts"] = oi_hist["sumOpenInterest"].iloc[-1]
        row["oi_value_now"] = oi_hist["sumOpenInterestValue"].iloc[-1]
        row["oi_value_1h_ago"] = oi_hist["sumOpenInterestValue"].iloc[0]
        if row["oi_value_1h_ago"]:
            row["oi_change_pct_1h"] = round(
                (row["oi_value_now"] - row["oi_value_1h_ago"]) / row["oi_value_1h_ago"] * 100,
                3,
            )

    if "funding_rate" not in row:
        funding_rate = fetch_funding_rate(symbol, provider=provider)
        if funding_rate is not None:
            row["funding_rate"] = funding_rate

    return row
