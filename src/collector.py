"""Market-data collection with provider fallback and optional proxy routing."""

from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import (
    BINANCE_FAPI_BASE,
    BYBIT_V5_BASE,
    EXCLUDED_SYMBOLS,
    KLINE_LIMIT,
    MARKET_DATA_PROVIDERS,
    MARKET_PROXY_FALLBACK_DIRECT,
    MARKET_PROXY_HOST,
    MARKET_PROXY_PORT,
    MARKET_PROXY_URL,
    MAX_SYMBOLS_PER_SCAN,
    MIN_OI_USDT,
    MIN_VOLUME_24H_USDT,
    REQUEST_DELAY,
    RSI_PERIOD,
    TIMEFRAMES,
)


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoScanner/3.0"})
REQUEST_ERRORS: list[dict] = []
PROXY_MAP = {"http": MARKET_PROXY_URL, "https": MARKET_PROXY_URL} if MARKET_PROXY_URL else None
PROXY_LABEL = (
    f"{MARKET_PROXY_HOST}:{MARKET_PROXY_PORT}"
    if MARKET_PROXY_HOST and MARKET_PROXY_PORT
    else "configured-proxy"
)


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
        r"Tunnel connection failed:\s*(\d{3})",
        r"status code[:=]?\s*(\d{3})",
        r"\bHTTP\s+(\d{3})\b",
    ]:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.search(r"\b(\d{3})\b", message)
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
    route_counts = Counter(item["route"] for item in REQUEST_ERRORS if item.get("route"))

    return {
        "count": len(REQUEST_ERRORS),
        "status_counts": dict(status_counts),
        "endpoint_counts": dict(endpoint_counts),
        "provider_counts": dict(provider_counts),
        "route_counts": dict(route_counts),
        "examples": REQUEST_ERRORS[:max_examples],
    }


def _request_routes() -> list[tuple[str, dict[str, str] | None]]:
    routes: list[tuple[str, dict[str, str] | None]] = []
    if PROXY_MAP:
        routes.append((f"proxy:{PROXY_LABEL}", PROXY_MAP))
    if not PROXY_MAP or MARKET_PROXY_FALLBACK_DIRECT:
        routes.append(("direct", None))
    return routes


def _http_get(
    url: str,
    params: dict,
    timeout: int,
    endpoint: str,
    provider: str,
    **details,
) -> tuple[requests.Response, str]:
    last_error: Exception | None = None
    for route_name, proxies in _request_routes():
        try:
            response = SESSION.get(url, params=params, timeout=timeout, proxies=proxies)
            response.raise_for_status()
            return response, route_name
        except Exception as exc:
            last_error = exc
            _record_request_error(
                endpoint,
                exc,
                provider=provider,
                route=route_name,
                **details,
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No request route available for {provider} {endpoint}")


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
    return {symbol: {"_provider": provider} for symbol in symbols}


def _apply_scan_cap(symbols: list[str]) -> list[str]:
    if MAX_SYMBOLS_PER_SCAN > 0:
        return symbols[:MAX_SYMBOLS_PER_SCAN]
    return symbols


def _fallback_symbols_for(provider: str, universe: list[str] | None = None) -> list[str]:
    if universe:
        subset = [symbol for symbol in _FALLBACK_SYMBOLS if symbol in set(universe)]
        if subset:
            return _apply_scan_cap(subset)
    return _apply_scan_cap(list(_FALLBACK_SYMBOLS))


def _bybit_require_ok(data: dict, endpoint: str) -> None:
    ret_code = data.get("retCode")
    if ret_code not in (None, 0, "0"):
        raise RuntimeError(
            f"Bybit {endpoint} returned retCode={ret_code}: {data.get('retMsg')}"
        )


def _bybit_ticker_row(payload: dict) -> dict:
    price_change_fraction = _float_or_none(payload.get("price24hPcnt"))
    open_interest = _float_or_none(payload.get("openInterest"))
    open_interest_value = _float_or_none(payload.get("openInterestValue"))
    funding_rate = _float_or_none(payload.get("fundingRate"))
    mark_price = _float_or_none(payload.get("markPrice"))

    row = {
        "price": float(payload["lastPrice"]),
        "price_change_pct_24h": (
            price_change_fraction * 100 if price_change_fraction is not None else None
        ),
        "high_24h": float(payload["highPrice24h"]),
        "low_24h": float(payload["lowPrice24h"]),
        "volume_24h": float(payload["turnover24h"]),
    }
    if open_interest is not None:
        row["oi_contracts"] = open_interest
    if open_interest_value is not None:
        row["oi_value_now"] = open_interest_value
    if funding_rate is not None:
        row["funding_rate"] = funding_rate
    if mark_price is not None:
        row["mark_price"] = mark_price
    return row


def _binance_ticker_row(payload: dict, premium: dict | None = None) -> dict:
    funding_rate = _float_or_none((premium or {}).get("lastFundingRate"))
    mark_price = _float_or_none((premium or {}).get("markPrice"))

    row = {
        "price": float(payload["lastPrice"]),
        "price_change_pct_24h": _float_or_none(payload.get("priceChangePercent")),
        "high_24h": float(payload["highPrice"]),
        "low_24h": float(payload["lowPrice"]),
        "volume_24h": float(payload["quoteVolume"]),
    }
    if funding_rate is not None:
        row["funding_rate"] = funding_rate
    if mark_price is not None:
        row["mark_price"] = mark_price
    return row


def _bybit_fetch_all_futures_symbols() -> list[str]:
    url = f"{BYBIT_V5_BASE}/market/instruments-info"
    params = {"category": "linear", "limit": 1000}
    try:
        response, _ = _http_get(url, params, timeout=15, endpoint="exchange_info", provider="bybit")
        data = response.json()
        _bybit_require_ok(data, "exchange_info")
        symbols = []
        for item in data.get("result", {}).get("list", []):
            if (
                item.get("status") == "Trading"
                and item.get("contractType") == "LinearPerpetual"
                and item.get("quoteCoin") == "USDT"
                and item["symbol"] not in EXCLUDED_SYMBOLS
            ):
                symbols.append(item["symbol"])
        return sorted(symbols)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("exchange_info", exc, provider="bybit")
        print(f"  [ERR] bybit exchange info: {exc}")
        return []


def _bybit_fetch_all_tickers_bulk() -> dict[str, dict]:
    url = f"{BYBIT_V5_BASE}/market/tickers"
    params = {"category": "linear"}
    try:
        response, _ = _http_get(url, params, timeout=15, endpoint="bulk_ticker", provider="bybit")
        data = response.json()
        _bybit_require_ok(data, "bulk_ticker")
        return {
            item["symbol"]: _bybit_ticker_row(item)
            for item in data.get("result", {}).get("list", [])
        }
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("bulk_ticker", exc, provider="bybit")
        print(f"  [ERR] bybit bulk ticker: {exc}")
        return {}


def _bybit_fetch_ticker(symbol: str) -> Optional[dict]:
    url = f"{BYBIT_V5_BASE}/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    try:
        response, _ = _http_get(url, params, timeout=10, endpoint="ticker", provider="bybit", symbol=symbol)
        data = response.json()
        _bybit_require_ok(data, "ticker")
        items = data.get("result", {}).get("list", [])
        return _bybit_ticker_row(items[0]) if items else None
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("ticker", exc, provider="bybit", symbol=symbol)
        print(f"  [ERR] bybit ticker {symbol}: {exc}")
        return None


def _bybit_fetch_klines(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    interval_map = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }
    url = f"{BYBIT_V5_BASE}/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval_map.get(interval, interval),
        "limit": limit,
    }
    try:
        response, _ = _http_get(
            url,
            params,
            timeout=10,
            endpoint="klines",
            provider="bybit",
            symbol=symbol,
            interval=interval,
        )
        data = response.json()
        _bybit_require_ok(data, "klines")
        klines_data = data.get("result", {}).get("list", [])
        if not klines_data:
            return None
        df = pd.DataFrame(klines_data)
        if df.shape[1] < 7:
            raise ValueError(f"Bybit kline returned {df.shape[1]} columns, expected at least 7")
        df = df.iloc[:, :7]
        df.columns = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
        for column in ["open", "high", "low", "close", "volume", "turnover"]:
            df[column] = df[column].astype(float)
        df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
        df.rename(columns={"turnover": "quote_volume"}, inplace=True)
        df["close_time"] = df["open_time"]
        return df.sort_values("open_time").reset_index(drop=True)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("klines", exc, provider="bybit", symbol=symbol, interval=interval)
        print(f"  [ERR] bybit klines {symbol} {interval}: {exc}")
        return None


def _bybit_fetch_open_interest(symbol: str) -> Optional[dict]:
    url = f"{BYBIT_V5_BASE}/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    try:
        response, _ = _http_get(
            url,
            params,
            timeout=10,
            endpoint="open_interest",
            provider="bybit",
            symbol=symbol,
        )
        data = response.json()
        _bybit_require_ok(data, "open_interest")
        items = data.get("result", {}).get("list", [])
        if not items:
            return None
        payload = items[0]
        open_interest = _float_or_none(payload.get("openInterest"))
        open_interest_value = _float_or_none(payload.get("openInterestValue"))
        if open_interest is None:
            return None
        return {
            "oi_contracts": open_interest,
            "oi_value_now": open_interest_value,
            "oi_time": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("open_interest", exc, provider="bybit", symbol=symbol)
        print(f"  [ERR] bybit open interest {symbol}: {exc}")
        return None


def _bybit_fetch_oi_history(symbol: str, period: str, limit: int) -> Optional[pd.DataFrame]:
    period_map = {
        "1m": "5min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    url = f"{BYBIT_V5_BASE}/market/open-interest"
    params = {
        "category": "linear",
        "symbol": symbol,
        "intervalTime": period_map.get(period, period),
        "limit": limit,
    }
    try:
        response, _ = _http_get(
            url,
            params,
            timeout=10,
            endpoint="open_interest_history",
            provider="bybit",
            symbol=symbol,
            period=period,
        )
        data = response.json()
        _bybit_require_ok(data, "open_interest_history")
        rows = data.get("result", {}).get("list", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["openInterest"] = df["openInterest"].astype(float)
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
        df["sumOpenInterestValue"] = df["openInterest"]
        df.rename(columns={"openInterest": "sumOpenInterest"}, inplace=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error(
                "open_interest_history",
                exc,
                provider="bybit",
                symbol=symbol,
                period=period,
            )
        print(f"  [ERR] bybit OI history {symbol}: {exc}")
        return None


def _bybit_fetch_funding_rate(symbol: str) -> Optional[float]:
    ticker = _bybit_fetch_ticker(symbol)
    return _float_or_none((ticker or {}).get("funding_rate"))


def _binance_fetch_all_futures_symbols() -> list[str]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/exchangeInfo"
    try:
        response, _ = _http_get(url, {}, timeout=15, endpoint="exchange_info", provider="binance")
        data = response.json()
        symbols = []
        for item in data.get("symbols", []):
            if (
                item.get("status") == "TRADING"
                and item.get("contractType") == "PERPETUAL"
                and item.get("quoteAsset") == "USDT"
                and item["symbol"] not in EXCLUDED_SYMBOLS
            ):
                symbols.append(item["symbol"])
        return sorted(symbols)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("exchange_info", exc, provider="binance")
        print(f"  [ERR] binance exchange info: {exc}")
        return []


def _binance_fetch_premium_index_bulk() -> dict[str, dict]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/premiumIndex"
    try:
        response, _ = _http_get(
            url,
            {},
            timeout=15,
            endpoint="bulk_premium_index",
            provider="binance",
        )
        data = response.json()
        if isinstance(data, list):
            return {item["symbol"]: item for item in data if "symbol" in item}
        if isinstance(data, dict) and data.get("symbol"):
            return {data["symbol"]: data}
        return {}
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("bulk_premium_index", exc, provider="binance")
        print(f"  [ERR] binance premium index bulk: {exc}")
        return {}


def _binance_fetch_premium_index(symbol: str) -> dict | None:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/premiumIndex"
    params = {"symbol": symbol}
    try:
        response, _ = _http_get(
            url,
            params,
            timeout=10,
            endpoint="premium_index",
            provider="binance",
            symbol=symbol,
        )
        data = response.json()
        return data if isinstance(data, dict) and data.get("symbol") else None
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("premium_index", exc, provider="binance", symbol=symbol)
        print(f"  [ERR] binance premium index {symbol}: {exc}")
        return None


def _binance_fetch_all_tickers_bulk() -> dict[str, dict]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/ticker/24hr"
    premiums = _binance_fetch_premium_index_bulk()
    try:
        response, _ = _http_get(url, {}, timeout=15, endpoint="bulk_ticker", provider="binance")
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Binance bulk ticker response was not a list")
        tickers = {}
        for item in data:
            symbol = item.get("symbol")
            if not symbol:
                continue
            tickers[symbol] = _binance_ticker_row(item, premiums.get(symbol))
        return tickers
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("bulk_ticker", exc, provider="binance")
        print(f"  [ERR] binance bulk ticker: {exc}")
        return {}


def _binance_fetch_ticker(symbol: str) -> Optional[dict]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/ticker/24hr"
    params = {"symbol": symbol}
    premium = _binance_fetch_premium_index(symbol)
    try:
        response, _ = _http_get(
            url,
            params,
            timeout=10,
            endpoint="ticker",
            provider="binance",
            symbol=symbol,
        )
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Binance ticker response was not an object")
        return _binance_ticker_row(data, premium)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("ticker", exc, provider="binance", symbol=symbol)
        print(f"  [ERR] binance ticker {symbol}: {exc}")
        return None


def _binance_fetch_klines(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response, _ = _http_get(
            url,
            params,
            timeout=10,
            endpoint="klines",
            provider="binance",
            symbol=symbol,
            interval=interval,
        )
        data = response.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        if df.shape[1] < 8:
            raise ValueError(f"Binance kline returned {df.shape[1]} columns, expected at least 8")
        df = df.iloc[:, :8]
        df.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
        ]
        for column in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[column] = df[column].astype(float)
        df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(pd.to_numeric(df["close_time"]), unit="ms", utc=True)
        return df.sort_values("open_time").reset_index(drop=True)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("klines", exc, provider="binance", symbol=symbol, interval=interval)
        print(f"  [ERR] binance klines {symbol} {interval}: {exc}")
        return None


def _binance_fetch_open_interest(symbol: str) -> Optional[dict]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    premium = _binance_fetch_premium_index(symbol)
    try:
        response, _ = _http_get(
            url,
            params,
            timeout=10,
            endpoint="open_interest",
            provider="binance",
            symbol=symbol,
        )
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Binance open interest response was not an object")
        open_interest = _float_or_none(data.get("openInterest"))
        mark_price = _float_or_none((premium or {}).get("markPrice"))
        if open_interest is None:
            return None
        oi_value_now = open_interest * mark_price if mark_price is not None else None
        return {
            "oi_contracts": open_interest,
            "oi_value_now": oi_value_now,
            "oi_time": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error("open_interest", exc, provider="binance", symbol=symbol)
        print(f"  [ERR] binance open interest {symbol}: {exc}")
        return None


def _binance_fetch_oi_history(symbol: str, period: str, limit: int) -> Optional[pd.DataFrame]:
    period_map = {
        "1m": "5m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    url = f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period_map.get(period, period), "limit": limit}
    try:
        response, _ = _http_get(
            url,
            params,
            timeout=10,
            endpoint="open_interest_history",
            provider="binance",
            symbol=symbol,
            period=period,
        )
        data = response.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        if not isinstance(exc, requests.RequestException):
            _record_request_error(
                "open_interest_history",
                exc,
                provider="binance",
                symbol=symbol,
                period=period,
            )
        print(f"  [ERR] binance OI history {symbol}: {exc}")
        return None


def _binance_fetch_funding_rate(symbol: str) -> Optional[float]:
    premium = _binance_fetch_premium_index(symbol)
    return _float_or_none((premium or {}).get("lastFundingRate"))


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

    Returns `(symbols, bulk_tickers)` where bulk ticker rows include a hidden
    `_provider` key so downstream requests can prefer the same exchange.
    """
    for provider in _provider_order():
        if provider == "binance":
            all_symbols = _binance_fetch_all_futures_symbols()
            if not all_symbols:
                continue
            tickers = _binance_fetch_all_tickers_bulk()
        elif provider == "bybit":
            all_symbols = _bybit_fetch_all_futures_symbols()
            if not all_symbols:
                continue
            tickers = _bybit_fetch_all_tickers_bulk()
        else:
            continue

        if tickers:
            filtered = filter_symbols_by_volume(all_symbols, tickers)
            if filtered:
                return filtered, _annotate_bulk_tickers(provider, tickers)

        fallback = _fallback_symbols_for(provider, universe=all_symbols)
        print(f"  [WARN] {provider} bulk tickers unavailable, using fallback symbol subset")
        return fallback, _provider_markers(provider, fallback)

    print("  [WARN] All providers unavailable, using fallback symbol list")
    fallback = _fallback_symbols_for(_provider_order()[0] if _provider_order() else "binance")
    return fallback, _provider_markers(_provider_order()[0] if _provider_order() else "binance", fallback)


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


def fetch_klines(symbol: str, interval: str, limit: int = KLINE_LIMIT, provider: str | None = None) -> Optional[pd.DataFrame]:
    """Fetch futures klines from the preferred provider, then configured fallbacks."""
    for candidate in _provider_order(provider):
        if candidate == "binance":
            df = _binance_fetch_klines(symbol, interval, limit)
        elif candidate == "bybit":
            df = _bybit_fetch_klines(symbol, interval, limit)
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
        if candidate == "binance":
            row = _binance_fetch_open_interest(symbol)
        elif candidate == "bybit":
            row = _bybit_fetch_open_interest(symbol)
        else:
            continue
        if row:
            return row
    return None


def fetch_oi_history(symbol: str, period: str = "5m", limit: int = 30, provider: str | None = None) -> Optional[pd.DataFrame]:
    """Fetch open-interest history from the preferred provider chain."""
    for candidate in _provider_order(provider):
        if candidate == "binance":
            df = _binance_fetch_oi_history(symbol, period, limit)
        elif candidate == "bybit":
            df = _bybit_fetch_oi_history(symbol, period, limit)
        else:
            continue
        if df is not None and not df.empty:
            return df
    return None


def fetch_ticker(symbol: str, provider: str | None = None) -> Optional[dict]:
    """Fetch a single ticker snapshot from the preferred provider chain."""
    for candidate in _provider_order(provider):
        if candidate == "binance":
            row = _binance_fetch_ticker(symbol)
        elif candidate == "bybit":
            row = _bybit_fetch_ticker(symbol)
        else:
            continue
        if row:
            return {"_provider": candidate, **row}
    return None


def fetch_funding_rate(symbol: str, provider: str | None = None) -> Optional[float]:
    """Fetch the current funding rate from the preferred provider chain."""
    for candidate in _provider_order(provider):
        if candidate == "binance":
            funding_rate = _binance_fetch_funding_rate(symbol)
        elif candidate == "bybit":
            funding_rate = _bybit_fetch_funding_rate(symbol)
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
