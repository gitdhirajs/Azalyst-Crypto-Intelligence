"""
src/exchange_fallback.py — spot + perp price helpers for the BasisEngine.

Used ONLY by BasisEngine. Order:
   KuCoin → Bitget → OKX → (optionally Binance/Bybit)

Spot endpoint mapping:
   KuCoin spot:  /api/v1/market/orderbook/level1?symbol=BTC-USDT
   Bitget spot:  /api/v2/spot/market/tickers?symbol=BTCUSDT
   OKX spot:     /api/v5/market/ticker?instId=BTC-USDT

Perp endpoint mapping:
   KuCoin Futures: /api/v1/contracts/{symbol}    (markPrice)
   Bitget Futures: /api/v2/mix/market/ticker     (lastPr)
   OKX SWAP:       /api/v5/market/ticker         (last)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

log = logging.getLogger("derived.fallback")


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AzalystCryptoScanner/2.1"})

KUCOIN_FUTURES = "https://api-futures.kucoin.com"
KUCOIN_SPOT = "https://api.kucoin.com"
BITGET_BASE = "https://api.bitget.com"
OKX_BASE = "https://www.okx.com"
BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_SPOT = "https://api.binance.com"
ALLOW_BLOCKED = os.getenv("ALLOW_BLOCKED_EXCHANGES", "").strip() in ("1", "true", "yes")


def _http_get(url: str, params: dict = None, timeout: int = 8) -> Optional[dict]:
    try:
        r = SESSION.get(url, params=params or {}, timeout=timeout)
        if r.status_code in (451, 403):
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        log.debug("HTTP failed %s: %s", url, exc)
        return None
    except ValueError:
        return None


def _to_kucoin_perp(canonical: str) -> str:
    if canonical.endswith("USDT"):
        base = canonical[:-4]
        if base == "BTC":
            base = "XBT"
        return f"{base}USDTM"
    return canonical


def _to_kucoin_spot(canonical: str) -> str:
    if canonical.endswith("USDT"):
        return f"{canonical[:-4]}-USDT"
    return canonical


def _to_okx_spot(canonical: str) -> str:
    if canonical.endswith("USDT"):
        return f"{canonical[:-4]}-USDT"
    return canonical


def _to_okx_swap(canonical: str) -> str:
    if canonical.endswith("USDT"):
        return f"{canonical[:-4]}-USDT-SWAP"
    return canonical


# ──────────────────────────────────────────────────────────────────────────
# Spot price (for BasisEngine)
# ──────────────────────────────────────────────────────────────────────────
def fetch_spot_price(symbol: str) -> Optional[float]:
    # KuCoin spot
    d = _http_get(f"{KUCOIN_SPOT}/api/v1/market/orderbook/level1",
                  {"symbol": _to_kucoin_spot(symbol)})
    if d and d.get("data"):
        try:
            return float(d["data"].get("price") or 0) or None
        except (TypeError, ValueError):
            pass

    # Bitget spot
    d = _http_get(f"{BITGET_BASE}/api/v2/spot/market/tickers", {"symbol": symbol})
    if d and str(d.get("code", "")) == "00000":
        rows = d.get("data") or []
        if rows:
            try:
                return float(rows[0].get("lastPr") or 0) or None
            except (TypeError, ValueError):
                pass

    # OKX spot
    d = _http_get(f"{OKX_BASE}/api/v5/market/ticker",
                  {"instId": _to_okx_spot(symbol)})
    if d and str(d.get("code", "")) == "0":
        rows = d.get("data") or []
        if rows:
            try:
                return float(rows[0].get("last") or 0) or None
            except (TypeError, ValueError):
                pass

    if ALLOW_BLOCKED:
        d = _http_get(f"{BINANCE_SPOT}/api/v3/ticker/price", {"symbol": symbol})
        if d and "price" in d:
            try:
                return float(d["price"])
            except (TypeError, ValueError):
                pass
    return None


def fetch_perp_price(symbol: str) -> Optional[float]:
    # KuCoin perp (mark)
    d = _http_get(f"{KUCOIN_FUTURES}/api/v1/contracts/{_to_kucoin_perp(symbol)}")
    if d and d.get("data"):
        try:
            return float(d["data"].get("markPrice") or 0) or None
        except (TypeError, ValueError):
            pass

    # Bitget perp
    d = _http_get(f"{BITGET_BASE}/api/v2/mix/market/ticker",
                  {"symbol": symbol, "productType": "USDT-FUTURES"})
    if d and str(d.get("code", "")) == "00000":
        data = d.get("data") or []
        row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
        if row:
            try:
                return float(row.get("lastPr") or 0) or None
            except (TypeError, ValueError):
                pass

    # OKX SWAP
    d = _http_get(f"{OKX_BASE}/api/v5/market/ticker",
                  {"instId": _to_okx_swap(symbol)})
    if d and str(d.get("code", "")) == "0":
        rows = d.get("data") or []
        if rows:
            try:
                return float(rows[0].get("last") or 0) or None
            except (TypeError, ValueError):
                pass

    if ALLOW_BLOCKED:
        d = _http_get(f"{BINANCE_FAPI}/fapi/v1/ticker/price", {"symbol": symbol})
        if d and "price" in d:
            try:
                return float(d["price"])
            except (TypeError, ValueError):
                pass
    return None
