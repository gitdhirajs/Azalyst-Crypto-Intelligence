"""
src/derived_data.py — FREE multi-exchange aggregator for engine layer.

DESIGNED FOR GITHUB ACTIONS:
  Uses only exchanges that respond to GCP/AWS IPs:
    • KuCoin Futures
    • Bitget Futures (v2 API)
    • OKX SWAP

  Binance and Bybit return HTTP 451 to GitHub-hosted runner IPs in many
  regions, so they're not in the primary path. They become available
  only if you set ALLOW_BLOCKED_EXCHANGES=1 (useful locally).

What it provides (drop-in replacement for the old Azalyst client):

    Liquidation heatmap   → OKX public liquidation orders (free, 7-day) +
                            implied liquidation zones from cross-exchange OI
                            distributed across leverage tiers.

    Cross-exchange funding → KuCoin + Bitget + OKX funding endpoints,
                             aggregated locally with OI weighting.

    Long/short ratios     → Bitget account-long-short + position-long-short
                            + OKX rubik long-short-account-ratio
                            + OKX taker-volume-contract.

    Aggregated OI         → KuCoin + Bitget + OKX OI snapshots.

NO API KEYS. ALL FREE.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("derived.data")


KUCOIN_FUTURES = "https://api-futures.kucoin.com"
BITGET_BASE = "https://api.bitget.com"
OKX_BASE = "https://www.okx.com"

DEFAULT_TIMEOUT = float(os.getenv("DERIVED_TIMEOUT", "10"))
DERIVED_RPS = float(os.getenv("DERIVED_RPS", "8"))


LEVERAGE_TIERS = [
    (10,  0.30),
    (25,  0.30),
    (50,  0.25),
    (100, 0.15),
]
MAINTENANCE_MARGIN = 0.005


# ──────────────────────────────────────────────────────────────────────────
@dataclass
class LiquidationLevel:
    price: float
    notional_usdt: float
    side: str


@dataclass
class LiquidationHeatmap:
    symbol: str
    timeframe: str
    last_price: float
    levels: List[LiquidationLevel] = field(default_factory=list)

    def levels_above(self, ref_price: float, max_pct: float = 5.0) -> List[LiquidationLevel]:
        upper = ref_price * (1 + max_pct / 100.0)
        return [lvl for lvl in self.levels
                if lvl.side == "short" and ref_price < lvl.price <= upper]

    def levels_below(self, ref_price: float, max_pct: float = 5.0) -> List[LiquidationLevel]:
        lower = ref_price * (1 - max_pct / 100.0)
        return [lvl for lvl in self.levels
                if lvl.side == "long" and lower <= lvl.price < ref_price]


@dataclass
class FundingSnapshot:
    symbol: str
    avg_funding: float
    max_funding: float
    min_funding: float
    spread_bps: float
    oi_weighted_funding: float
    asof_ms: int
    exchanges: Dict[str, float] = field(default_factory=dict)


@dataclass
class LongShortSnapshot:
    symbol: str
    asof_ms: int
    top_account_ratio: Optional[float] = None
    global_account_ratio: Optional[float] = None
    top_position_ratio: Optional[float] = None
    taker_buy_sell_ratio: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────
class DerivedDataClient:
    """
    Aggregates cross-exchange data from KuCoin + Bitget + OKX.
    All free, no API keys, GitHub-Actions-friendly.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, rps: float = DERIVED_RPS):
        self.timeout = timeout
        self._min_interval = 1.0 / max(rps, 0.5)
        self._last_call_ts = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "AzalystCryptoScanner/2.1",
        })

    @property
    def enabled(self) -> bool:
        return True

    def _throttle(self):
        wait = self._min_interval - (time.time() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[dict]:
        self._throttle()
        try:
            r = self.session.get(url, params=params or {}, timeout=self.timeout)
            if r.status_code in (451, 403):
                log.debug("HTTP %s on %s", r.status_code, url)
                return None
            if r.status_code == 429:
                time.sleep(5)
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            log.debug("Request failed %s: %s", url, exc)
            return None
        except ValueError as exc:
            log.debug("JSON parse error %s: %s", url, exc)
            return None

    # ── Symbol mappers ────────────────────────────────────────────────────
    @staticmethod
    def _to_kucoin(base: str) -> str:
        if base.upper() == "BTC":
            return "XBTUSDTM"
        return f"{base.upper()}USDTM"

    @staticmethod
    def _to_bitget(base: str) -> str:
        return f"{base.upper()}USDT"

    @staticmethod
    def _to_okx(base: str) -> str:
        return f"{base.upper()}-USDT-SWAP"

    @staticmethod
    def _from_okx_uly(base: str) -> str:
        return f"{base.upper()}-USDT"

    # ── Aggregated OI snapshot (USDT) ────────────────────────────────────
    def aggregated_oi_now(self, base: str) -> Dict[str, float]:
        out: Dict[str, float] = {}

        # KuCoin
        kc_sym = self._to_kucoin(base)
        d = self._get(f"{KUCOIN_FUTURES}/api/v1/contracts/{kc_sym}")
        if d and d.get("data"):
            try:
                contracts = float(d["data"].get("openInterest") or 0)
                multiplier = abs(float(d["data"].get("multiplier") or 1))
                price = float(d["data"].get("markPrice") or 0)
                if contracts > 0 and price > 0:
                    out["kucoin"] = contracts * multiplier * price
            except (TypeError, ValueError):
                pass

        # Bitget
        bg_sym = self._to_bitget(base)
        d = self._get(f"{BITGET_BASE}/api/v2/mix/market/open-interest",
                      {"symbol": bg_sym, "productType": "USDT-FUTURES"})
        if d and str(d.get("code", "")) == "00000":
            data = d.get("data") or {}
            entries = data.get("openInterestList") or []
            if entries:
                try:
                    size = float(entries[0].get("size") or 0)
                    # Bitget OI is in base coin units; need price to convert
                    px = self._bitget_last_price(bg_sym)
                    if px and size > 0:
                        out["bitget"] = size * px
                except (TypeError, ValueError):
                    pass

        # OKX
        d = self._get(f"{OKX_BASE}/api/v5/public/open-interest",
                      {"instType": "SWAP", "instId": self._to_okx(base)})
        if d and str(d.get("code", "")) == "0":
            rows = d.get("data") or []
            if rows:
                try:
                    out["okx"] = float(rows[0].get("oiCcy") or 0) * self._okx_last_price(base) or 0.0
                    if not out["okx"]:
                        out["okx"] = float(rows[0].get("oiUsd") or 0)
                except (TypeError, ValueError):
                    pass

        out["total"] = sum(v for k, v in out.items() if k != "total")
        return out

    def _bitget_last_price(self, sym: str) -> Optional[float]:
        d = self._get(f"{BITGET_BASE}/api/v2/mix/market/ticker",
                      {"symbol": sym, "productType": "USDT-FUTURES"})
        if d and str(d.get("code", "")) == "00000":
            data = d.get("data") or []
            row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
            if row:
                try:
                    return float(row.get("lastPr") or 0) or None
                except (TypeError, ValueError):
                    return None
        return None

    def _okx_last_price(self, base: str) -> Optional[float]:
        d = self._get(f"{OKX_BASE}/api/v5/market/ticker",
                      {"instId": self._to_okx(base)})
        if d and str(d.get("code", "")) == "0":
            rows = d.get("data") or []
            if rows:
                try:
                    return float(rows[0].get("last") or 0) or None
                except (TypeError, ValueError):
                    return None
        return None

    def _kucoin_last_price(self, base: str) -> Optional[float]:
        d = self._get(f"{KUCOIN_FUTURES}/api/v1/contracts/{self._to_kucoin(base)}")
        if d and d.get("data"):
            try:
                return float(d["data"].get("markPrice") or 0) or None
            except (TypeError, ValueError):
                return None
        return None

    # ── Liquidation heatmap (OKX events + implied zones from OI) ─────────
    def liquidation_heatmap(
        self,
        symbol: str = "BTC",
        timeframe: str = "1d",
        exchange: str = "all",
    ) -> Optional[LiquidationHeatmap]:
        base = symbol.replace("USDT", "").replace("-USDT-SWAP", "")
        last_price = (self._okx_last_price(base) or self._kucoin_last_price(base)
                      or self._bitget_last_price(self._to_bitget(base)))
        if not last_price or last_price <= 0:
            return None

        levels: List[LiquidationLevel] = []

        # Component 1: real OKX liquidations
        levels.extend(self._okx_liquidation_levels(base, last_price))

        # Component 2: implied zones from cross-exchange aggregated OI
        oi_dict = self.aggregated_oi_now(base)
        total_oi = oi_dict.get("total", 0.0)
        if total_oi > 0:
            levels.extend(self._implied_liquidation_zones(last_price, total_oi))

        if not levels:
            return None

        levels = self._merge_close_levels(levels, tol_pct=0.1)
        levels.sort(key=lambda l: -l.notional_usdt)
        levels = levels[:50]
        levels.sort(key=lambda l: l.price)

        return LiquidationHeatmap(
            symbol=base, timeframe=timeframe,
            last_price=last_price, levels=levels,
        )

    def _okx_liquidation_levels(self, base: str, ref_price: float) -> List[LiquidationLevel]:
        out: List[LiquidationLevel] = []
        seen = 0
        before: Optional[str] = None

        for _ in range(3):
            params = {
                "instType": "SWAP",
                "uly": self._from_okx_uly(base),
                "limit": "100",
            }
            if before:
                params["before"] = before
            d = self._get(f"{OKX_BASE}/api/v5/public/liquidation-orders", params)
            if not d or str(d.get("code", "")) != "0":
                break
            rows = d.get("data") or []
            if not rows:
                break
            for row in rows:
                details = row.get("details") or [row]
                for detail in details:
                    try:
                        price = float(detail.get("bkPx") or detail.get("fillPx") or 0)
                        size_contracts = float(detail.get("sz") or 0)
                        side = (detail.get("side") or detail.get("posSide") or "").lower()
                    except (TypeError, ValueError):
                        continue
                    if price <= 0 or size_contracts <= 0:
                        continue
                    notional = price * size_contracts
                    if notional < 1000:
                        continue
                    if side in ("sell", "long"):
                        liq_side = "long"
                    elif side in ("buy", "short"):
                        liq_side = "short"
                    else:
                        liq_side = "short" if price > ref_price else "long"
                    out.append(LiquidationLevel(
                        price=price, notional_usdt=notional, side=liq_side,
                    ))
                    seen += 1
            last_ts = rows[-1].get("uTime") or rows[-1].get("ts")
            if not last_ts:
                break
            before = str(last_ts)
            if seen >= 300:
                break
        return out

    def _implied_liquidation_zones(self, ref_price: float, total_oi_usdt: float) -> List[LiquidationLevel]:
        out: List[LiquidationLevel] = []
        half_oi = total_oi_usdt / 2
        for leverage, weight in LEVERAGE_TIERS:
            liq_drop = (1.0 / leverage) - MAINTENANCE_MARGIN
            long_liq = ref_price * (1 - liq_drop)
            short_liq = ref_price * (1 + liq_drop)
            notional = half_oi * weight
            out.append(LiquidationLevel(price=long_liq, notional_usdt=notional, side="long"))
            out.append(LiquidationLevel(price=short_liq, notional_usdt=notional, side="short"))
        return out

    def _merge_close_levels(
        self, levels: List[LiquidationLevel], tol_pct: float = 0.1,
    ) -> List[LiquidationLevel]:
        if not levels:
            return []
        levels = sorted(levels, key=lambda l: l.price)
        merged: List[LiquidationLevel] = []
        for lvl in levels:
            if merged:
                last = merged[-1]
                pct_diff = abs(lvl.price - last.price) / last.price * 100
                if pct_diff < tol_pct and lvl.side == last.side:
                    total = last.notional_usdt + lvl.notional_usdt
                    weighted = (last.price * last.notional_usdt +
                                lvl.price * lvl.notional_usdt) / max(total, 1)
                    merged[-1] = LiquidationLevel(
                        price=weighted, notional_usdt=total, side=last.side,
                    )
                    continue
            merged.append(lvl)
        return merged

    # ── Cross-exchange funding aggregate ──────────────────────────────────
    def funding_aggregated(self, symbol: str = "BTC") -> Optional[FundingSnapshot]:
        base = symbol.replace("USDT", "").replace("-USDT-SWAP", "")
        rates: Dict[str, float] = {}

        # KuCoin
        kc_sym = self._to_kucoin(base)
        d = self._get(f"{KUCOIN_FUTURES}/api/v1/funding-rate/{kc_sym}/current")
        if d and d.get("data"):
            try:
                v = d["data"].get("value")
                if v is not None:
                    rates["kucoin"] = float(v)
            except (TypeError, ValueError):
                pass

        # Bitget
        bg_sym = self._to_bitget(base)
        d = self._get(f"{BITGET_BASE}/api/v2/mix/market/current-fund-rate",
                      {"symbol": bg_sym, "productType": "USDT-FUTURES"})
        if d and str(d.get("code", "")) == "00000":
            data = d.get("data") or []
            row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
            if row and row.get("fundingRate") is not None:
                try:
                    rates["bitget"] = float(row["fundingRate"])
                except (TypeError, ValueError):
                    pass

        # OKX
        d = self._get(f"{OKX_BASE}/api/v5/public/funding-rate",
                      {"instId": self._to_okx(base)})
        if d and str(d.get("code", "")) == "0":
            rows = d.get("data") or []
            if rows and rows[0].get("fundingRate") is not None:
                try:
                    rates["okx"] = float(rows[0]["fundingRate"])
                except (TypeError, ValueError):
                    pass

        if not rates:
            return None

        values = list(rates.values())
        avg_f = sum(values) / len(values)
        max_f = max(values)
        min_f = min(values)
        spread_bps = (max_f - min_f) * 10000.0

        # OI-weighted using our OI snapshot
        oi = self.aggregated_oi_now(base)
        total_oi = oi.get("total", 0.0)
        if total_oi > 0:
            num = sum(rates.get(ex, avg_f) * oi.get(ex, 0.0)
                      for ex in oi if ex != "total")
            oi_w = num / total_oi if total_oi > 0 else avg_f
        else:
            oi_w = avg_f

        return FundingSnapshot(
            symbol=base,
            avg_funding=avg_f,
            max_funding=max_f,
            min_funding=min_f,
            spread_bps=spread_bps,
            oi_weighted_funding=oi_w,
            asof_ms=int(time.time() * 1000),
            exchanges=rates,
        )

    # ── Long/short history (Bitget + OKX rubik) ──────────────────────────
    def longshort_history(
        self,
        symbol: str = "BTC",
        interval: str = "1h",
        limit: int = 24,
    ) -> List[LongShortSnapshot]:
        """
        Bitget: account-long-short + position-long-short (free, GitHub-runner-friendly).
        OKX rubik: long-short-account-ratio + taker-volume-contract.
        """
        base = symbol.replace("USDT", "").replace("-USDT-SWAP", "")
        snaps: Dict[int, LongShortSnapshot] = {}

        # Bitget account L/S — period like '5m'/'15m'/'1h'/'4h'/'1d'
        bg_sym = self._to_bitget(base)
        bg_period = interval if interval in ("5m", "15m", "30m", "1h", "4h", "12h", "1d") else "1h"
        d = self._get(f"{BITGET_BASE}/api/v2/mix/market/account-long-short",
                      {"symbol": bg_sym, "period": bg_period})
        if d and str(d.get("code", "")) == "00000":
            for row in (d.get("data") or [])[-limit:]:
                try:
                    ts = int(row.get("ts") or row.get("timestamp") or 0)
                    if not ts:
                        continue
                    long_acc = float(row.get("longAccount") or row.get("longRatio") or 0)
                    short_acc = float(row.get("shortAccount") or row.get("shortRatio") or 0)
                    if short_acc > 0:
                        snaps.setdefault(ts, LongShortSnapshot(symbol=base, asof_ms=ts))
                        snaps[ts].global_account_ratio = long_acc / short_acc
                except (TypeError, ValueError):
                    continue

        # Bitget position L/S
        d = self._get(f"{BITGET_BASE}/api/v2/mix/market/position-long-short",
                      {"symbol": bg_sym, "period": bg_period})
        if d and str(d.get("code", "")) == "00000":
            for row in (d.get("data") or [])[-limit:]:
                try:
                    ts = int(row.get("ts") or row.get("timestamp") or 0)
                    if not ts:
                        continue
                    long_pos = float(row.get("longPosition") or row.get("longRatio") or 0)
                    short_pos = float(row.get("shortPosition") or row.get("shortRatio") or 0)
                    if short_pos > 0:
                        snaps.setdefault(ts, LongShortSnapshot(symbol=base, asof_ms=ts))
                        snaps[ts].top_position_ratio = long_pos / short_pos
                except (TypeError, ValueError):
                    continue

        # OKX rubik L/S — uses base coin, not pair
        okx_period = self._okx_period(interval)
        d = self._get(f"{OKX_BASE}/api/v5/rubik/stat/contracts/long-short-account-ratio",
                      {"ccy": base.upper(), "period": okx_period, "limit": str(limit)})
        if d and str(d.get("code", "")) == "0":
            for row in (d.get("data") or []):
                try:
                    ts = int(row[0])
                    ratio = float(row[1])
                    snaps.setdefault(ts, LongShortSnapshot(symbol=base, asof_ms=ts))
                    snaps[ts].top_account_ratio = ratio
                except (TypeError, ValueError, IndexError):
                    continue

        # OKX rubik taker volume → ratio
        d = self._get(f"{OKX_BASE}/api/v5/rubik/stat/taker-volume-contract",
                      {"instId": self._to_okx(base),
                       "period": okx_period, "limit": str(limit), "unit": "usdt"})
        if d and str(d.get("code", "")) == "0":
            for row in (d.get("data") or []):
                try:
                    ts = int(row[0])
                    sell_vol = float(row[1])
                    buy_vol = float(row[2])
                    if sell_vol > 0:
                        snaps.setdefault(ts, LongShortSnapshot(symbol=base, asof_ms=ts))
                        snaps[ts].taker_buy_sell_ratio = buy_vol / sell_vol
                except (TypeError, ValueError, IndexError):
                    continue

        return sorted(snaps.values(), key=lambda s: s.asof_ms)

    @staticmethod
    def _okx_period(p: str) -> str:
        return {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H",
                "2h": "2H", "4h": "4H", "1d": "1D"}.get(p, "1H")

    # ── Aggregated OI history ────────────────────────────────────────────
    def aggregated_oi_history(
        self,
        symbol: str = "BTC",
        interval: str = "1h",
        limit: int = 168,
    ) -> List[Dict[str, Any]]:
        """OKX rubik /open-interest-volume — works on Actions, free."""
        base = symbol.replace("USDT", "").replace("-USDT-SWAP", "")
        d = self._get(f"{OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-volume",
                      {"instId": self._to_okx(base),
                       "period": self._okx_period(interval),
                       "limit": str(min(limit, 100))})
        if not d or str(d.get("code", "")) != "0":
            return []
        rows = d.get("data") or []
        # Convert to list-of-dict
        return [
            {"timestamp": int(row[0]), "oi_usd": float(row[1]),
             "vol_usd": float(row[2]) if len(row) > 2 else 0.0}
            for row in rows
        ]


# Backward-compat alias
AzalystClient = DerivedDataClient
