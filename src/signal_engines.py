"""
src/signal_engines.py — Leading-indicator signal engines.

The original scanner has ONE engine: an ML model trained on Binance snapshots.
World-class crypto futures alpha comes from independent confirmation across
multiple data types — that way no single noisy signal causes a bad trade.

Five engines run alongside the ML model:

    1. LiquidationProximityEngine   — finds the leverage cluster the price
                                       is most likely to test next
    2. FundingExtremeEngine         — flags overcrowded / squeezed funding
    3. LongShortExtremeEngine       — top-trader vs retail divergence
    4. BasisEngine                  — perp vs spot premium / discount
    5. OIDeltaEngine                — OI direction + price quadrant

Each engine emits a SignalCard with a direction (LONG/SHORT/NEUTRAL), a
0–100 strength score, and a reason string. SignalFuser (separate file)
merges the cards per symbol into a Tier A/B/C consensus.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.derived_data import (
    DerivedDataClient,
    FundingSnapshot,
    LiquidationHeatmap,
    LongShortSnapshot,
)
# Backward-compat alias so engine code below keeps using the old name
AzalystClient = DerivedDataClient

log = logging.getLogger("azalyst.engines")


@dataclass
class SignalCard:
    """One engine's per-symbol verdict."""
    symbol: str
    engine: str
    direction: str           # 'LONG' / 'SHORT' / 'NEUTRAL'
    strength: float          # 0–100
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
class LiquidationProximityEngine:
    """
    Maps the liquidation heatmap to a directional signal.

    Logic:
      - If a large SHORT-liq cluster sits within 1–3% above current price,
        longs have squeeze fuel → LONG signal (squeeze higher to liquidate
        the over-leveraged shorts above).
      - If a large LONG-liq cluster sits within 1–3% below, shorts have
        cascade fuel → SHORT signal.
      - The ratio of cluster magnitudes (above vs below) controls strength.
      - Clusters TOO close (<0.4%) are penalized — usually a distribution
        zone, not a clean target.
    """

    NEAR_PCT = 3.0          # consider clusters within this % of price
    MIN_PCT = 0.4           # ignore clusters closer than this
    MIN_CLUSTER_USDT = 10_000_000   # ignore tiny clusters (<$10M notional)

    def __init__(self, client: AzalystClient):
        self.client = client

    def run(self, symbol: str = "BTC", timeframe: str = "1d") -> Optional[SignalCard]:
        if not self.client.enabled:
            return None
        heatmap = self.client.liquidation_heatmap(symbol, timeframe)
        if not heatmap or not heatmap.last_price or not heatmap.levels:
            return None

        ref = heatmap.last_price
        above = heatmap.levels_above(ref, self.NEAR_PCT)
        below = heatmap.levels_below(ref, self.NEAR_PCT)

        # Filter clusters that are too close or too small
        def _filt(levels, ref_price, sign):
            out = []
            for lvl in levels:
                pct_dist = abs(lvl.price - ref_price) / ref_price * 100
                if pct_dist < self.MIN_PCT:
                    continue
                if lvl.notional_usdt < self.MIN_CLUSTER_USDT:
                    continue
                out.append((lvl, pct_dist))
            return out

        above_f = _filt(above, ref, +1)
        below_f = _filt(below, ref, -1)

        # Sum-of-cluster magnitudes weighted by inverse-distance (closer = more pull)
        def _weighted_pull(filtered):
            return sum(
                lvl.notional_usdt / max(pct, 0.5) for lvl, pct in filtered
            )

        pull_up = _weighted_pull(above_f)
        pull_down = _weighted_pull(below_f)

        if pull_up == 0 and pull_down == 0:
            return SignalCard(
                symbol=symbol, engine="liq_proximity",
                direction="NEUTRAL", strength=0.0,
                reason="No significant liquidation clusters within 3% of price.",
                metrics={"last_price": ref, "pull_up": 0, "pull_down": 0},
            )

        ratio = pull_up / max(pull_down, 1)
        if ratio >= 1.5:
            direction = "LONG"
            strength = min(60 + 20 * math.log10(ratio + 1), 95)
            top = max(above_f, key=lambda x: x[0].notional_usdt)[0]
            reason = (
                f"Short-liq cluster of ${top.notional_usdt/1e6:,.0f}M at "
                f"${top.price:,.2f} ({(top.price/ref-1)*100:+.2f}%). "
                f"Squeeze fuel to the upside."
            )
        elif ratio <= 0.667:
            direction = "SHORT"
            strength = min(60 + 20 * math.log10(1/ratio + 1), 95)
            top = max(below_f, key=lambda x: x[0].notional_usdt)[0]
            reason = (
                f"Long-liq cluster of ${top.notional_usdt/1e6:,.0f}M at "
                f"${top.price:,.2f} ({(top.price/ref-1)*100:+.2f}%). "
                f"Cascade risk to the downside."
            )
        else:
            direction = "NEUTRAL"
            strength = 0.0
            reason = "Liquidation pull is symmetric within ±3%."

        return SignalCard(
            symbol=symbol, engine="liq_proximity",
            direction=direction, strength=round(strength, 1),
            reason=reason,
            metrics={
                "last_price": ref,
                "pull_up": pull_up,
                "pull_down": pull_down,
                "ratio": ratio,
            },
        )


# ──────────────────────────────────────────────────────────────────────────
class FundingExtremeEngine:
    """
    Crowded-side detector via cross-exchange funding.

    - Sustained positive funding > 0.05% per 8h = longs paying through the
      nose to stay long → contrarian SHORT setup.
    - Negative funding into a price rally = stubborn shorts being squeezed
      (longs being PAID to hold) → LONG signal continues.
    - Cross-exchange spread (max - min) > 30bps = inter-venue arb signal,
      usually leads a snap-back.
    """

    OVERHEATED_BPS = 5.0   # per-funding-period basis points (= 0.05%)
    SQUEEZE_BPS = -3.0
    SPREAD_TRIGGER_BPS = 30.0

    def __init__(self, client: AzalystClient):
        self.client = client

    def run(self, symbol: str = "BTC") -> Optional[SignalCard]:
        if not self.client.enabled:
            return None
        snap = self.client.funding_aggregated(symbol)
        if snap is None:
            return None

        avg_bps = snap.avg_funding * 10000.0
        oiw_bps = snap.oi_weighted_funding * 10000.0
        spread = snap.spread_bps

        direction = "NEUTRAL"
        strength = 0.0
        reason = f"Funding ~{avg_bps:.1f}bps (OI-weighted {oiw_bps:.1f}bps), spread {spread:.1f}bps."

        if oiw_bps >= self.OVERHEATED_BPS:
            direction = "SHORT"
            strength = min(40 + (oiw_bps - self.OVERHEATED_BPS) * 4, 85)
            reason = (
                f"OI-weighted funding {oiw_bps:.1f}bps — longs overcrowded; "
                f"contrarian short setup. Spread {spread:.1f}bps."
            )
        elif oiw_bps <= self.SQUEEZE_BPS:
            direction = "LONG"
            strength = min(40 + abs(oiw_bps - self.SQUEEZE_BPS) * 4, 85)
            reason = (
                f"OI-weighted funding {oiw_bps:.1f}bps — shorts paying longs; "
                f"squeeze setup. Spread {spread:.1f}bps."
            )

        # Cross-exchange dispersion adds confidence
        if spread >= self.SPREAD_TRIGGER_BPS and direction != "NEUTRAL":
            strength = min(strength + 10, 95)
            reason += " High cross-venue dispersion confirms imbalance."

        return SignalCard(
            symbol=symbol, engine="funding_extreme",
            direction=direction, strength=round(strength, 1), reason=reason,
            metrics={
                "avg_funding_bps": avg_bps,
                "oi_weighted_funding_bps": oiw_bps,
                "spread_bps": spread,
                "max_exchange_funding_bps": snap.max_funding * 10000.0,
                "min_exchange_funding_bps": snap.min_funding * 10000.0,
            },
        )


# ──────────────────────────────────────────────────────────────────────────
class LongShortExtremeEngine:
    """
    Top-trader vs retail divergence.

    - Retail crowded long (global L/S > 2.5) but top traders flipped short
      (top L/S < 0.9) = classic smart-money fade → SHORT.
    - Inverse setup → LONG.
    - Falling top-trader L/S over recent hours = early de-risking signal.
    """

    RETAIL_CROWDED = 2.2
    SMART_MONEY_DIVERGENCE = 0.4    # top-account ratio differs from retail by this much

    def __init__(self, client: AzalystClient):
        self.client = client

    def run(self, symbol: str = "BTC", interval: str = "1h") -> Optional[SignalCard]:
        if not self.client.enabled:
            return None
        history = self.client.longshort_history(symbol, interval, limit=24)
        if not history:
            return None
        latest = history[-1]
        if latest.global_account_ratio is None or latest.top_account_ratio is None:
            return None

        retail = latest.global_account_ratio
        top = latest.top_account_ratio
        delta = top - retail

        # Recent slope of top-trader ratio
        recent_top = [s.top_account_ratio for s in history[-6:] if s.top_account_ratio is not None]
        slope = (recent_top[-1] - recent_top[0]) if len(recent_top) >= 2 else 0.0

        direction = "NEUTRAL"
        strength = 0.0
        reason = f"Retail L/S {retail:.2f}, top trader L/S {top:.2f}, slope {slope:+.2f}."

        if retail >= self.RETAIL_CROWDED and delta <= -self.SMART_MONEY_DIVERGENCE:
            direction = "SHORT"
            strength = min(50 + abs(delta) * 30 + max(-slope, 0) * 20, 90)
            reason = (
                f"Retail crowded long ({retail:.2f}), top traders less long ({top:.2f}). "
                f"Smart money fade."
            )
        elif retail <= 1.0 and delta >= self.SMART_MONEY_DIVERGENCE:
            direction = "LONG"
            strength = min(50 + abs(delta) * 30 + max(slope, 0) * 20, 90)
            reason = (
                f"Retail short-leaning ({retail:.2f}), top traders long ({top:.2f}). "
                f"Smart money accumulation."
            )

        # Taker flow tilt
        if latest.taker_buy_sell_ratio is not None:
            if latest.taker_buy_sell_ratio >= 1.15 and direction == "LONG":
                strength = min(strength + 8, 95)
                reason += f" Aggressive taker-buy flow ({latest.taker_buy_sell_ratio:.2f})."
            elif latest.taker_buy_sell_ratio <= 0.85 and direction == "SHORT":
                strength = min(strength + 8, 95)
                reason += f" Aggressive taker-sell flow ({latest.taker_buy_sell_ratio:.2f})."

        return SignalCard(
            symbol=symbol, engine="ls_extreme",
            direction=direction, strength=round(strength, 1), reason=reason,
            metrics={
                "retail_ls": retail,
                "top_ls": top,
                "delta": delta,
                "slope_6h": slope,
                "taker_buy_sell": latest.taker_buy_sell_ratio or 0.0,
            },
        )


# ──────────────────────────────────────────────────────────────────────────
class BasisEngine:
    """
    Perp vs spot premium signal. Doesn't need Azalyst — uses Binance
    public spot + perp price you already fetch in collector.py.

    - Perp > spot by >+25bps = retail-leveraged FOMO; mean-reverts.
    - Perp < spot by >-25bps = institutional spot bid w/ perp short hedge;
      bullish for spot continuation but bearish for perp price.
    """
    PREMIUM_TRIGGER_BPS = 25.0

    def __init__(self, fetch_spot_price, fetch_perp_price):
        # Caller supplies the two fetch functions so we don't duplicate
        # the existing Binance HTTP layer.
        self._spot = fetch_spot_price
        self._perp = fetch_perp_price

    def run(self, symbol: str) -> Optional[SignalCard]:
        try:
            spot = self._spot(symbol)
            perp = self._perp(symbol)
        except Exception as exc:
            log.debug("Basis fetch failed for %s: %s", symbol, exc)
            return None
        if not spot or not perp:
            return None
        basis_bps = (perp - spot) / spot * 10000.0

        direction = "NEUTRAL"
        strength = 0.0
        reason = f"Perp {perp:.4f} vs spot {spot:.4f}: {basis_bps:+.1f}bps basis."

        if basis_bps >= self.PREMIUM_TRIGGER_BPS:
            direction = "SHORT"
            strength = min(30 + (basis_bps - self.PREMIUM_TRIGGER_BPS) * 1.5, 80)
            reason = f"Perp premium of {basis_bps:.1f}bps over spot — leveraged FOMO; fade."
        elif basis_bps <= -self.PREMIUM_TRIGGER_BPS:
            direction = "LONG"
            strength = min(30 + (abs(basis_bps) - self.PREMIUM_TRIGGER_BPS) * 1.5, 80)
            reason = f"Perp discount of {abs(basis_bps):.1f}bps to spot — institutional bid."

        return SignalCard(
            symbol=symbol, engine="basis",
            direction=direction, strength=round(strength, 1), reason=reason,
            metrics={"basis_bps": basis_bps, "spot": spot, "perp": perp},
        )


# ──────────────────────────────────────────────────────────────────────────
class OIDeltaEngine:
    """
    OI Δ + price quadrant — distinguishes shorts loading vs short cover
    vs long capitulation. Uses data already in scan_symbol() rows.

    Quadrant:                         price up      price down
                              ─────────────────────────────────────
                       OI up │ healthy long trend │ shorts loading │
                              ─────────────────────────────────────
                       OI dn │ short cover rally  │ long capitulate│
                              ─────────────────────────────────────
    """

    def run(self, scan_row: Dict) -> Optional[SignalCard]:
        symbol = scan_row.get("symbol", "?")
        oi_chg = scan_row.get("oi_change_pct_1h")
        px_chg = scan_row.get("price_change_pct_24h")
        if oi_chg is None or px_chg is None:
            return None

        try:
            oi = float(oi_chg)
            px = float(px_chg)
        except (TypeError, ValueError):
            return None

        direction = "NEUTRAL"
        strength = 0.0
        reason = f"OI {oi:+.2f}% / price {px:+.2f}% (no clear quadrant)."

        if abs(oi) < 1.0 or abs(px) < 1.0:
            return SignalCard(symbol, "oi_delta", direction, 0.0, reason,
                              {"oi_chg": oi, "px_chg": px})

        if oi > 0 and px > 0:
            direction = "LONG"
            strength = min(40 + abs(oi) * 3 + abs(px), 80)
            reason = f"OI +{oi:.1f}%, price +{px:.1f}% — healthy long trend."
        elif oi > 0 and px < 0:
            direction = "SHORT"
            strength = min(40 + abs(oi) * 3 + abs(px), 80)
            reason = f"OI +{oi:.1f}%, price {px:.1f}% — fresh shorts loading."
        elif oi < 0 and px > 0:
            direction = "NEUTRAL"     # short-cover rally — don't chase
            strength = 25
            reason = f"OI {oi:.1f}%, price +{px:.1f}% — short-cover rally; weak follow-through."
        elif oi < 0 and px < 0:
            direction = "NEUTRAL"     # capitulation — don't catch falling knife
            strength = 25
            reason = f"OI {oi:.1f}%, price {px:.1f}% — long capitulation; wait for base."

        return SignalCard(
            symbol=symbol, engine="oi_delta",
            direction=direction, strength=round(strength, 1), reason=reason,
            metrics={"oi_chg": oi, "px_chg": px},
        )
