"""
src/signal_fusion.py — Cross-engine consensus / Tier A/B/C ranking.

Same pattern proven on the Azalyst ETF scanner: independent engines,
direction vote, tier classification, divergence flag. Tuned for crypto:

  Tier A — ≥4 engines (out of 6 incl. ML) agree, no divergence
  Tier B — 3 engines agree, ≤1 dissent
  Tier C — 2 engines agree (research only, smaller size)
  divergent — engines split — usually means a regime change is
              underway; flag for manual review, do not auto-trade.

Signals decay in MINUTES in crypto so this runs every cycle, not
once per day like the ETF scanner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.signal_engines import SignalCard

log = logging.getLogger("azalyst.fusion")


@dataclass
class FusedCryptoSignal:
    symbol: str
    direction: str            # LONG / SHORT
    consensus_tier: str       # A / B / C
    fused_score: float        # 0–100
    engines_long: List[str] = field(default_factory=list)
    engines_short: List[str] = field(default_factory=list)
    engines_neutral: List[str] = field(default_factory=list)
    divergent: bool = False
    cards: List[SignalCard] = field(default_factory=list)
    ml_probability: Optional[float] = None
    ml_direction: Optional[str] = None
    summary: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "consensus_tier": self.consensus_tier,
            "fused_score": self.fused_score,
            "divergent": self.divergent,
            "engines_long": self.engines_long,
            "engines_short": self.engines_short,
            "engines_neutral": self.engines_neutral,
            "ml_probability": self.ml_probability,
            "ml_direction": self.ml_direction,
            "summary": self.summary,
            "reasons": [{"engine": c.engine, "direction": c.direction,
                         "strength": c.strength, "reason": c.reason}
                        for c in self.cards],
            "metrics": self.metrics,
        }


# Engine weight when computing fused score (sums to 1.0)
ENGINE_WEIGHTS: Dict[str, float] = {
    "liq_proximity":   0.28,    # highest — liq heatmap is the crypto edge
    "ml_main":         0.22,    # XGBoost is generic but broadly useful
    "ls_extreme":      0.16,
    "funding_extreme": 0.14,
    "basis":           0.10,
    "oi_delta":        0.10,
}


class CryptoSignalFuser:

    # Threshold for ML probability to count as a directional vote
    ML_LONG_THRESHOLD = 0.62
    ML_SHORT_THRESHOLD = 0.38   # symmetric on the other side

    def fuse_one(
        self,
        symbol: str,
        cards: List[SignalCard],
        ml_probability: Optional[float] = None,
    ) -> Optional[FusedCryptoSignal]:
        # Convert ML probability into a virtual SignalCard so it votes too
        ml_dir = None
        ml_card = None
        if ml_probability is not None:
            if ml_probability >= self.ML_LONG_THRESHOLD:
                ml_dir = "LONG"
                ml_card = SignalCard(
                    symbol=symbol, engine="ml_main",
                    direction="LONG",
                    strength=round((ml_probability - 0.5) * 200, 1),
                    reason=f"XGBoost probability {ml_probability:.2%} ≥ {self.ML_LONG_THRESHOLD:.0%}.",
                )
            elif ml_probability <= self.ML_SHORT_THRESHOLD:
                ml_dir = "SHORT"
                ml_card = SignalCard(
                    symbol=symbol, engine="ml_main",
                    direction="SHORT",
                    strength=round((0.5 - ml_probability) * 200, 1),
                    reason=f"XGBoost probability {ml_probability:.2%} ≤ {self.ML_SHORT_THRESHOLD:.0%}.",
                )
            else:
                ml_card = SignalCard(
                    symbol=symbol, engine="ml_main",
                    direction="NEUTRAL", strength=0.0,
                    reason=f"XGBoost probability {ml_probability:.2%} is in neutral zone.",
                )

        all_cards = [c for c in cards if c is not None]
        if ml_card is not None:
            all_cards.append(ml_card)

        if not all_cards:
            return None

        long_engines = [c.engine for c in all_cards if c.direction == "LONG"]
        short_engines = [c.engine for c in all_cards if c.direction == "SHORT"]
        neutral_engines = [c.engine for c in all_cards if c.direction == "NEUTRAL"]

        if not long_engines and not short_engines:
            return None

        # Direction = side with more votes; ties go to higher cumulative strength
        long_strength = sum(c.strength for c in all_cards if c.direction == "LONG")
        short_strength = sum(c.strength for c in all_cards if c.direction == "SHORT")
        if len(long_engines) > len(short_engines) or (
            len(long_engines) == len(short_engines) and long_strength >= short_strength
        ):
            direction = "LONG"
            agree = long_engines
            dissent = short_engines
        else:
            direction = "SHORT"
            agree = short_engines
            dissent = long_engines

        divergent = len(agree) >= 2 and len(dissent) >= 2

        # Consensus tier
        if len(agree) >= 4 and not divergent:
            tier = "A"
        elif len(agree) >= 3:
            tier = "B"
        elif len(agree) == 2:
            tier = "C"
        else:
            return None

        # Fused score: weighted-average strength of agreeing engines, with
        # a small boost for tier A and a heavy penalty for divergence
        weighted_sum = 0.0
        weight_total = 0.0
        for c in all_cards:
            if c.direction != direction:
                continue
            w = ENGINE_WEIGHTS.get(c.engine, 0.05)
            weighted_sum += c.strength * w
            weight_total += w

        base = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        if tier == "A":
            base = min(base * 1.15, 100)
        if tier == "C":
            base *= 0.85
        if divergent:
            base *= 0.7
        fused = round(base, 1)

        # Build summary
        reasons_compact = " | ".join(
            f"{c.engine}={c.direction[0]}{int(c.strength)}"
            for c in all_cards
        )
        summary = (
            f"{symbol} {direction} Tier-{tier} score {fused:.1f}/100 "
            f"({len(agree)} engines agree"
            + (f", {len(dissent)} dissent" if dissent else "")
            + f"). {reasons_compact}"
        )

        return FusedCryptoSignal(
            symbol=symbol,
            direction=direction,
            consensus_tier=tier,
            fused_score=fused,
            engines_long=long_engines,
            engines_short=short_engines,
            engines_neutral=neutral_engines,
            divergent=divergent,
            cards=all_cards,
            ml_probability=ml_probability,
            ml_direction=ml_dir,
            summary=summary,
            metrics={
                "long_strength_total": round(long_strength, 1),
                "short_strength_total": round(short_strength, 1),
            },
        )

    def fuse_many(
        self,
        per_symbol_cards: Dict[str, List[SignalCard]],
        ml_probabilities: Optional[Dict[str, float]] = None,
    ) -> List[FusedCryptoSignal]:
        ml_probabilities = ml_probabilities or {}
        out: List[FusedCryptoSignal] = []
        for symbol, cards in per_symbol_cards.items():
            fused = self.fuse_one(symbol, cards, ml_probabilities.get(symbol))
            if fused is not None:
                out.append(fused)
        # Tier A first, then by score
        tier_order = {"A": 0, "B": 1, "C": 2}
        out.sort(key=lambda s: (tier_order.get(s.consensus_tier, 3), -s.fused_score))
        return out
