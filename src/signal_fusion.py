"""
Dynamic signal fusion: weight engines based on historical performance,
replace fixed tier thresholds with continuous score + entropy penalty.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import LOGS_DIR
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


class DynamicSignalFuser:
    def __init__(self, weight_reg = None):
        # Logistic regression fits: engine_strengths -> probability of win
        self.reg = weight_reg or LogisticRegression(fit_intercept=False, max_iter=1000)
        # Default fallback weights if no history exists
        self.default_weights = {
            'liq_proximity': 0.28,
            'ml_main': 0.22,
            'ls_extreme': 0.16,
            'funding_extreme': 0.14,
            'basis': 0.10,
            'oi_delta': 0.10
        }

    def train_weights(self, history_file: Path = LOGS_DIR / 'fused_history.jsonl'):
        """Load past fused signals and their outcome to fit engine strength -> win."""
        if not history_file.exists():
            return
        records = []
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try: records.append(json.loads(line))
                    except: pass
        except Exception as e:
            log.warning(f"Failed to read fusion history: {e}")
            return
            
        if len(records) < 50:
            return
            
        df = pd.DataFrame(records)
        # Assume df has columns for each engine's strength and the final 'outcome' (1=win, 0=loss)
        # We derive outcome from future_return_pct if available
        required = ['outcome']
        engines = ['liq_proximity', 'ml_main', 'ls_extreme', 'funding_extreme', 'basis', 'oi_delta']
        
        # Check if we have engine columns
        available_engines = [e for e in engines if e in df.columns]
        if not available_engines or 'outcome' not in df.columns:
            return
            
        X = df[available_engines].fillna(0)
        y = df['outcome'].astype(int)
        
        if y.nunique() < 2:
            return
            
        self.reg.fit(X, y)
        coefs = dict(zip(available_engines, self.reg.coef_[0].tolist()))
        log.info(f"Dynamic fusion weights updated: {coefs}")

    def fuse_one(self, symbol, cards, ml_prob):
        # Convert cards to strength per engine
        strengths = {c.engine: (c.strength if c.direction == 'LONG' else -c.strength) 
                    for c in cards if c.direction != 'NEUTRAL'}
        
        # ML probability input (shifted to -100 to 100 range to match engine strengths)
        ml_strength = (ml_prob - 0.5) * 200 if ml_prob is not None else 0.0
        strengths['ml_main'] = ml_strength

        # Calculate base score
        if hasattr(self.reg, 'coef_'):
            # Use logistic regression prediction
            engines = ['liq_proximity', 'ml_main', 'ls_extreme', 'funding_extreme', 'basis', 'oi_delta']
            feature_vec = [strengths.get(e, 0.0) for e in engines]
            # Prob of 1 (win)
            prob_win = self.reg.predict_proba([feature_vec])[0, 1]
            dynamic_score = prob_win * 100
        else:
            # Fallback: weighted average
            weighted_sum = sum(abs(s) * self.default_weights.get(e, 0.05) for e, s in strengths.items())
            dynamic_score = weighted_sum

        # Determine overall direction
        long_cards = [c for c in cards if c.direction == 'LONG']
        short_cards = [c for c in cards if c.direction == 'SHORT']
        if ml_prob is not None:
            if ml_prob >= 0.62: long_cards.append(None) # Virtual card
            elif ml_prob <= 0.38: short_cards.append(None)

        if len(long_cards) > len(short_cards):
            direction = 'LONG'
        elif len(short_cards) > len(long_cards):
            direction = 'SHORT'
        else:
            # Tie break on strength
            long_s = sum(c.strength for c in cards if c.direction == 'LONG')
            short_s = sum(c.strength for c in cards if c.direction == 'SHORT')
            direction = 'LONG' if long_s >= short_s else 'SHORT'

        # Entropy penalty for divergence
        total_votes = len(long_cards) + len(short_cards)
        if total_votes > 1:
            p_long = len(long_cards) / total_votes
            p_short = len(short_cards) / total_votes
            # Binary entropy
            if p_long == 0 or p_short == 0:
                entropy_norm = 0
            else:
                entropy = -(p_long * np.log2(p_long) + p_short * np.log2(p_short))
                entropy_norm = entropy # max is 1.0 for binary
            
            agreement_factor = 1.0 - (0.5 * entropy_norm) # Max 50% penalty for full 50/50 split
        else:
            agreement_factor = 1.0

        final_score = round(dynamic_score * agreement_factor, 1)

        # Tiering logic
        agree_count = len(long_cards) if direction == 'LONG' else len(short_cards)
        divergent = (agreement_factor < 0.7)
        
        if final_score >= 70 and agree_count >= 4 and not divergent:
            tier = 'A'
        elif final_score >= 45 and agree_count >= 3:
            tier = 'B'
        elif final_score >= 25:
            tier = 'C'
        else:
            return None

        # Build summary
        summary = f"{symbol} {direction} Tier-{tier} Score:{final_score} (Agree:{agree_count}, Factor:{agreement_factor:.2f})"
        
        return FusedCryptoSignal(
            symbol=symbol, direction=direction, consensus_tier=tier,
            fused_score=final_score, divergent=divergent,
            engines_long=[c.engine for c in cards if c.direction == 'LONG'],
            engines_short=[c.engine for c in cards if c.direction == 'SHORT'],
            engines_neutral=[c.engine for c in cards if c.direction == 'NEUTRAL'],
            cards=cards, ml_probability=ml_prob,
            ml_direction="LONG" if ml_prob and ml_prob >= 0.5 else "SHORT",
            summary=summary,
            metrics={"agreement_factor": agreement_factor, "base_dynamic_score": dynamic_score}
        )

    def fuse_many(self, per_symbol_cards: Dict[str, List[SignalCard]], ml_probabilities: Optional[Dict[str, float]] = None) -> List[FusedCryptoSignal]:
        ml_probabilities = ml_probabilities or {}
        out = []
        for symbol, cards in per_symbol_cards.items():
            fused = self.fuse_one(symbol, cards, ml_probabilities.get(symbol))
            if fused:
                out.append(fused)
        
        tier_order = {"A": 0, "B": 1, "C": 2}
        out.sort(key=lambda s: (tier_order.get(s.consensus_tier, 3), -s.fused_score))
        return out
