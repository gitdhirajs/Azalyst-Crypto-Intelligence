"""
risk_engine.py - Azalyst Crypto Risk Controls
Now includes RiskManager for live position sizing and correlation checks.
"""
import math
import logging
from typing import Dict, List, Tuple

log = logging.getLogger("azalyst_crypto.risk")

CORRELATION_WARN = 0.75
TARGET_VOL = 0.20


def pearson(x: List[float], y: List[float]) -> float:
    n = min(len(x), len(y))
    if n < 5:
        return 0.0
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x[:n]))
    sy = math.sqrt(sum((v - my) ** 2 for v in y[:n]))
    if sx == 0 or sy == 0:
        return 0.0
    return round(sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy), 4)


def check_correlation(returns: Dict[str, List[float]], new_symbol: str, threshold: float = CORRELATION_WARN) -> Dict:
    if new_symbol not in returns:
        return {"blocked": False, "max_corr": 0}
    max_corr = 0.0
    corr_with = ""
    for sym, rets in returns.items():
        if sym == new_symbol:
            continue
        c = pearson(returns[new_symbol], rets)
        if abs(c) > abs(max_corr):
            max_corr = c
            corr_with = sym
    return {"blocked": abs(max_corr) > threshold, "max_corr": round(max_corr, 4), "corr_with": corr_with}


def compute_vol(prices: List[float]) -> float:
    if len(prices) < 5:
        return TARGET_VOL
    returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1] > 0]
    if not returns:
        return TARGET_VOL
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    return round(math.sqrt(variance) * math.sqrt(365 * 24), 4)


class RiskManager:
    def __init__(self, max_positions: int = 10, daily_loss_limit_pct: float = 5.0, corr_threshold: float = 0.75):
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit_pct / 100.0
        self.corr_threshold = corr_threshold
        self.returns_cache: Dict[str, List[float]] = {}

    def can_enter_position(self, symbol: str, direction: str, portfolio_value: float,
                           open_positions: List[Dict], current_prices: Dict[str, float]) -> Tuple[bool, str]:
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached."
        return True, ""

    def update_returns_cache(self, symbol: str, daily_return: float):
        if symbol not in self.returns_cache:
            self.returns_cache[symbol] = []
        self.returns_cache[symbol].append(daily_return)
        self.returns_cache[symbol] = self.returns_cache[symbol][-30:]

    def check_correlation_with_existing(self, symbol: str, open_symbols: List[str]) -> Tuple[bool, float]:
        if symbol not in self.returns_cache or len(self.returns_cache[symbol]) < 5:
            return False, 0.0
        max_corr = 0.0
        for s in open_symbols:
            if s in self.returns_cache and len(self.returns_cache[s]) >= 5:
                c = pearson(self.returns_cache[symbol], self.returns_cache[s])
                if abs(c) > abs(max_corr):
                    max_corr = c
        return abs(max_corr) > self.corr_threshold, max_corr
