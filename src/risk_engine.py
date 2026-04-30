"""risk_engine.py - Azalyst Crypto Risk Controls
Lightweight risk layer: correlation check + vol estimate."""
import math
import logging
from typing import Dict, List

log = logging.getLogger("azalyst_crypto.risk")

CORRELATION_WARN = 0.75
TARGET_VOL = 0.20

def pearson(x: List[float], y: List[float]) -> float:
    n = min(len(x), len(y))
    if n < 5: return 0.0
    mx = sum(x[:n]) / n; my = sum(y[:n]) / n
    sx = math.sqrt(sum((v-mx)**2 for v in x[:n])); sy = math.sqrt(sum((v-my)**2 for v in y[:n]))
    if sx == 0 or sy == 0: return 0.0
    return round(sum((x[i]-mx)*(y[i]-my) for i in range(n)) / (sx*sy), 4)

def check_correlation(returns: Dict[str, List[float]], new_symbol: str, threshold: float = CORRELATION_WARN) -> Dict:
    if new_symbol not in returns: return {"blocked": False, "max_corr": 0}
    max_corr = 0.0; corr_with = ""
    for sym, rets in returns.items():
        if sym == new_symbol: continue
        c = pearson(returns[new_symbol], rets)
        if abs(c) > abs(max_corr): max_corr = c; corr_with = sym
    return {"blocked": abs(max_corr) > threshold, "max_corr": round(max_corr, 4), "corr_with": corr_with}

def compute_vol(prices: List[float]) -> float:
    if len(prices) < 5: return TARGET_VOL
    returns = [math.log(prices[i]/prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
    if not returns: return TARGET_VOL
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r)**2 for r in returns) / len(returns)
    return round(math.sqrt(variance) * math.sqrt(365*24), 4)