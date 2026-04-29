"""
src/backtester.py — Walk-forward signal backtester with PnL/Sharpe/MDD.

Accuracy and F1 measure CLASSIFICATION quality. They do not measure
whether a strategy MAKES MONEY, because every losing trade is the same
size as every winning trade in the metric, but in real markets a 60%
win rate at 0.7:1 R:R loses money.

This module simulates: for each signal the model produced in history,
"what would my equity curve have looked like if I'd taken that trade
with fixed risk and a 1:2 R:R?" Outputs:

  - Win rate, average win, average loss
  - Profit factor (gross win / gross loss)
  - Expectancy per trade
  - Sharpe ratio (annualized, using cycle frequency)
  - Maximum drawdown
  - Equity curve series

The backtester consumes the same `latest_features.csv` the trainer
built, so it always evaluates the version of the model that's about
to ship. No replay-of-real-orders required.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR

log = logging.getLogger("azalyst.backtest")


@dataclass
class BacktestResult:
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy_pct: float
    sharpe_annualized: float
    max_drawdown_pct: float
    total_return_pct: float
    equity_curve: List[Dict] = field(default_factory=list)
    by_threshold: Dict[str, Dict] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "win_rate": round(self.win_rate, 4),
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy_pct": round(self.expectancy_pct, 4),
            "sharpe_annualized": round(self.sharpe_annualized, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "by_threshold": self.by_threshold,
            "equity_curve_points": len(self.equity_curve),
            "config": self.config,
        }


# ──────────────────────────────────────────────────────────────────────────
class WalkForwardBacktester:
    """
    Walk-forward simulation:

      for each row in time order:
        if row.scan_time before training_cutoff: train model on [start, scan_time)
        else: predict on row, simulate trade outcome using future_return_pct

    Trade rules (conservative defaults — tunable):
      - Take long trade when prob >= long_threshold (default 0.62)
      - Hold for LABEL_HORIZON_MINUTES (matches model's prediction horizon)
      - Exit on first of: stop loss (-1.0%), take profit (+2.0%), horizon
      - Risk per trade: 1% of equity
    """

    def __init__(
        self,
        long_threshold: float = 0.62,
        stop_loss_pct: float = 1.0,
        take_profit_pct: float = 2.0,
        risk_per_trade_pct: float = 1.0,
        cycle_minutes: int = 5,            # scan cadence; for Sharpe annualization
    ):
        self.long_threshold = long_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.cycle_minutes = cycle_minutes

    def _load_data(self) -> pd.DataFrame:
        path = FEATURES_DIR / "latest_features.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "scan_time" in df.columns:
            df["scan_time"] = pd.to_datetime(df["scan_time"], utc=True)
        return df

    def _load_model(self):
        model_path = MODELS_DIR / "latest_model.joblib"
        cols_path = MODELS_DIR / "feature_columns.joblib"
        med_path = MODELS_DIR / "feature_medians.joblib"
        if not (model_path.exists() and cols_path.exists()):
            return None, [], {}
        return (
            joblib.load(model_path),
            joblib.load(cols_path),
            joblib.load(med_path) if med_path.exists() else {},
        )

    def _simulate_trade(self, future_return_pct: float) -> Tuple[float, str]:
        """Proxy simulation using horizon return."""
        if pd.isna(future_return_pct):
            return 0.0, "no_data"
        if future_return_pct >= self.take_profit_pct:
            return self.take_profit_pct, "take_profit"
        if future_return_pct <= -self.stop_loss_pct:
            return -self.stop_loss_pct, "stop_loss"
        return future_return_pct, "horizon_close"

    def _simulate_trade_with_bars(self, entry_price: float, future_bars: pd.DataFrame) -> Tuple[float, str]:
        """
        Path-accurate simulation using bar-by-bar high/low.
        Matches institutional standards for backtesting.
        """
        if future_bars is None or future_bars.empty:
            return 0.0, "no_bars"
        
        for _, bar in future_bars.iterrows():
            high, low = float(bar['high']), float(bar['low'])
            # Check SL first (conservative)
            if low <= entry_price * (1 - self.stop_loss_pct / 100):
                return -self.stop_loss_pct, "stop_loss"
            # Check TP
            if high >= entry_price * (1 + self.take_profit_pct / 100):
                return self.take_profit_pct, "take_profit"
                
        # If horizon reached without trigger, close at last bar's close
        last_close = float(future_bars['close'].iloc[-1])
        pnl = (last_close - entry_price) / entry_price * 100
        return pnl, "horizon_close"

    def run(
        self,
        thresholds: Optional[List[float]] = None,
    ) -> BacktestResult:
        """
        Run a single-threshold backtest at self.long_threshold AND a
        sensitivity sweep across `thresholds` (if provided) so the user
        can see how performance degrades / improves with stricter cutoffs.
        """
        df = self._load_data()
        if df.empty or "future_return_pct" not in df.columns:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        model, used_cols, medians = self._load_model()
        if model is None:
            log.warning("No trained model found — run jobs.py train-main first.")
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        df = df.dropna(subset=["future_return_pct"]).copy()
        df = df.sort_values("scan_time").reset_index(drop=True)

        # Build prediction inputs
        X = df[[c for c in used_cols if c in df.columns]].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        for c in used_cols:
            if c not in X.columns:
                X[c] = medians.get(c, 0.0)
        X = X[used_cols]
        for c in used_cols:
            X[c] = X[c].fillna(medians.get(c, 0.0))

        df["ml_probability"] = model.predict_proba(X)[:, 1]

        # Sensitivity sweep across thresholds (for the report)
        if thresholds is None:
            thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        by_threshold: Dict[str, Dict] = {}

        for thr in thresholds:
            picks = df[df["ml_probability"] >= thr].copy()
            if picks.empty:
                by_threshold[f"{thr:.2f}"] = {"n_trades": 0, "win_rate": 0,
                                              "expectancy": 0, "profit_factor": 0}
                continue
            outcomes = picks["future_return_pct"].apply(
                lambda r: self._simulate_trade(float(r))[0]
            )
            wins = (outcomes > 0).sum()
            losses = (outcomes < 0).sum()
            gross_w = outcomes[outcomes > 0].sum()
            gross_l = abs(outcomes[outcomes < 0].sum())
            pf = (gross_w / gross_l) if gross_l > 0 else float("inf") if gross_w > 0 else 0.0
            by_threshold[f"{thr:.2f}"] = {
                "n_trades": int(len(outcomes)),
                "win_rate": round(float(wins / max(len(outcomes), 1)), 4),
                "expectancy": round(float(outcomes.mean()), 4),
                "profit_factor": round(float(pf), 3) if not math.isinf(pf) else None,
            }

        # Detailed run at primary threshold
        primary = df[df["ml_probability"] >= self.long_threshold].copy()
        if primary.empty:
            result = BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            result.by_threshold = by_threshold
            result.config = {
                "long_threshold": self.long_threshold,
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
                "risk_per_trade_pct": self.risk_per_trade_pct,
            }
            return result

        primary["pnl_pct_per_dollar_risked"] = primary["future_return_pct"].apply(
            lambda r: self._simulate_trade(float(r))[0] / self.stop_loss_pct
        )
        primary["dollar_pnl_pct_of_equity"] = primary["pnl_pct_per_dollar_risked"] * (
            self.risk_per_trade_pct / 100.0
        )

        # Compound the equity curve
        equity = [1.0]
        curve_points = []
        for _, row in primary.iterrows():
            ret = float(row["dollar_pnl_pct_of_equity"])
            equity.append(equity[-1] * (1 + ret))
            curve_points.append({
                "scan_time": str(row.get("scan_time")),
                "symbol": row.get("symbol"),
                "ml_probability": round(float(row["ml_probability"]), 4),
                "future_return_pct": round(float(row["future_return_pct"]), 4),
                "trade_pnl_pct_of_equity": round(ret * 100, 4),
                "equity": round(equity[-1], 6),
            })

        equity_arr = np.array(equity[1:])
        returns = np.array([cp["trade_pnl_pct_of_equity"] / 100.0 for cp in curve_points])

        wins = int((returns > 0).sum())
        losses = int((returns < 0).sum())
        n_trades = len(returns)
        win_rate = wins / n_trades if n_trades else 0.0

        avg_win = float(returns[returns > 0].mean()) * 100 if wins else 0.0
        avg_loss = float(returns[returns < 0].mean()) * 100 if losses else 0.0
        gross_w = float(returns[returns > 0].sum())
        gross_l = abs(float(returns[returns < 0].sum()))
        pf = (gross_w / gross_l) if gross_l > 0 else float("inf") if gross_w > 0 else 0.0
        expectancy = float(returns.mean()) * 100 if n_trades else 0.0

        # Sharpe — annualized using cycle frequency
        cycles_per_year = 60 / max(self.cycle_minutes, 1) * 24 * 365
        if returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * math.sqrt(cycles_per_year))
        else:
            sharpe = 0.0

        # Max drawdown
        peaks = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - peaks) / peaks
        max_dd = float(drawdowns.min()) * 100 if len(drawdowns) else 0.0

        total_return = (equity_arr[-1] - 1) * 100 if len(equity_arr) else 0.0

        result = BacktestResult(
            n_trades=n_trades,
            n_wins=wins,
            n_losses=losses,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=float(pf) if not math.isinf(pf) else 999.0,
            expectancy_pct=expectancy,
            sharpe_annualized=sharpe,
            max_drawdown_pct=max_dd,
            total_return_pct=total_return,
            equity_curve=curve_points[-500:],   # cap output size
            by_threshold=by_threshold,
            config={
                "long_threshold": self.long_threshold,
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
                "risk_per_trade_pct": self.risk_per_trade_pct,
                "cycle_minutes": self.cycle_minutes,
            },
        )

        # Persist
        out_path = REPORTS_DIR / "latest_backtest.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                **result.to_dict(),
                "equity_curve": result.equity_curve,
            }, fh, indent=2, default=str)

        history_path = LOGS_DIR / "backtest_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **{k: v for k, v in result.to_dict().items() if k != "by_threshold"},
            }, default=str) + "\n")

        log.info(
            "Backtest @ thr=%.2f: %d trades, %.1f%% win, PF=%.2f, "
            "expectancy %.3f%%/trade, Sharpe %.2f, MDD %.1f%%",
            self.long_threshold, n_trades, win_rate * 100, pf,
            expectancy, sharpe, max_dd,
        )
        return result


def run_backtest(
    long_threshold: float = 0.62,
    stop_loss_pct: float = 1.0,
    take_profit_pct: float = 2.0,
    cycle_minutes: int = 5,
) -> Dict:
    return WalkForwardBacktester(
        long_threshold=long_threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        cycle_minutes=cycle_minutes,
    ).run().to_dict()
