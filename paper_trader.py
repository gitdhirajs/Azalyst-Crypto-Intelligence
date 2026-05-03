"""
paper_trader.py - Azalyst Crypto Paper Trading Engine v2

Integrated with risk manager, atomic writes, mid-price, slippage, Discord alerts.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from src.risk_engine import RiskManager

log = logging.getLogger("azalyst_crypto.trader")

MAX_POSITIONS = 10
MIN_TRADE_USD = 50
STOP_LOSS_PCT = 0.10
TRAILING_STOP_PCT = 0.08
SLIPPAGE = 0.0005

DISCORD_WEBHOOK = os.getenv("DISCORD_TRADING_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", "")).strip()


def atomic_write_json(obj: Dict, filepath: str) -> None:
    temp = filepath + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(temp, filepath)


class Portfolio:
    def __init__(self, portfolio_file: str = "portfolio.json", risk_manager: Optional[RiskManager] = None):
        self.file = portfolio_file
        self.risk_manager = risk_manager or RiskManager()
        self.open_positions: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.cash_usdt = 10000.0
        self.total_deposited = 10000.0
        self.trade_counter = 0
        self.daily_pnl_realised = 0.0
        self.last_reset_day = datetime.now(timezone.utc).date()
        self._load()

    def _load(self):
        if not os.path.exists(self.file):
            return
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.open_positions = data.get("open_positions", [])
            self.closed_trades = data.get("closed_trades", [])
            self.cash_usdt = float(data.get("cash_usdt", 10000.0))
            self.total_deposited = float(data.get("total_deposited", 10000.0))
            self.trade_counter = int(data.get("trade_counter", 0))
            log.info("Portfolio loaded: $%.0f, %d open, %d closed", self.cash_usdt, len(self.open_positions), len(self.closed_trades))
        except Exception as e:
            log.error("Portfolio load error: %s", e)

    def save(self):
        self._reset_daily_pnl_if_new_day()
        atomic_write_json({
            "open_positions": self.open_positions,
            "closed_trades": self.closed_trades,
            "cash_usdt": round(self.cash_usdt, 2),
            "total_deposited": round(self.total_deposited, 2),
            "trade_counter": self.trade_counter,
            "daily_pnl_realised": self.daily_pnl_realised,
        }, self.file)

    def _reset_daily_pnl_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_day:
            self.daily_pnl_realised = 0.0
            self.last_reset_day = today

    def _mid_price(self, symbol: str, prices: Dict[str, float]) -> Optional[float]:
        return prices.get(symbol)

    def enter_position(self, symbol: str, entry_price: float, units: float, confidence: float,
                       current_prices: Dict[str, float]) -> Optional[Dict]:
        entry_price_with_slip = entry_price * (1 + SLIPPAGE)
        cost = round(entry_price_with_slip * units, 2)
        if cost < MIN_TRADE_USD or cost > self.cash_usdt:
            return None

        open_symbols = [p["symbol"] for p in self.open_positions]
        allowed, reason = self.risk_manager.can_enter_position(symbol, "LONG", self.portfolio_value(),
                                                                self.open_positions, current_prices)
        if not allowed:
            log.info("Risk block: %s", reason)
            return None

        high_corr, max_corr = self.risk_manager.check_correlation_with_existing(symbol, open_symbols)
        if high_corr:
            log.info("Correlation block: %s with existing positions (max corr=%.2f)", symbol, max_corr)
            return None

        if self.daily_pnl_realised + self.unrealised_pnl() < -self.total_deposited * self.risk_manager.daily_loss_limit:
            log.info("Daily loss limit hit. No new entries.")
            return None

        self.trade_counter += 1
        pos = {
            "id": f"C{self.trade_counter:04d}",
            "symbol": symbol,
            "entry_price": entry_price_with_slip,
            "current_price": entry_price_with_slip,
            "units": units,
            "invested": cost,
            "entry_date": datetime.now(timezone.utc).isoformat(),
            "confidence": confidence,
            "peak_price": entry_price_with_slip,
            "trail_stop": round(entry_price_with_slip * (1 - STOP_LOSS_PCT), 4),
        }
        self.cash_usdt -= cost
        self.open_positions.append(pos)
        self.save()
        self._send_discord_alert("ENTRY", pos)
        return pos

    def update_prices(self, prices: Dict[str, float]):
        for pos in self.open_positions:
            mid = self._mid_price(pos["symbol"], prices)
            if mid:
                pos["current_price"] = mid
                if mid > pos["peak_price"]:
                    pos["peak_price"] = mid
                if pos["peak_price"] >= pos["entry_price"] * 1.05:
                    pos["trail_stop"] = round(max(
                        pos["entry_price"] * (1 - STOP_LOSS_PCT),
                        pos["peak_price"] * (1 - TRAILING_STOP_PCT)
                    ), 4)
        self.save()

    def check_exits(self) -> List[Dict]:
        exits = []
        for pos in list(self.open_positions):
            exit_price = None
            reason = ""
            if pos["current_price"] <= pos["trail_stop"]:
                exit_price = pos["current_price"]
                reason = "Stop-loss / Trailing stop hit"
            if exit_price:
                exit_slippage = exit_price * (1 - SLIPPAGE)
                pnl = round((exit_slippage - pos["entry_price"]) * pos["units"], 2)
                self.open_positions.remove(pos)
                trade = {
                    "id": pos["id"],
                    "symbol": pos["symbol"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_slippage,
                    "units": pos["units"],
                    "pnl": pnl,
                    "pnl_pct": round((exit_slippage - pos["entry_price"]) / pos["entry_price"] * 100, 2),
                    "entry_date": pos["entry_date"],
                    "exit_date": datetime.now(timezone.utc).isoformat(),
                    "exit_reason": reason,
                }
                self.closed_trades.append(trade)
                self.cash_usdt += round(exit_slippage * pos["units"], 2)
                self.daily_pnl_realised += pnl
                exits.append(trade)
                self._send_discord_alert("EXIT", trade)
        if exits:
            self.save()
        return exits

    def unrealised_pnl(self) -> float:
        return sum((p["current_price"] - p["entry_price"]) * p["units"] for p in self.open_positions)

    def portfolio_value(self) -> float:
        return self.cash_usdt + sum(p["current_price"] * p["units"] for p in self.open_positions)

    def get_summary(self) -> Dict:
        unreal = self.unrealised_pnl()
        total_val = self.portfolio_value()
        closed_pnl = sum(t["pnl"] for t in self.closed_trades)
        return {
            "cash_usdt": round(self.cash_usdt, 2),
            "unrealised_pnl": round(unreal, 2),
            "closed_pnl": round(closed_pnl, 2),
            "portfolio_value": round(total_val, 2),
            "total_return_pct": round((total_val - self.total_deposited) / self.total_deposited * 100, 2),
            "open_count": len(self.open_positions),
            "closed_count": len(self.closed_trades),
            "daily_pnl_realised": round(self.daily_pnl_realised, 2),
        }

    def _send_discord_alert(self, action: str, data: Dict) -> bool:
        if not DISCORD_WEBHOOK:
            return False
        try:
            if action == "ENTRY":
                color = 0x2ECC71
                title = f"📈 PAPER ENTRY: {data['symbol']}"
                description = (
                    f"**Trade ID:** `{data['id']}`\n"
                    f"**Direction:** LONG\n"
                    f"**Entry Price:** ${data['entry_price']:,.4f}\n"
                    f"**Units:** `{data['units']:.4f}`\n"
                    f"**Invested:** ${data['invested']:,.2f}\n"
                    f"**Confidence:** `{data['confidence']}/100`\n"
                    f"**Stop Loss:** ${data['trail_stop']:,.4f}"
                )
            else:
                color = 0xE74C3C if data["pnl"] < 0 else 0x2ECC71
                emoji = "💰" if data["pnl"] > 0 else "💸"
                title = f"{emoji} PAPER EXIT: {data['symbol']}"
                pnl_sign = "+" if data["pnl"] >= 0 else ""
                description = (
                    f"**Trade ID:** `{data['id']}`\n"
                    f"**Entry:** ${data['entry_price']:,.4f}\n"
                    f"**Exit:** ${data['exit_price']:,.4f}\n"
                    f"**P&L:** `{pnl_sign}{data['pnl']:,.2f}` ({pnl_sign}{data['pnl_pct']:.2f}%)\n"
                    f"**Reason:** {data['exit_reason']}"
                )
            embed = {
                "title": title,
                "description": description,
                "color": color,
                "footer": {"text": "Paper Trading Only — Not Financial Advice"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            resp = requests.post(DISCORD_WEBHOOK, json={"embeds": [embed]}, timeout=8)
            if resp.status_code in (200, 204):
                return True
            log.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            log.warning("Discord alert failed: %s", e)
        return False
