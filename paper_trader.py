"""
paper_trader.py - Azalyst Crypto Paper Trading Engine

Lightweight portfolio tracker for crypto perpetual scan signals.
Tracks entries/exits, P&L, and provides summary for dashboard.
"""

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("azalyst_crypto.trader")

# ── Constants ────────────────────────────────────────────────────────────────
MAX_POSITIONS = 10
MIN_TRADE_USD = 50
STOP_LOSS_PCT = 0.10
TRAILING_STOP_PCT = 0.08
MAX_HOLD_HOURS = 48

# Discord webhook for paper trade alerts
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


class Portfolio:
    """Simple crypto paper portfolio for dashboard P&L tracking."""

    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.file = portfolio_file
        self.open_positions: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.cash_usdt = 10000.0
        self.total_deposited = 10000.0
        self.trade_counter = 0
        self._load()

    def _load(self):
        if not os.path.exists(self.file):
            return
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.open_positions = data.get("open_positions", [])
            self.closed_trades = data.get("closed_trades", [])
            self.cash_usdt = data.get("cash_usdt", 10000.0)
            self.total_deposited = data.get("total_deposited", 10000.0)
            self.trade_counter = data.get("trade_counter", 0)
            log.info("Portfolio loaded: %.0f USDT, %d open, %d closed",
                     self.cash_usdt, len(self.open_positions), len(self.closed_trades))
        except Exception as e:
            log.error("Portfolio load error: %s", e)

    def save(self):
        try:
            with open(self.file, "w", encoding="utf-8") as f:
                json.dump({
                    "open_positions": self.open_positions,
                    "closed_trades": self.closed_trades,
                    "cash_usdt": round(self.cash_usdt, 2),
                    "total_deposited": round(self.total_deposited, 2),
                    "trade_counter": self.trade_counter,
                }, f, indent=2)
        except Exception as e:
            log.error("Portfolio save error: %s", e)

    def enter_position(self, symbol: str, entry_price: float, units: float, signal_confidence: int) -> Optional[Dict]:
        cost = round(entry_price * units, 2)
        if cost < MIN_TRADE_USD or cost > self.cash_usdt:
            return None
        if len(self.open_positions) >= MAX_POSITIONS:
            return None

        self.trade_counter += 1
        pos = {
            "id": f"C{self.trade_counter:04d}",
            "symbol": symbol,
            "entry_price": entry_price,
            "current_price": entry_price,
            "units": units,
            "invested": cost,
            "entry_date": datetime.now(timezone.utc).isoformat(),
            "confidence": signal_confidence,
            "peak_price": entry_price,
            "trail_stop": round(entry_price * (1 - STOP_LOSS_PCT), 4),
        }
        self.cash_usdt -= cost
        self.open_positions.append(pos)
        self.save()
        log.info("ENTER %s: %s @ %.4f x %.2f = $%.2f", pos["id"], symbol, entry_price, units, cost)
        self._send_discord_alert("ENTRY", pos)
        return pos

    def update_prices(self, prices: Dict[str, float]):
        for pos in self.open_positions:
            new_price = prices.get(pos["symbol"])
            if new_price:
                pos["current_price"] = new_price
                if new_price > pos["peak_price"]:
                    pos["peak_price"] = new_price
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
            pnl_pct = (pos["current_price"] - pos["entry_price"]) / pos["entry_price"] * 100
            if exit_price:
                pnl = round((exit_price - pos["entry_price"]) * pos["units"], 2)
                self.open_positions.remove(pos)
                trade = {
                    "id": pos["id"],
                    "symbol": pos["symbol"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "units": pos["units"],
                    "pnl": pnl,
                    "pnl_pct": round(pnl_pct, 2),
                    "entry_date": pos["entry_date"],
                    "exit_date": datetime.now(timezone.utc).isoformat(),
                    "exit_reason": reason,
                }
                self.closed_trades.append(trade)
                self.cash_usdt += round(exit_price * pos["units"], 2)
                exits.append(trade)
                log.info("EXIT %s: %s P&L $%.2f (%+.2f%%) — %s", pos["id"], pos["symbol"], pnl, pnl_pct, reason)
                self._send_discord_alert("EXIT", trade)
        if exits:
            self.save()
        return exits

    def get_summary(self) -> Dict:
        invested = sum(p["invested"] for p in self.open_positions)
        current = sum(p["current_price"] * p["units"] for p in self.open_positions)
        unreal = round(current - invested, 2)
        closed_pnl = sum(t["pnl"] for t in self.closed_trades)
        portfolio_value = round(self.cash_usdt + current, 2)
        total_return = round((portfolio_value - self.total_deposited) / self.total_deposited * 100, 2) if self.total_deposited else 0

        winners = [t for t in self.closed_trades if t["pnl"] > 0]
        losers = [t for t in self.closed_trades if t["pnl"] < 0]
        win_rate = round(len(winners) / len(self.closed_trades) * 100, 1) if self.closed_trades else 0

        return {
            "cash_usdt": round(self.cash_usdt, 2),
            "invested": round(invested, 2),
            "current_value": round(current, 2),
            "unrealised_pnl": unreal,
            "closed_pnl": round(closed_pnl, 2),
            "portfolio_value": portfolio_value,
            "total_deposited": round(self.total_deposited, 2),
            "total_return_pct": total_return,
            "open_count": len(self.open_positions),
            "closed_count": len(self.closed_trades),
            "win_rate": win_rate,
            "winners": len(winners),
            "losers": len(losers),
        }

    def _send_discord_alert(self, action: str, data: Dict) -> bool:
        """Send paper trade entry/exit alerts to Discord webhook."""
        if not DISCORD_WEBHOOK:
            return False
        
        try:
            if action == "ENTRY":
                color = 0x2ECC71
                title = f"📈 PAPER ENTRY: {data['symbol']}"
                description = (
                    f"**Trade ID:** `{data['id']}`\n"
                    f"**Direction:** LONG\n"
                    f"**Entry Price:** `${data['entry_price']:,.4f}`\n"
                    f"**Units:** `{data['units']:.4f}`\n"
                    f"**Invested:** `${data['invested']:,.2f}`\n"
                    f"**Confidence:** `{data['confidence']}/100`\n"
                    f"**Stop Loss:** `${data['trail_stop']:,.4f}`"
                )
            else:  # EXIT
                color = 0xE74C3C if data["pnl"] < 0 else 0x2ECC71
                emoji = "💰" if data["pnl"] > 0 else "💸"
                title = f"{emoji} PAPER EXIT: {data['symbol']}"
                pnl_sign = "+" if data["pnl"] >= 0 else ""
                description = (
                    f"**Trade ID:** `{data['id']}`\n"
                    f"**Entry Price:** `${data['entry_price']:,.4f}`\n"
                    f"**Exit Price:** `${data['exit_price']:,.4f}`\n"
                    f"**P&L:** `${pnl_sign}{data['pnl']:,.2f}` ({pnl_sign}{data['pnl_pct']:.2f}%)\n"
                    f"**Reason:** {data['exit_reason']}"
                )
            
            embed = {
                "title": title,
                "description": description,
                "color": color,
                "footer": {"text": "Azalyst Crypto Paper Trading"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            resp = requests.post(
                DISCORD_WEBHOOK,
                json={"embeds": [embed]},
                timeout=8,
            )
            if resp.status_code in (200, 204):
                log.info("Discord paper trade alert sent: %s %s", action, data.get("symbol", ""))
                return True
            log.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            log.warning("Discord paper trade alert failed: %s", exc)
        return False