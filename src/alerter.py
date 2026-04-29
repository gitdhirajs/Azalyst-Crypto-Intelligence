"""
src/alerter.py — Discord webhook alerter for Tier-A crypto signals.

Crypto futures signals decay in MINUTES. CSV/JSON to a dashboard isn't
fast enough; you need a push notification the moment a Tier-A signal
fires. This module:

  - Sends rich-embed Discord messages for fused Tier A and B signals
  - Deduplicates: each (symbol, direction) sent at most once per
    DEDUPE_MINUTES (default 30) so a signal that re-fires every 5min
    doesn't spam your channel
  - Optionally pushes to Telegram if a bot token is set
  - Fully no-op if no webhook is configured (graceful degrade)

Set env vars:
  DISCORD_WEBHOOK_URL   — main alert channel
  TELEGRAM_BOT_TOKEN    — optional, also push to TG
  TELEGRAM_CHAT_ID      — required if TG token set
  ALERT_MIN_TIER        — A (default), B, or C
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.config import LOGS_DIR
from src.signal_fusion import FusedCryptoSignal

log = logging.getLogger("azalyst.alerter")


DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
ALERT_MIN_TIER = os.getenv("ALERT_MIN_TIER", "A").upper()
ALERT_DEDUPE_MINUTES = int(os.getenv("ALERT_DEDUPE_MINUTES", "30"))

DEDUPE_FILE = LOGS_DIR / "alert_dedupe.json"
TIER_RANK = {"A": 0, "B": 1, "C": 2}


# ──────────────────────────────────────────────────────────────────────────
def _load_dedupe() -> Dict[str, float]:
    if not DEDUPE_FILE.exists():
        return {}
    try:
        return json.loads(DEDUPE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_dedupe(d: Dict[str, float]) -> None:
    cutoff = time.time() - 24 * 3600
    pruned = {k: v for k, v in d.items() if v > cutoff}
    DEDUPE_FILE.write_text(json.dumps(pruned), encoding="utf-8")


def _should_send(signal: FusedCryptoSignal, dedupe: Dict[str, float]) -> bool:
    if TIER_RANK.get(signal.consensus_tier, 9) > TIER_RANK.get(ALERT_MIN_TIER, 0):
        return False
    key = f"{signal.symbol}|{signal.direction}|{signal.consensus_tier}"
    last = dedupe.get(key, 0)
    if time.time() - last < ALERT_DEDUPE_MINUTES * 60:
        return False
    dedupe[key] = time.time()
    return True


# ──────────────────────────────────────────────────────────────────────────
def _build_discord_embed(signal: FusedCryptoSignal) -> Dict:
    color = 0x2ECC71 if signal.direction == "LONG" else 0xE74C3C
    if signal.divergent:
        color = 0xF39C12

    fields = []
    for card in signal.cards[:8]:
        emoji = "🟢" if card.direction == "LONG" else "🔴" if card.direction == "SHORT" else "⚪"
        fields.append({
            "name": f"{emoji} {card.engine} ({card.direction})",
            "value": f"`{card.strength:.0f}/100` — {card.reason[:200]}",
            "inline": False,
        })

    if signal.ml_probability is not None:
        fields.append({
            "name": "🧠 ML probability",
            "value": f"`{signal.ml_probability:.2%}` (long={signal.ml_direction or 'neutral'})",
            "inline": True,
        })

    embed = {
        "title": f"🚨 Tier-{signal.consensus_tier}  {signal.direction}  {signal.symbol}",
        "description": (
            f"**Fused score:** `{signal.fused_score:.1f}/100`  •  "
            f"**Engines agreeing:** "
            f"`{len(signal.engines_long if signal.direction=='LONG' else signal.engines_short)}`"
            + (" ⚠️ **DIVERGENT**" if signal.divergent else "")
        ),
        "color": color,
        "fields": fields,
        "footer": {"text": "Azalyst Crypto Scanner v2 — multi-engine consensus"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return embed


def _send_discord(embeds: List[Dict]) -> bool:
    if not DISCORD_WEBHOOK or not embeds:
        return False
    try:
        resp = requests.post(
            DISCORD_WEBHOOK,
            json={"embeds": embeds[:10]},   # Discord cap
            timeout=8,
        )
        if resp.status_code in (200, 204):
            return True
        log.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        log.warning("Discord webhook failed: %s", exc)
    return False


def _send_telegram(signals: List[FusedCryptoSignal]) -> bool:
    if not TG_TOKEN or not TG_CHAT or not signals:
        return False
    try:
        lines = []
        for s in signals:
            arrow = "🟢" if s.direction == "LONG" else "🔴"
            lines.append(
                f"{arrow} *Tier-{s.consensus_tier}* {s.direction} *{s.symbol}* "
                f"`{s.fused_score:.0f}/100`"
                + (" ⚠️" if s.divergent else "")
                + f"\n_{s.summary[:280]}_"
            )
        text = "\n\n".join(lines)
        resp = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=8,
        )
        return resp.status_code == 200
    except Exception as exc:
        log.warning("Telegram send failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────────
def alert_fused_signals(signals: List[FusedCryptoSignal]) -> Dict[str, int]:
    """
    Send Discord (+ optional Telegram) alerts for high-tier fused signals.
    Returns a small stats dict.
    """
    stats = {"received": len(signals), "sent_discord": 0, "sent_telegram": 0,
             "skipped_tier": 0, "skipped_dedupe": 0}
    if not signals:
        return stats

    dedupe = _load_dedupe()
    selected: List[FusedCryptoSignal] = []
    for sig in signals:
        if TIER_RANK.get(sig.consensus_tier, 9) > TIER_RANK.get(ALERT_MIN_TIER, 0):
            stats["skipped_tier"] += 1
            continue
        if not _should_send(sig, dedupe):
            stats["skipped_dedupe"] += 1
            continue
        selected.append(sig)

    if not selected:
        _save_dedupe(dedupe)
        return stats

    embeds = [_build_discord_embed(s) for s in selected]
    if _send_discord(embeds):
        stats["sent_discord"] = len(selected)
    if _send_telegram(selected):
        stats["sent_telegram"] = len(selected)

    _save_dedupe(dedupe)
    log.info("Alerter: sent %d signals (discord=%d, tg=%d), %d skipped (tier=%d, dedupe=%d)",
             len(selected), stats["sent_discord"], stats["sent_telegram"],
             stats["skipped_tier"] + stats["skipped_dedupe"],
             stats["skipped_tier"], stats["skipped_dedupe"])
    return stats
