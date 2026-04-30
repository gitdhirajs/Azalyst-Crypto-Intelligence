"""
generate_dashboard.py - Azalyst Crypto Intelligence Dashboard Builder

Reads runtime pipeline artifacts (model reports, scan signals, hourly patterns)
and produces a unified status.json matching the ETF Intelligence contract.
"""

import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
OUTPUT_FILE = ROOT / "status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return round(out, 4)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(safe_float(value, float(default)))


def sign_pct(value: float) -> str:
    return f"+{value:.2f}%" if value >= 0 else f"{value:.2f}%"


def sign_usd(value: float) -> str:
    return f"+${value:,.2f}" if value >= 0 else f"${value:,.2f}"


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv_signals(path: Path) -> List[Dict]:
    """Load latest quant signals CSV into a list of dicts."""
    if not path.exists():
        return []
    import csv
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------

def collect_model_health() -> Dict:
    """Gather training metrics from latest main and hourly model reports."""
    main_report = load_json(LOGS_DIR / "latest_train_report.json")
    hourly_report = load_json(LOGS_DIR / "latest_hourly_train_report.json")

    return {
        "main_model": {
            "status": main_report.get("status", "unknown"),
            "accuracy": safe_float(main_report.get("accuracy")) * 100 if main_report.get("accuracy") else None,
            "f1_score": safe_float(main_report.get("f1_score")) * 100 if main_report.get("f1_score") else None,
            "n_train": safe_int(main_report.get("n_train")),
            "n_test": safe_int(main_report.get("n_test")),
            "n_symbols": safe_int(main_report.get("n_symbols")),
            "top_features": main_report.get("top_features", [])[:5],
        },
        "hourly_model": {
            "status": hourly_report.get("status", "unknown"),
            "accuracy": safe_float(hourly_report.get("accuracy")) * 100 if hourly_report.get("accuracy") else None,
            "f1_score": safe_float(hourly_report.get("f1_score")) * 100 if hourly_report.get("f1_score") else None,
            "n_train": safe_int(hourly_report.get("n_train")),
            "n_test": safe_int(hourly_report.get("n_test")),
            "top_features": hourly_report.get("top_features", [])[:5],
        },
    }


def collect_live_signals() -> List[Dict]:
    """Parse latest scan signals from CSV."""
    rows = load_csv_signals(REPORTS_DIR / "latest_quant_signals.csv")
    signals = []
    for row in rows:
        signals.append({
            "symbol": row.get("symbol", "?"),
            "chain": row.get("chain", ""),
            "label": row.get("label", "watch"),
            "pump_score": safe_float(row.get("pump_score")),
            "dump_score": safe_float(row.get("dump_score")),
            "anomaly_score": safe_float(row.get("anomaly_score")),
            "smart_money_score": safe_float(row.get("smart_money_score")),
            "risk_score": safe_float(row.get("risk_score")),
            "direction": "BULLISH" if safe_float(row.get("pump_score")) > safe_float(row.get("dump_score")) else "BEARISH" if safe_float(row.get("dump_score")) > 55 else "NEUTRAL",
            "reasons": (row.get("reasons") or "").split(";")[:3],
        })
    return signals


def collect_outcome_stats() -> Dict:
    """Calculate hit rate from latest outcomes."""
    rows = load_csv_signals(REPORTS_DIR / "latest_quant_outcomes.csv")
    if not rows:
        return {"total": 0, "hits": 0, "hit_rate": None}

    hits = sum(1 for r in rows if r.get("is_true", "").lower() in ("true", "1"))
    return {
        "total": len(rows),
        "hits": hits,
        "hit_rate": round(hits / len(rows) * 100, 1) if rows else None,
    }


def collect_system_metrics() -> Dict:
    """Gather scanner pipeline metrics."""
    summary = (REPORTS_DIR / "latest_summary.md").read_text(encoding="utf-8")[:2000] if (REPORTS_DIR / "latest_summary.md").exists() else ""

    # Count data files
    raw_files = list(DATA_DIR.rglob("scans_*.csv")) if DATA_DIR.exists() else []
    feature_files = list(DATA_DIR.rglob("features_*.csv")) if DATA_DIR.exists() else []

    return {
        "scans_collected": len(raw_files),
        "feature_batches": len(feature_files),
        "latest_summary_snippet": summary[:300] if summary else "",
    }


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

def generate_status() -> Dict:
    """Build unified status.json matching ETF contract."""
    now = utc_now()
    models = collect_model_health()
    signals = collect_live_signals()
    outcomes = collect_outcome_stats()
    system = collect_system_metrics()

    # Build market snapshot from top signal tokens (proxy for crypto market)
    top_tokens = [s for s in signals if safe_float(s.get("pump_score", 0)) > 30 or safe_float(s.get("anomaly_score", 0)) > 40]
    market_tiles = []
    for s in top_tokens[:12]:
        direction = "up" if s["direction"] == "BULLISH" else "down" if s["direction"] == "BEARISH" else "neu"
        market_tiles.append({
            "label": s["symbol"],
            "ticker": f"{s['symbol']}-USDT",
            "region": s.get("chain", "").upper()[:6],
            "price": s.get("pump_score", 0),
            "currency": "SCORE",
            "change": s.get("pump_score", 0) - s.get("dump_score", 0),
            "change_pct": s.get("pump_score", 0) - s.get("dump_score", 0),
            "change_str": sign_pct(s.get("pump_score", 0) - s.get("dump_score", 0)),
            "direction": direction,
        })

    # Build confidence map from signal scores
    confidence_map = []
    for s in signals[:20]:
        score = max(s["pump_score"], s["dump_score"], s["anomaly_score"], s.get("smart_money_score", 0))
        confidence_map.append({
            "symbol": s["symbol"],
            "score": round(score, 1),
            "label": s["label"],
        })

    # Build signal breakdowns
    signal_cards = []
    for s in signals[:15]:
        card = {
            "sector_key": f"{s.get('chain','')}|{s.get('symbol','')}",
            "sector_label": f"{s['symbol']} ({s.get('chain','').upper()})",
            "confidence": round(max(s["pump_score"], s["dump_score"], s.get("anomaly_score", 0))),
            "severity": "CRITICAL" if s["pump_score"] >= 70 else "HIGH" if s["pump_score"] >= 50 else "MEDIUM",
            "direction": s["direction"],
            "direction_score": round(s["pump_score"] - s["dump_score"], 2),
            "ml_sentiment_label": s["label"].upper(),
            "ml_sentiment_score": round(max(s["pump_score"], s["dump_score"]) / 100, 4),
            "ml_sentiment_mode": "xgboost",
            "article_count": len(s.get("reasons", [])),
            "latest_at": now,
            "headline": f"{s['symbol']} — {'Pump' if s['direction']=='BULLISH' else 'Dump' if s['direction']=='BEARISH' else 'Watch'} signal ({max(s['pump_score'],s['dump_score']):.0f}/100)",
            "regions": [s.get("chain", "").upper()],
            "sources": s.get("reasons", [])[:3],
            "primary_etf": f"{s['symbol']}-USDT",
            "top_etfs": [f"{s['symbol']}-USDT"],
            "access_markets": ["Perpetual Futures"],
            "india_etfs": [],
            "global_etfs": [],
            "is_legacy": False,
            "breakdown": {
                "signal_strength": s["pump_score"],
                "volume_confirmation": s.get("smart_money_score", 0),
                "source_diversity": min(len(s.get("reasons", [])) * 3.3, 25),
                "recency": 20.0,
                "geopolitical_severity": s["risk_score"],
            },
        }
        signal_cards.append(card)

    # Build articles feed
    articles = []
    for s in signals[:12]:
        direction = s["direction"]
        tag = "tag-bull" if direction == "BULLISH" else "tag-bear" if direction == "BEARISH" else "tag-neu"
        badge = "Bullish" if direction == "BULLISH" else "Bearish" if direction == "BEARISH" else "Neutral"
        articles.append({
            "tag": tag,
            "label": badge,
            "text": f"{s['chain'].upper()} - {s['symbol']} - {s['label']} - Pump:{s['pump_score']:.0f} Dump:{s['dump_score']:.0f} Anom:{s['anomaly_score']:.0f}",
        })

    # Build risk controls
    high_risk = [s for s in signals if s["risk_score"] >= 50]
    risk_controls = {
        "circuit_breaker_active": len(high_risk) > 8,
        "drawdown_from_peak_pct": round(len(high_risk) / max(len(signals), 1) * 100, 1),
        "portfolio_peak": len(signals),
        "vix": None,
        "vix_regime": "HIGH" if len(high_risk) > 5 else "NORMAL",
        "sector_concentration": [
            {"sector": f"{s['symbol']} ({s.get('chain','').upper()})", "weight": s["pump_score"] * 0.3, "at_cap": s["pump_score"] > 70}
            for s in signals[:8]
        ],
        "max_drawdown_pct": round(len(high_risk) / max(len(signals), 1) * 100, 1),
        "sector_cap_pct": 30,
        "trailing_stop_pct": 8,
        "hard_stop_pct": 10,
        "partial_profit_pct": 8,
    }

    # Build track record from outcomes
    track_record = {
        "total_trades": outcomes.get("total", 0),
        "winners": outcomes.get("hits", 0),
        "losers": outcomes.get("total", 0) - outcomes.get("hits", 0),
        "win_rate": outcomes.get("hit_rate") or 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "sharpe_proxy": 0.0,
        "best": None,
        "worst": None,
    }

    logs = [
        f"{now} [INFO] AZALYST CRYPTO — status.json generated",
        f"{now} [INFO] SCANNER — {len(signals)} active signals",
        f"{now} [INFO] OUTCOMES — {outcomes.get('hits',0)}/{outcomes.get('total',0)} hits ({outcomes.get('hit_rate','N/A')}%)",
        f"{now} [INFO] MODELS — Main: {models['main_model'].get('status','?')} | Hourly: {models['hourly_model'].get('status','?')}",
    ]

    status = {
        "dashboard_type": "crypto_intelligence",
        "generated_at": now,
        "portfolio_value": len(signals) * 100,
        "total_deposited": 10000,
        "cash": 5000,
        "monthly_reserve": 5000,
        "market_value": len(signals) * 50,
        "total_invested": len(signals) * 50,
        "unrealised_pnl": 0,
        "unrealised_str": "+0.00",
        "realised_pnl": outcomes.get("hits", 0) * 10.0,
        "realised_str": f"+{outcomes.get('hits', 0) * 10.0:,.2f}" if outcomes.get("hits") else "+0.00",
        "change": "+0.00%",
        "change_raw": 0.0,
        "closed_trades": outcomes.get("total", 0),
        "usd_inr_rate": 83.5,
        "positions": [],
        "closed_trades_list": [],
        "track_record": track_record,
        "confidence_threshold": 50,
        "allocation": {"labels": ["BTC", "ETH", "SOL", "CASH"], "values": [30, 25, 20, 25]},
        "pnl": {"labels": [], "values": []},
        "confidence": confidence_map,
        "signals": signal_cards,
        "articles": articles,
        "market_snapshot": market_tiles,
        "risk_controls": risk_controls,
        "model_health": models,
        "system_metrics": system,
        "aladdin_risk": {},
        "logs": logs,
    }

    return status


def main():
    status = generate_status()
    OUTPUT_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[OK] status.json written -> {OUTPUT_FILE}")
    print(f"     Signals: {len(status['signals'])} | Outcomes: {status['track_record']['total_trades']}")


if __name__ == "__main__":
    main()