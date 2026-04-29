"""
src/pipeline.py — Multi-engine scheduled pipeline.

v2.1 cycle:

    1. Scan market (KuCoin primary + Bitget/OKX fallback) — collector
    2. Build features                                        — features
    3. Predict ML probabilities                              — trainer
    4. Pull Azalyst per-symbol context (top-N by vol)      — azalyst
    5. Run liq_proximity / funding / L/S / basis / oi_delta  — engines
    6. Fuse signals → Tier A/B/C                             — fusion
    7. Send Discord/TG alerts for Tier-A (and optional B)    — alerter
    8. Persist runtime payload + summary                     — pipeline

The original `run_scheduled_pipeline()` keeps the same signature so
the existing GitHub Actions workflows don't need to change. New
`scan_with_engines()` is the v2.1 entry point.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from src.alerter import alert_fused_signals
from src.backtester import WalkForwardBacktester
from src.derived_data import DerivedDataClient
from src.collector import (
    get_active_symbols,
    get_request_error_summary,
    reset_request_errors,
    scan_symbol,
)
from src.config import (
    BATCH_DELAY, BATCH_SIZE, LOGS_DIR, MIN_VOLUME_24H_USDT,
    RAW_DIR, REPORTS_DIR,
)
from src.exchange_fallback import fetch_perp_price, fetch_spot_price
from src.features import run_feature_pipeline
from src.hourly_trainer import train_hourly_model
from src.signal_engines import (
    BasisEngine, FundingExtremeEngine, LiquidationProximityEngine,
    LongShortExtremeEngine, OIDeltaEngine,
)
from src.signal_fusion import DynamicSignalFuser
from src.trainer import predict_current, train_model, incremental_train

log = logging.getLogger("azalyst.pipeline")


# ──────────────────────────────────────────────────────────────────────────
def scan_market_once(show_progress: bool = True) -> pd.DataFrame:
    """Run a single full-market scan and persist raw rows. Unchanged from v2.0."""
    now = datetime.now(timezone.utc)
    reset_request_errors()
    symbols, bulk_tickers = get_active_symbols()
    rows, failed = [], 0

    def _process(symbol: str) -> None:
        nonlocal failed
        preloaded = bulk_tickers.get(symbol)
        row = scan_symbol(symbol, preloaded_ticker=preloaded, scan_time=now)
        if row:
            rows.append(row)
        else:
            failed += 1

    if show_progress and symbols:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Scanning market...", total=len(symbols))
            for batch_start in range(0, len(symbols), BATCH_SIZE):
                batch = symbols[batch_start:batch_start + BATCH_SIZE]
                for sym in batch:
                    _process(sym)
                    progress.update(task, advance=1, description=f"Scanning {sym}...")
                if batch_start + BATCH_SIZE < len(symbols):
                    time.sleep(BATCH_DELAY)
    elif symbols:
        for batch_start in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[batch_start:batch_start + BATCH_SIZE]
            for sym in batch:
                _process(sym)
            if batch_start + BATCH_SIZE < len(symbols):
                time.sleep(BATCH_DELAY)
    else:
        symbols = []

    df = pd.DataFrame(rows)
    if not df.empty:
        date_str = now.strftime("%Y%m%d")
        raw_path = RAW_DIR / f"scans_{date_str}.csv"
        if raw_path.exists():
            df.to_csv(raw_path, mode="a", header=False, index=False)
        else:
            df.to_csv(raw_path, index=False)

    request_errors = get_request_error_summary()
    summary = {
        "timestamp": now.isoformat(),
        "status": "healthy" if not df.empty else "no_data",
        "symbols_attempted": int(len(symbols)),
        "symbols_scanned": int(len(rows)),
        "symbols_failed": int(failed),
        "min_volume_24h_usdt": MIN_VOLUME_24H_USDT,
        "request_errors": request_errors,
    }
    if not symbols:
        summary["status"] = "no_symbols"
    elif df.empty and request_errors.get("status_counts", {}).get("451"):
        summary["status"] = "blocked_exchange"
    elif df.empty and failed:
        summary["status"] = "degraded"

    with open(REPORTS_DIR / "latest_scan_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return df


def save_live_signals(scan_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if scan_df.empty:
        return scan_df
    ordered = scan_df.copy()
    if "ml_probability" in ordered.columns:
        ordered = ordered.sort_values(
            ["ml_probability", "volume_24h"], ascending=[False, False],
        )
    elif "volume_24h" in ordered.columns:
        ordered = ordered.sort_values("volume_24h", ascending=False)
    out = ordered.head(top_n).copy()
    out.to_csv(REPORTS_DIR / "latest_scan_signals.csv", index=False)
    out.to_json(REPORTS_DIR / "latest_scan_signals.json",
                orient="records", indent=2, date_format="iso")
    return out


def run_main_training(force: bool = False) -> dict:
    featured = run_feature_pipeline()
    if featured.empty:
        return {"status": "no_data"}
    return train_model(force=force)


# ──────────────────────────────────────────────────────────────────────────
# NEW v2.1: Multi-engine analysis stage
# ──────────────────────────────────────────────────────────────────────────
TOP_SYMBOLS_FOR_ENGINES = int(os.getenv("ENGINE_TOP_SYMBOLS", "12"))
"""Cap how many symbols get the heavyweight Azalyst analysis (rate limits)."""


def run_multi_engine(scan_df: pd.DataFrame) -> List[Dict]:
    """
    For each top-volume symbol with high ML probability, run the four
    leading-indicator engines and fuse them with the ML signal.
    Returns serializable signal dicts ranked Tier A → C.
    """
    if scan_df is None or scan_df.empty:
        return []

    # Pick candidate symbols: top by volume that also have meaningful ML prob
    df = scan_df.copy()
    if "ml_probability" in df.columns:
        df = df[df["ml_probability"].notna()]
    if "volume_24h" in df.columns:
        df = df.sort_values("volume_24h", ascending=False)

    # Always include BTC + ETH if present, even if not top by volume
    forced = [s for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT") if s in df["symbol"].values]
    candidates = list(dict.fromkeys(forced + df["symbol"].head(TOP_SYMBOLS_FOR_ENGINES).tolist()))
    candidates = candidates[:TOP_SYMBOLS_FOR_ENGINES]

    cg = DerivedDataClient()
    liq_engine = LiquidationProximityEngine(cg)
    fund_engine = FundingExtremeEngine(cg)
    ls_engine = LongShortExtremeEngine(cg)
    basis_engine = BasisEngine(fetch_spot_price, fetch_perp_price)
    oi_engine = OIDeltaEngine()
    fuser = DynamicSignalFuser()
    fuser.train_weights() # Attempt to load dynamic weights from history

    per_symbol_cards: Dict[str, list] = {}
    ml_probs: Dict[str, float] = {}

    for sym in candidates:
        row_match = df[df["symbol"] == sym]
        if row_match.empty:
            continue
        row = row_match.iloc[0].to_dict()
        ml_probs[sym] = float(row.get("ml_probability") or 0.5)

        # Trend Gating proxy: 24h momentum
        trend_score = np.clip(float(row.get("price_change_pct_24h") or 0) / 10, -1, 1)

        base_asset = sym[:-4] if sym.endswith("USDT") else sym
        cards = []
        cards.append(liq_engine.run(base_asset, "1d"))
        cards.append(fund_engine.run(base_asset, trend_strength=trend_score))
        cards.append(ls_engine.run(base_asset, "1h"))
        cards.append(basis_engine.run(sym))
        cards.append(oi_engine.run(row))
        per_symbol_cards[sym] = [c for c in cards if c is not None]

    fused = fuser.fuse_many(per_symbol_cards, ml_probs)

    # Persist engine outputs for the dashboard
    out = [s.to_dict() for s in fused]
    with open(REPORTS_DIR / "latest_fused_signals.json", "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "free_public_apis",
            "candidates": candidates,
            "signals": out,
        }, fh, indent=2, default=str)
    log.info("Fused %d/%d candidate symbols (free public APIs).",
             len(out), len(candidates))
             
    # Educational Frame Bridge: Copy relevant methodology frames for dashboard UI
    try:
        inject_educational_frames(out)
    except Exception as e:
        log.warning(f"Educational frame injection failed: {e}")
        
    return out

def inject_educational_frames(signals: List[Dict]):
    """Finds methodology frames matching the signals and copies them to reports/frames/"""
    manifest_path = Path(r"D:\Azalyst Bernd Skorupinski\_audit\manifest.json")
    if not manifest_path.exists():
        return

    frames_dir = REPORTS_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    lessons = manifest.get("lessons", [])
    
    for s in signals:
        # Map signal to keywords
        keywords = ["Candle", "Zone"]
        if s.get("direction") == "LONG":
            keywords.append("Demand")
        else:
            keywords.append("Supply")
            
        # Find a matching lesson
        matching_frame = None
        for L in lessons:
            if any(k.lower() in L.get("rel_path", "").lower() for k in keywords):
                frs = L.get("frames", [])
                if frs:
                    # Pick a frame from the middle
                    pick = frs[len(frs)//2]
                    src = Path(L.get("abs_dir")) / pick["file"]
                    if src.exists():
                        dest_name = f"{s['symbol']}_edu.jpg"
                        dest = frames_dir / dest_name
                        shutil.copy2(src, dest)
                        matching_frame = f"reports/frames/{dest_name}"
                        break
        
        if matching_frame:
            s["edu_frame"] = matching_frame


# ──────────────────────────────────────────────────────────────────────────
def _read_json(path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_text(path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _clean_nan(obj):
    """Recursively replace NaN with None for JSON compatibility."""
    import math
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nan(x) for x in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj



def write_runtime_payload(
    main_report=None, hourly_report=None, scan_signals=None,
    fused_signals: Optional[List[Dict]] = None,
    backtest: Optional[Dict] = None,
) -> Dict:
    scan_summary = _read_json(REPORTS_DIR / "latest_scan_summary.json") or {}
    latest_main = main_report or _read_json(LOGS_DIR / "latest_train_report.json")
    latest_hourly = hourly_report or _read_json(LOGS_DIR / "latest_hourly_train_report.json")

    latest_scan_signals = (
        scan_signals.to_dict(orient="records")
        if scan_signals is not None and not scan_signals.empty
        else (_read_json(REPORTS_DIR / "latest_scan_signals.json") or [])
    )
    latest_hourly_signals = _read_json(REPORTS_DIR / "hourly_live_signals.json") or []
    latest_fused = (fused_signals if fused_signals is not None else
                    (_read_json(REPORTS_DIR / "latest_fused_signals.json") or {}).get("signals", []))
    latest_backtest = backtest or _read_json(REPORTS_DIR / "latest_backtest.json")

    error_counts = ((scan_summary.get("request_errors") or {}).get("status_counts") or {})
    scanner_status = scan_summary.get("status", "unknown")
    notes = []
    if scanner_status == "blocked_exchange" or error_counts.get("451"):
        notes.append("Some exchanges returned 451; routed via KuCoin/Bitget/OKX fallback chain.")
    elif scanner_status == "healthy":
        notes.append("Scanner healthy this cycle.")

    runtime_status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dashboard_name": "Azalyst Crypto Futures Scanner v2.1",
        "source_mode": "github_actions" if os.getenv("GITHUB_ACTIONS") == "true" else "local",
        "data_source": "free_public_apis (KuCoin/Bitget/OKX — no paid keys, no blocked exchanges)",
        "scanner": scan_summary,
        "main_model": ({
            "status": latest_main.get("status"),
            "timestamp": latest_main.get("timestamp"),
            "accuracy": latest_main.get("accuracy"),
            "roc_auc": latest_main.get("roc_auc"),
            "samples": latest_main.get("n_samples"),
            "calibrated": latest_main.get("calibrated"),
            "primary_split": latest_main.get("primary_split"),
            "edge_over_baseline_pct": latest_main.get("edge_over_baseline_pct"),
        } if latest_main else {"status": "missing"}),
        "hourly_model": ({
            "status": latest_hourly.get("status"),
            "timestamp": latest_hourly.get("timestamp"),
            "accuracy": latest_hourly.get("accuracy"),
            "roc_auc": latest_hourly.get("roc_auc"),
            "samples": latest_hourly.get("n_samples"),
        } if latest_hourly else {"status": "missing"}),
        "backtest_summary": ({
            "n_trades": latest_backtest.get("n_trades"),
            "win_rate": latest_backtest.get("win_rate"),
            "profit_factor": latest_backtest.get("profit_factor"),
            "expectancy_pct": latest_backtest.get("expectancy_pct"),
            "sharpe_annualized": latest_backtest.get("sharpe_annualized"),
            "max_drawdown_pct": latest_backtest.get("max_drawdown_pct"),
            "total_return_pct": latest_backtest.get("total_return_pct"),
        } if latest_backtest else {"status": "no_backtest_run"}),
        "tier_a_signals": [s for s in latest_fused if s.get("consensus_tier") == "A"][:10],
        "tier_b_signals": [s for s in latest_fused if s.get("consensus_tier") == "B"][:10],
        "workflow_schedules": {"main_scanner": "*/15 * * * *", "hourly_patterns": "7 * * * *"},
        "notes": notes,
    }

    dashboard_payload = {
        "generated_at": runtime_status["generated_at"],
        "dashboard_name": runtime_status["dashboard_name"],
        "runtime_status": runtime_status,
        "main_report": latest_main,
        "hourly_report": latest_hourly,
        "latest_scan_signals": latest_scan_signals,
        "hourly_live_signals": latest_hourly_signals,
        "fused_signals": latest_fused,
        "backtest": latest_backtest,
        "summary_markdown": _read_text(REPORTS_DIR / "latest_summary.md"),
    }

    dashboard_payload = _clean_nan(dashboard_payload)

    with open(REPORTS_DIR / "latest_runtime_status.json", "w", encoding="utf-8") as fh:
        json.dump(_clean_nan(runtime_status), fh, indent=2)
    with open(REPORTS_DIR / "latest_dashboard_payload.json", "w", encoding="utf-8") as fh:
        json.dump(dashboard_payload, fh, indent=2)
    return dashboard_payload


def write_summary_markdown(
    main_report=None, hourly_report=None, scan_signals=None,
    fused_signals=None, backtest=None, summary_path: Optional[str] = None,
) -> str:

    def _table(frame: pd.DataFrame) -> List[str]:
        headers = list(frame.columns)
        rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
        return ["| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"] + \
               ["| " + " | ".join(row) + " |" for row in rows]

    lines = ["# Azalyst Crypto Intelligence v2.1 Summary", ""]

    if main_report and main_report.get("status") == "trained":
        lines += [
            "## Main Scanner Model (calibrated XGBoost, grouped split)",
            f"- Accuracy (grouped): **{main_report.get('accuracy')}** (baseline {main_report.get('baseline_accuracy')}, edge **{main_report.get('edge_over_baseline_pct'):+.1f}%**)",
            f"- F1 / ROC-AUC: {main_report.get('f1_score')} / {main_report.get('roc_auc')}",
            f"- Time-split (legacy): {main_report.get('time_split_metrics', {}).get('accuracy')}",
            f"- Samples / symbols: {main_report.get('n_samples')} / {main_report.get('n_symbols')}",
            f"- Calibrated: {main_report.get('calibrated')}, scale_pos_weight: {main_report.get('scale_pos_weight')}",
            "",
        ]

    if hourly_report and hourly_report.get("status") == "trained":
        lines += [
            "## Hourly Candle Pattern Model",
            f"- Accuracy: {hourly_report.get('accuracy')}, F1: {hourly_report.get('f1_score')}, AUC: {hourly_report.get('roc_auc')}",
            f"- Samples / symbols: {hourly_report.get('n_samples')} / {hourly_report.get('n_symbols')}",
            "",
        ]

    if backtest:
        lines += [
            "## Walk-Forward Backtest (PnL on historical signals)",
            f"- Trades: {backtest.get('n_trades')}, Win rate: {backtest.get('win_rate', 0)*100:.1f}%",
            f"- Profit factor: **{backtest.get('profit_factor')}**, Expectancy/trade: {backtest.get('expectancy_pct'):+.3f}%",
            f"- Sharpe (annualized): {backtest.get('sharpe_annualized')}",
            f"- Max DD: {backtest.get('max_drawdown_pct'):.2f}%, Total return: {backtest.get('total_return_pct'):+.2f}%",
            "",
        ]

    if fused_signals:
        tier_a = [s for s in fused_signals if s.get("consensus_tier") == "A"][:8]
        if tier_a:
            lines += ["## Tier-A Multi-Engine Signals", ""]
            tdf = pd.DataFrame([{
                "symbol": s["symbol"],
                "dir": s["direction"],
                "score": s["fused_score"],
                "engines_agree": len(s.get("engines_long" if s["direction"]=="LONG" else "engines_short", [])),
                "ml_prob": (f"{s['ml_probability']:.2f}" if s.get("ml_probability") is not None else "—"),
                "divergent": "yes" if s.get("divergent") else "",
            } for s in tier_a])
            lines += _table(tdf)
            lines.append("")

    if scan_signals is not None and not scan_signals.empty:
        lines += ["## Top Current Signals", ""]
        cols = [c for c in ["symbol", "price", "price_change_pct_24h", "ml_probability"]
                if c in scan_signals.columns]
        if cols:
            lines += _table(scan_signals[cols].head(10))
            lines.append("")

    md = "\n".join(lines).strip() + "\n"
    dest = summary_path or str(REPORTS_DIR / "latest_summary.md")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(md)

    gh_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as fh:
            fh.write(md)
    return md


# ──────────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────────
def run_scheduled_pipeline(
    train_main_model: bool = True,
    train_hourly_pattern_model: bool = True,
    force_main: bool = False,
    force_hourly: bool = False,
    show_progress: bool = False,
    summary_path: Optional[str] = None,
    run_engines: bool = True,
    run_backtest: bool = True,
    send_alerts: bool = True,
) -> dict:
    """v2.1 scheduled entry — preserves the v2.0 signature plus three new flags."""
    scan_df = scan_market_once(show_progress=show_progress)

    main_report = None
    top_scan_signals = pd.DataFrame()
    fused_signals: List[Dict] = []
    backtest_dict: Optional[Dict] = None

    if not scan_df.empty:
        if train_main_model:
            main_report = run_main_training(force=force_main)
        scored_scan = predict_current(scan_df.copy())
        top_scan_signals = save_live_signals(scored_scan)

        if run_engines:
            fused_signals = run_multi_engine(scored_scan)

            if send_alerts and fused_signals:
                from src.signal_fusion import FusedCryptoSignal
                # Rebuild FusedCryptoSignal objects to feed the alerter
                # (alerter accepts the dataclass for richer embeds)
                rebuilt = []
                for fs in fused_signals:
                    rebuilt.append(FusedCryptoSignal(
                        symbol=fs["symbol"],
                        direction=fs["direction"],
                        consensus_tier=fs["consensus_tier"],
                        fused_score=fs["fused_score"],
                        engines_long=fs.get("engines_long", []),
                        engines_short=fs.get("engines_short", []),
                        engines_neutral=fs.get("engines_neutral", []),
                        divergent=fs.get("divergent", False),
                        cards=[],   # alerter pulls from `reasons` field too
                        ml_probability=fs.get("ml_probability"),
                        ml_direction=fs.get("ml_direction"),
                        summary=fs.get("summary", ""),
                    ))
                alert_fused_signals(rebuilt)

        if run_backtest and main_report and main_report.get("status") == "trained":
            backtest_dict = WalkForwardBacktester().run().to_dict()

    hourly_report = None
    if train_hourly_pattern_model:
        hourly_report = train_hourly_model(force=force_hourly)

    md = write_summary_markdown(
        main_report=main_report, hourly_report=hourly_report,
        scan_signals=top_scan_signals, fused_signals=fused_signals,
        backtest=backtest_dict, summary_path=summary_path,
    )
    payload = write_runtime_payload(
        main_report=main_report, hourly_report=hourly_report,
        scan_signals=top_scan_signals, fused_signals=fused_signals,
        backtest=backtest_dict,
    )

    return {
        "scan_rows": int(len(scan_df)),
        "main_report": main_report,
        "hourly_report": hourly_report,
        "fused_signal_count": len(fused_signals),
        "tier_a_count": sum(1 for s in fused_signals if s.get("consensus_tier") == "A"),
        "backtest": backtest_dict,
        "runtime_status": payload.get("runtime_status"),
        "summary_markdown": md,
    }
