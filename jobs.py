#!/usr/bin/env python3
"""Command-line entrypoint for v2.1 multi-engine scanner jobs."""

from __future__ import annotations

import argparse
import json

from src.backtester import run_backtest
from src.hourly_trainer import train_hourly_model
from src.pipeline import (
    run_main_training,
    run_multi_engine,
    run_scheduled_pipeline,
    save_live_signals,
    scan_market_once,
)
from src.trainer import predict_current


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Azalyst Crypto Scanner v2.1")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── Existing commands ────────────────────────────────────────────────
    p_scan = sub.add_parser("scan-once", help="One full market scan (Binance + fallback).")
    p_scan.add_argument("--show-progress", action="store_true")
    p_scan.add_argument("--predict", action="store_true")

    p_main = sub.add_parser("train-main", help="Train the calibrated main model.")
    p_main.add_argument("--force", action="store_true")

    p_hour = sub.add_parser("train-hourly", help="Train the 1h candle-pattern model.")
    p_hour.add_argument("--force", action="store_true")
    p_hour.add_argument("--symbol-limit", type=int, default=None)
    p_hour.add_argument("--kline-limit", type=int, default=None)

    p_sched = sub.add_parser("scheduled", help="Full v2.1 cycle: scan + train + engines + alerts + backtest.")
    p_sched.add_argument("--skip-main", action="store_true")
    p_sched.add_argument("--skip-hourly", action="store_true")
    p_sched.add_argument("--skip-engines", action="store_true")
    p_sched.add_argument("--skip-backtest", action="store_true")
    p_sched.add_argument("--skip-alerts", action="store_true")
    p_sched.add_argument("--force-main", action="store_true")
    p_sched.add_argument("--force-hourly", action="store_true")
    p_sched.add_argument("--show-progress", action="store_true")
    p_sched.add_argument("--summary-path", default=None)

    # ── NEW v2.1 commands ────────────────────────────────────────────────
    p_eng = sub.add_parser("engines",
        help="Run multi-engine analysis (liq, funding, L/S, basis, OI) and fuse.")
    p_eng.add_argument("--with-alerts", action="store_true",
                       help="Push Tier-A signals to Discord/TG.")

    p_bt = sub.add_parser("backtest",
        help="Walk-forward backtest the trained model. Prints PnL/Sharpe/MDD.")
    p_bt.add_argument("--threshold", type=float, default=0.62,
                      help="Long entry probability threshold (default 0.62).")
    p_bt.add_argument("--stop-loss", type=float, default=1.0)
    p_bt.add_argument("--take-profit", type=float, default=2.0)
    p_bt.add_argument("--cycle-minutes", type=int, default=5)

    p_alert_test = sub.add_parser("test-alert",
        help="Send a synthetic Tier-A alert to verify webhooks.")

    args = parser.parse_args()

    # ── Dispatch ─────────────────────────────────────────────────────────
    if args.command == "scan-once":
        scan_df = scan_market_once(show_progress=args.show_progress)
        if args.predict:
            scan_df = predict_current(scan_df)
            save_live_signals(scan_df)
        _print({"status": "ok", "rows": int(len(scan_df))})
        return

    if args.command == "train-main":
        _print(run_main_training(force=args.force))
        return

    if args.command == "train-hourly":
        _print(train_hourly_model(
            force=args.force,
            symbol_limit=args.symbol_limit,
            kline_limit=args.kline_limit,
        ))
        return

    if args.command == "scheduled":
        _print(run_scheduled_pipeline(
            train_main_model=not args.skip_main,
            train_hourly_pattern_model=not args.skip_hourly,
            run_engines=not args.skip_engines,
            run_backtest=not args.skip_backtest,
            send_alerts=not args.skip_alerts,
            force_main=args.force_main,
            force_hourly=args.force_hourly,
            show_progress=args.show_progress,
            summary_path=args.summary_path,
        ))
        return

    if args.command == "engines":
        # Run a fresh scan, predict, then engines
        scan_df = scan_market_once(show_progress=False)
        if scan_df.empty:
            _print({"status": "no_scan_rows"})
            return
        scan_df = predict_current(scan_df)
        save_live_signals(scan_df)
        fused = run_multi_engine(scan_df)
        if args.with_alerts and fused:
            from src.alerter import alert_fused_signals
            from src.signal_fusion import FusedCryptoSignal
            rebuilt = [
                FusedCryptoSignal(
                    symbol=fs["symbol"], direction=fs["direction"],
                    consensus_tier=fs["consensus_tier"],
                    fused_score=fs["fused_score"],
                    engines_long=fs.get("engines_long", []),
                    engines_short=fs.get("engines_short", []),
                    engines_neutral=fs.get("engines_neutral", []),
                    divergent=fs.get("divergent", False),
                    cards=[], ml_probability=fs.get("ml_probability"),
                    ml_direction=fs.get("ml_direction"),
                    summary=fs.get("summary", ""),
                ) for fs in fused
            ]
            alert_fused_signals(rebuilt)
        _print({"status": "ok", "fused_count": len(fused),
                "tier_a": sum(1 for s in fused if s["consensus_tier"] == "A"),
                "tier_b": sum(1 for s in fused if s["consensus_tier"] == "B"),
                "signals": fused})
        return

    if args.command == "backtest":
        _print(run_backtest(
            long_threshold=args.threshold,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            cycle_minutes=args.cycle_minutes,
        ))
        return

    if args.command == "test-alert":
        from src.alerter import alert_fused_signals
        from src.signal_fusion import FusedCryptoSignal
        synthetic = FusedCryptoSignal(
            symbol="BTCUSDT", direction="LONG", consensus_tier="A",
            fused_score=82.5,
            engines_long=["liq_proximity", "ml_main", "funding_extreme", "ls_extreme"],
            engines_short=[], engines_neutral=["basis"],
            divergent=False, cards=[], ml_probability=0.74,
            ml_direction="LONG",
            summary="TEST ALERT — synthetic Tier-A signal to validate webhook.",
        )
        stats = alert_fused_signals([synthetic])
        _print(stats)
        return


if __name__ == "__main__":
    main()
