#!/usr/bin/env python3
"""Command-line entrypoint for one-shot scan and training jobs."""

from __future__ import annotations

import argparse
import json

from src.config import validate as validate_config
from src.hourly_trainer import train_hourly_model
from src.pipeline import (
    reset_runtime_artifacts,
    run_main_training,
    run_scheduled_pipeline,
    scan_market_once,
    save_live_signals,
)
from src.trainer import predict_current


def main() -> None:
    validate_config()
    parser = argparse.ArgumentParser(description="Run scanner and ML jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan-once", help="Run one full market scan.")
    scan_parser.add_argument("--show-progress", action="store_true")
    scan_parser.add_argument("--predict", action="store_true")

    main_train_parser = subparsers.add_parser("train-main", help="Train the main scanner model.")
    main_train_parser.add_argument("--force", action="store_true")

    hourly_parser = subparsers.add_parser("train-hourly", help="Train the 1h candle-pattern model.")
    hourly_parser.add_argument("--force", action="store_true")
    hourly_parser.add_argument("--symbol-limit", type=int, default=None)
    hourly_parser.add_argument("--kline-limit", type=int, default=None)

    reset_parser = subparsers.add_parser(
        "reset-runtime",
        help="Clear generated data, logs, models, and reports for a fresh start.",
    )
    reset_parser.add_argument("--yes", action="store_true", help="Confirm deletion.")

    scheduled_parser = subparsers.add_parser(
        "scheduled",
        help="Run scan, main training, hourly training, and summary generation.",
    )
    scheduled_parser.add_argument("--skip-main", action="store_true")
    scheduled_parser.add_argument("--skip-hourly", action="store_true")
    scheduled_parser.add_argument("--force-main", action="store_true")
    scheduled_parser.add_argument("--force-hourly", action="store_true")
    scheduled_parser.add_argument("--show-progress", action="store_true")
    scheduled_parser.add_argument("--summary-path", default=None)

    args = parser.parse_args()

    if args.command == "scan-once":
        scan_df = scan_market_once(show_progress=args.show_progress)
        if args.predict:
            scan_df = predict_current(scan_df)
            save_live_signals(scan_df)
        print(json.dumps({"status": "ok", "rows": int(len(scan_df))}, indent=2))
        return

    if args.command == "train-main":
        print(json.dumps(run_main_training(force=args.force), indent=2, default=str))
        return

    if args.command == "train-hourly":
        print(
            json.dumps(
                train_hourly_model(
                    force=args.force,
                    symbol_limit=args.symbol_limit,
                    kline_limit=args.kline_limit,
                ),
                indent=2,
                default=str,
            )
        )
        return

    if args.command == "reset-runtime":
        if not args.yes:
            raise SystemExit("Refusing to reset runtime artifacts without --yes")
        print(json.dumps(reset_runtime_artifacts(), indent=2))
        return

    result = run_scheduled_pipeline(
        train_main_model=not args.skip_main,
        train_hourly_pattern_model=not args.skip_hourly,
        force_main=args.force_main,
        force_hourly=args.force_hourly,
        show_progress=args.show_progress,
        summary_path=args.summary_path,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
