"""Shared orchestration for one-shot scan and training jobs."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.collector import get_active_symbols, scan_symbol
from src.config import BATCH_DELAY, BATCH_SIZE, MIN_VOLUME_24H_USDT, RAW_DIR, REPORTS_DIR
from src.features import run_feature_pipeline
from src.hourly_trainer import train_hourly_model
from src.trainer import predict_current, train_model


def scan_market_once(show_progress: bool = True) -> pd.DataFrame:
    """Run a single full-market scan and persist the raw rows."""
    now = datetime.now(timezone.utc)
    symbols, bulk_tickers = get_active_symbols()
    if not symbols:
        return pd.DataFrame()

    rows = []
    failed = 0

    def _process_symbol(symbol: str) -> None:
        nonlocal failed
        preloaded = bulk_tickers.get(symbol)
        row = scan_symbol(symbol, preloaded_ticker=preloaded, scan_time=now)
        if row:
            rows.append(row)
        else:
            failed += 1

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Scanning market...", total=len(symbols))
            for batch_start in range(0, len(symbols), BATCH_SIZE):
                batch = symbols[batch_start : batch_start + BATCH_SIZE]
                for symbol in batch:
                    _process_symbol(symbol)
                    progress.update(task, advance=1, description=f"Scanning {symbol}...")
                if batch_start + BATCH_SIZE < len(symbols):
                    progress.console.print(
                        f"[dim]Rate-limit pause after batch ending at {batch[-1]}[/]"
                    )
                    time.sleep(BATCH_DELAY)
    else:
        for batch_start in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[batch_start : batch_start + BATCH_SIZE]
            for symbol in batch:
                _process_symbol(symbol)
            if batch_start + BATCH_SIZE < len(symbols):
                time.sleep(BATCH_DELAY)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    date_str = now.strftime("%Y%m%d")
    raw_path = RAW_DIR / f"scans_{date_str}.csv"
    if raw_path.exists():
        df.to_csv(raw_path, mode="a", header=False, index=False)
    else:
        df.to_csv(raw_path, index=False)

    summary = {
        "timestamp": now.isoformat(),
        "symbols_scanned": int(len(rows)),
        "symbols_failed": int(failed),
        "min_volume_24h_usdt": MIN_VOLUME_24H_USDT,
    }
    with open(REPORTS_DIR / "latest_scan_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return df


def save_live_signals(scan_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Persist the latest scan signals for GitHub reports and dashboards."""
    if scan_df.empty:
        return scan_df

    ordered = scan_df.copy()
    if "ml_probability" in ordered.columns:
        ordered = ordered.sort_values(
            ["ml_probability", "volume_24h"],
            ascending=[False, False],
        )
    elif "volume_24h" in ordered.columns:
        ordered = ordered.sort_values("volume_24h", ascending=False)

    out = ordered.head(top_n).copy()
    out.to_csv(REPORTS_DIR / "latest_scan_signals.csv", index=False)
    out.to_json(
        REPORTS_DIR / "latest_scan_signals.json",
        orient="records",
        indent=2,
        date_format="iso",
    )
    return out


def run_main_training(force: bool = False) -> dict:
    """Build scanner features and train the main model."""
    featured = run_feature_pipeline()
    if featured.empty:
        return {"status": "no_data"}
    return train_model(force=force)


def write_summary_markdown(
    main_report: dict | None = None,
    hourly_report: dict | None = None,
    scan_signals: pd.DataFrame | None = None,
    summary_path: str | None = None,
) -> str:
    """Create a compact markdown summary for GitHub or local review."""

    def _markdown_table(frame: pd.DataFrame) -> list[str]:
        headers = list(frame.columns)
        rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        header_line = "| " + " | ".join(headers) + " |"
        divider = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows]
        return [header_line, divider, *body]

    lines = ["# Coinglass ML Scanner Summary", ""]

    if main_report:
        lines.extend(
            [
                "## Main Scanner Model",
                f"- Status: {main_report.get('status')}",
                f"- Accuracy: {main_report.get('accuracy', 'n/a')}",
                f"- F1: {main_report.get('f1_score', 'n/a')}",
                f"- ROC AUC: {main_report.get('roc_auc', 'n/a')}",
                f"- Samples: {main_report.get('n_samples', 'n/a')}",
                f"- Symbols: {main_report.get('n_symbols', 'n/a')}",
                f"- Label horizon minutes: {main_report.get('label_horizon_minutes', 'n/a')}",
                "",
            ]
        )

    if hourly_report:
        lines.extend(
            [
                "## Hourly Candle Pattern Model",
                f"- Status: {hourly_report.get('status')}",
                f"- Accuracy: {hourly_report.get('accuracy', 'n/a')}",
                f"- F1: {hourly_report.get('f1_score', 'n/a')}",
                f"- ROC AUC: {hourly_report.get('roc_auc', 'n/a')}",
                f"- Samples: {hourly_report.get('n_samples', 'n/a')}",
                f"- Symbols: {hourly_report.get('n_symbols', 'n/a')}",
                f"- Continuation window hours: {hourly_report.get('continuation_window_hours', 'n/a')}",
                "",
            ]
        )

    if scan_signals is not None and not scan_signals.empty:
        lines.extend(["## Top Current Signals", ""])
        preview_cols = [
            column
            for column in ["symbol", "price", "price_change_pct_24h", "ml_probability"]
            if column in scan_signals.columns
        ]
        if preview_cols:
            lines.extend(_markdown_table(scan_signals[preview_cols].head(10)))
            lines.append("")

    markdown = "\n".join(lines).strip() + "\n"
    destination = summary_path or str(REPORTS_DIR / "latest_summary.md")
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    github_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)

    return markdown


def run_scheduled_pipeline(
    train_main_model: bool = True,
    train_hourly_pattern_model: bool = True,
    force_main: bool = False,
    force_hourly: bool = False,
    show_progress: bool = False,
    summary_path: str | None = None,
) -> dict:
    """Run the GitHub-friendly one-shot scanner and model training jobs."""
    scan_df = scan_market_once(show_progress=show_progress)

    main_report = None
    top_scan_signals = pd.DataFrame()
    if not scan_df.empty:
        if train_main_model:
            main_report = run_main_training(force=force_main)
        scored_scan = predict_current(scan_df.copy())
        top_scan_signals = save_live_signals(scored_scan)

    hourly_report = None
    if train_hourly_pattern_model:
        hourly_report = train_hourly_model(force=force_hourly)

    markdown = write_summary_markdown(
        main_report=main_report,
        hourly_report=hourly_report,
        scan_signals=top_scan_signals,
        summary_path=summary_path,
    )

    return {
        "scan_rows": int(len(scan_df)),
        "main_report": main_report,
        "hourly_report": hourly_report,
        "summary_markdown": markdown,
    }
