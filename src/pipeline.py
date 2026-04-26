"""Shared orchestration for one-shot scan and training jobs."""

from __future__ import annotations

import json
import os
import shutil
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

from src.collector import (
    get_active_symbols,
    get_request_error_summary,
    reset_request_errors,
    scan_symbol,
)
from src.config import (
    BATCH_DELAY,
    BATCH_SIZE,
    DATA_DIR,
    FEATURES_DIR,
    HOURLY_DIR,
    LABELS_DIR,
    LOGS_DIR,
    MARKET_DATA_PROVIDERS,
    MARKET_PROXY_HOST,
    MARKET_PROXY_PORT,
    MARKET_PROXY_URL,
    MIN_VOLUME_24H_USDT,
    MODELS_DIR,
    RAW_DIR,
    REPORTS_DIR,
)
from src.features import run_feature_pipeline
from src.hourly_trainer import train_hourly_model
from src.trainer import predict_current, train_model


def _has_access_block(error_counts: dict) -> bool:
    return bool(error_counts.get("403") or error_counts.get("451"))


def reset_runtime_artifacts() -> dict:
    """Delete generated runtime artifacts and recreate empty runtime directories."""
    removed = []
    for path in [DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))

    for path in [
        RAW_DIR,
        FEATURES_DIR,
        HOURLY_DIR,
        LABELS_DIR,
        LOGS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return {"status": "reset", "removed": removed}


def scan_market_once(show_progress: bool = True) -> pd.DataFrame:
    """Run a single full-market scan and persist the raw rows."""
    now = datetime.now(timezone.utc)
    reset_request_errors()
    symbols, bulk_tickers = get_active_symbols()
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

    if show_progress and symbols:
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
    elif symbols:
        for batch_start in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[batch_start : batch_start + BATCH_SIZE]
            for symbol in batch:
                _process_symbol(symbol)
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
        "headline": "Market scan completed successfully." if not df.empty else "No scan rows were collected.",
        "symbols_attempted": int(len(symbols)),
        "symbols_scanned": int(len(rows)),
        "symbols_failed": int(failed),
        "min_volume_24h_usdt": MIN_VOLUME_24H_USDT,
        "provider_priority": MARKET_DATA_PROVIDERS,
        "proxy": {
            "enabled": bool(MARKET_PROXY_URL),
            "endpoint": (
                f"{MARKET_PROXY_HOST}:{MARKET_PROXY_PORT}"
                if MARKET_PROXY_HOST and MARKET_PROXY_PORT
                else None
            ),
        },
        "request_errors": request_errors,
    }
    if not symbols:
        summary["status"] = "no_symbols"
        summary["headline"] = "No active symbols were available for scanning."
    elif df.empty and _has_access_block(request_errors.get("status_counts", {})):
        summary["status"] = "blocked"
        summary["headline"] = "Configured market-data providers rejected requests from the runner."
    elif df.empty and failed:
        summary["status"] = "degraded"
        summary["headline"] = "All symbol scans failed before producing usable rows."

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


def _read_json(path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_text(path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_runtime_payload(
    main_report: dict | None = None,
    hourly_report: dict | None = None,
    scan_signals: pd.DataFrame | None = None,
) -> dict:
    """Persist machine-readable runtime health for the web dashboard."""
    scan_summary = _read_json(REPORTS_DIR / "latest_scan_summary.json") or {}
    latest_main = main_report or _read_json(LOGS_DIR / "latest_train_report.json")
    latest_hourly = hourly_report or _read_json(LOGS_DIR / "latest_hourly_train_report.json")

    latest_scan_signals = (
        scan_signals.to_dict(orient="records")
        if scan_signals is not None and not scan_signals.empty
        else (_read_json(REPORTS_DIR / "latest_scan_signals.json") or [])
    )
    latest_hourly_signals = _read_json(REPORTS_DIR / "hourly_live_signals.json") or []

    error_counts = (
        (scan_summary.get("request_errors") or {}).get("status_counts")
        if isinstance(scan_summary, dict)
        else {}
    ) or {}
    scanner_status = scan_summary.get("status", "unknown") if isinstance(scan_summary, dict) else "unknown"
    notes = []

    if scanner_status == "blocked" or _has_access_block(error_counts):
        notes.append(
            "Configured market-data providers are rejecting requests from this runner, so automated scans are not collecting live rows."
        )
    elif scanner_status == "degraded":
        notes.append("The scanner ran but did not produce usable rows for the latest cycle.")
    elif scanner_status == "healthy":
        notes.append("The scanner completed and produced live market rows for the latest cycle.")

    if MARKET_DATA_PROVIDERS:
        notes.append(f"Provider order: {' -> '.join(MARKET_DATA_PROVIDERS)}.")
    if MARKET_PROXY_URL:
        if MARKET_PROXY_HOST and MARKET_PROXY_PORT:
            notes.append(f"Outbound proxy enabled via {MARKET_PROXY_HOST}:{MARKET_PROXY_PORT}.")
        else:
            notes.append("Outbound proxy is enabled for market-data requests.")

    if latest_main and latest_main.get("status") == "trained":
        notes.append("Main scanner model metrics below come from the most recent successful training snapshot.")
    if latest_hourly and latest_hourly.get("status") == "trained":
        notes.append("Hourly candle-pattern model metrics below come from the most recent successful training snapshot.")

    runtime_status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dashboard_name": "Coinglass Scanner",
        "source_mode": "github_actions" if os.getenv("GITHUB_ACTIONS") == "true" else "local",
        "scanner": scan_summary,
        "main_model": {
            "status": latest_main.get("status") if latest_main else "missing",
            "timestamp": latest_main.get("timestamp") if latest_main else None,
            "accuracy": latest_main.get("accuracy") if latest_main else None,
            "roc_auc": latest_main.get("roc_auc") if latest_main else None,
            "samples": latest_main.get("n_samples") if latest_main else None,
        },
        "hourly_model": {
            "status": latest_hourly.get("status") if latest_hourly else "missing",
            "timestamp": latest_hourly.get("timestamp") if latest_hourly else None,
            "accuracy": latest_hourly.get("accuracy") if latest_hourly else None,
            "roc_auc": latest_hourly.get("roc_auc") if latest_hourly else None,
            "samples": latest_hourly.get("n_samples") if latest_hourly else None,
        },
        "workflow_schedules": {
            "main_scanner": "*/15 * * * *",
            "hourly_patterns": "7 * * * *",
        },
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
        "summary_markdown": _read_text(REPORTS_DIR / "latest_summary.md"),
    }

    with open(REPORTS_DIR / "latest_runtime_status.json", "w", encoding="utf-8") as handle:
        json.dump(runtime_status, handle, indent=2, default=str)

    with open(REPORTS_DIR / "latest_dashboard_payload.json", "w", encoding="utf-8") as handle:
        json.dump(dashboard_payload, handle, indent=2, default=str)

    return dashboard_payload


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
    dashboard_payload = write_runtime_payload(
        main_report=main_report,
        hourly_report=hourly_report,
        scan_signals=top_scan_signals,
    )

    return {
        "scan_rows": int(len(scan_df)),
        "main_report": main_report,
        "hourly_report": hourly_report,
        "runtime_status": dashboard_payload.get("runtime_status"),
        "summary_markdown": markdown,
    }
