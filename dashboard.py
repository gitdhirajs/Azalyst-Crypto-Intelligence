#!/usr/bin/env python3
"""Simple local dashboard for scanner and model artifacts."""

from __future__ import annotations

import json

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import logging
from src.config import FEATURES_DIR, HOURLY_DIR, LOGS_DIR, RAW_DIR

log = logging.getLogger(__name__)
console = Console()

def safe_read_csv(path, **kwargs) -> pd.DataFrame:
    try:
        if not path.exists():
            log.warning("Missing CSV: %s", path)
            return pd.DataFrame()
        df = pd.read_csv(path, **kwargs)
        if df.empty:
            log.warning("Empty CSV: %s", path)
        return df
    except Exception as e:
        log.exception("Failed to read %s: %s", path, e)
        return pd.DataFrame()


def show_data_summary() -> None:
    raw_files = sorted(RAW_DIR.glob("scans_*.csv"))
    total_rows = 0
    for file_path in raw_files:
        df_raw = safe_read_csv(file_path)
        if df_raw.empty:
            continue
        total_rows += len(df_raw)

    feat_path = FEATURES_DIR / "latest_features.csv"
    hourly_path = HOURLY_DIR / "latest_hourly_features.csv"

    labeled_main = 0
    labeled_hourly = 0
    
    df_feat = safe_read_csv(feat_path)
    if not df_feat.empty and "label" in df_feat.columns:
        labeled_main = int(df_feat["label"].notna().sum())
        
    df_hourly = safe_read_csv(hourly_path)
    if not df_hourly.empty and "continuation_label" in df_hourly.columns:
        labeled_hourly = int(df_hourly["continuation_label"].notna().sum())

    console.print(
        Panel(
            f"Raw scan files: {len(raw_files)}\n"
            f"Total scan rows: {total_rows:,}\n"
            f"Main model labeled rows: {labeled_main:,}\n"
            f"Hourly model labeled rows: {labeled_hourly:,}",
            title="Data Summary",
            border_style="cyan",
        )
    )


def _show_history(path_name: str, title: str) -> None:
    history_path = LOGS_DIR / path_name
    if not history_path.exists():
        console.print(f"[yellow]No history found for {title}.[/]")
        return

    table = Table(title=title, box=box.SIMPLE)
    table.add_column("Timestamp", width=20)
    table.add_column("Accuracy", justify="right", width=10)
    table.add_column("F1", justify="right", width=10)
    table.add_column("AUC", justify="right", width=10)
    table.add_column("Train", justify="right", width=8)
    table.add_column("Test", justify="right", width=8)

    for line in history_path.read_text(encoding="utf-8").splitlines():
        report = json.loads(line)
        if report.get("status") != "trained":
            continue
        table.add_row(
            report["timestamp"][:19],
            f"{report.get('accuracy', 0):.1%}",
            f"{report.get('f1_score', 0):.1%}",
            str(report.get("roc_auc", "n/a")),
            str(report.get("n_train", "n/a")),
            str(report.get("n_test", "n/a")),
        )

    console.print(table)


def main() -> None:
    console.print("[bold blue]CRYPTO SCANNER DASHBOARD[/]\n")
    show_data_summary()
    _show_history("training_history.jsonl", "Main Scanner Training History")
    _show_history("hourly_training_history.jsonl", "Hourly Candle Training History")


if __name__ == "__main__":
    main()
