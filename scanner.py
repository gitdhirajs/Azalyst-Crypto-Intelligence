#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  AZALYST CRYPTO INTELLIGENCE v2.1                                 ║
║  Full-Market RSI × OI × Price Action ML Trainer              ║
║                                                              ║
║  Dynamically scans KuCoin, Bitget, and OKX USDT perps,       ║
║  giving the ML model proper training data                    ║
║  across the entire market.                                   ║
╚══════════════════════════════════════════════════════════════╝
"""
import sys
import time
import signal
import traceback
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

from src.config import (
    SCAN_INTERVAL_SECONDS, RAW_DIR, FEATURES_DIR,
    RETRAIN_EVERY_N_SCANS, MIN_SAMPLES_TO_TRAIN,
    BATCH_SIZE, BATCH_DELAY, MIN_VOLUME_24H_USDT,
)
from src.collector import scan_symbol, get_active_symbols
from src.features import run_feature_pipeline
from src.trainer import train_model, predict_current
from src.pipeline import write_runtime_payload, save_live_signals

console = Console()
SCAN_COUNT = 0
RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    console.print("\n[bold red]⏹  Shutting down scanner...[/]")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_scan() -> pd.DataFrame:
    """Dynamically discover ALL symbols, scan in batches, return DataFrame."""
    global SCAN_COUNT
    SCAN_COUNT += 1
    now = datetime.now(timezone.utc)

    console.print(f"\n[bold cyan]━━━ SCAN #{SCAN_COUNT}  @  {now.strftime('%Y-%m-%d %H:%M:%S UTC')} ━━━[/]")

    # Step 1: Discover all active symbols
    console.print("  [dim]Fetching exchange active contracts (KuCoin/Bitget)...[/]")
    symbols, bulk_tickers = get_active_symbols()

    if not symbols:
        console.print("[yellow]  No symbols discovered. Check network.[/]")
        return pd.DataFrame()

    console.print(
        f"  [green]✓ Found {len(symbols)} tradeable pairs[/] "
        f"[dim](filtered by >${MIN_VOLUME_24H_USDT/1e6:.0f}M daily volume)[/]"
    )

    # Step 2: Scan in batches with progress bar
    rows = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning...", total=len(symbols))

        for batch_start in range(0, len(symbols), BATCH_SIZE):
            if not RUNNING:
                break

            batch = symbols[batch_start:batch_start + BATCH_SIZE]

            for sym in batch:
                if not RUNNING:
                    break

                # Pass preloaded ticker data to avoid redundant API calls
                preloaded = bulk_tickers.get(sym)
                row = scan_symbol(sym, preloaded_ticker=preloaded, scan_time=now)
                if row:
                    rows.append(row)
                else:
                    failed += 1

                progress.update(task, advance=1, description=f"Scanning {sym}...")

            # Pause between batches to respect rate limits
            if batch_start + BATCH_SIZE < len(symbols) and RUNNING:
                time.sleep(BATCH_DELAY)

    if not rows:
        console.print("[yellow]  No data collected this cycle.[/]")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Save raw scan
    date_str = now.strftime("%Y%m%d")
    raw_path = RAW_DIR / f"scans_{date_str}.csv"
    if raw_path.exists():
        df.to_csv(raw_path, mode="a", header=False, index=False)
    else:
        df.to_csv(raw_path, index=False)

    console.print(
        f"  [green]✓ Scanned {len(rows)} symbols[/] "
        f"[dim]({failed} failed) → {raw_path.name}[/]"
    )
    return df


def display_scan_table(df: pd.DataFrame, top_n: int = 50):
    """Pretty-print current scan results (top N by volume)."""
    if df.empty:
        return

    # Sort by volume and show top N
    if "volume_24h" in df.columns:
        df_sorted = df.sort_values("volume_24h", ascending=False).head(top_n)
    else:
        df_sorted = df.head(top_n)

    table = Table(
        title=f"LIVE SCAN RESULTS  ({len(df)} total, showing top {min(top_n, len(df))})",
        box=box.ROUNDED,
        title_style="bold white on blue",
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Symbol", style="bold white", width=10)
    table.add_column("Price", justify="right", width=12)
    table.add_column("24h%", justify="right", width=8)
    table.add_column("Vol$M", justify="right", width=8)
    table.add_column("RSI 5m", justify="right", width=7)
    table.add_column("RSI 15m", justify="right", width=8)
    table.add_column("RSI 1h", justify="right", width=7)
    table.add_column("RSI 4h", justify="right", width=7)
    table.add_column("OI Δ1h%", justify="right", width=9)
    table.add_column("Funding", justify="right", width=9)
    if "ml_probability" in df.columns:
        table.add_column("ML Prob", justify="right", width=8)

    for _, r in df_sorted.iterrows():
        def rsi_style(val):
            if val is None or pd.isna(val):
                return "[dim]—[/]"
            v = float(val)
            if v <= 30:
                return f"[bold green]{v:.1f}[/]"
            elif v >= 70:
                return f"[bold red]{v:.1f}[/]"
            elif v >= 60:
                return f"[yellow]{v:.1f}[/]"
            else:
                return f"{v:.1f}"

        chg_24 = r.get("price_change_pct_24h", 0)
        chg_style = "green" if chg_24 >= 0 else "red"

        vol_m = r.get("volume_24h", 0) / 1e6
        vol_str = f"{vol_m:,.0f}" if vol_m >= 1 else f"{vol_m:.1f}"

        oi_chg = r.get("oi_change_pct_1h")
        oi_str = f"{oi_chg:+.2f}" if oi_chg and not pd.isna(oi_chg) else "—"

        funding = r.get("funding_rate")
        fund_str = f"{funding*100:.4f}%" if funding and not pd.isna(funding) else "—"

        row_data = [
            r["symbol"].replace("USDT", ""),
            f"${r['price']:,.4f}" if r["price"] < 1 else f"${r['price']:,.2f}",
            f"[{chg_style}]{chg_24:+.2f}%[/]",
            vol_str,
            rsi_style(r.get("rsi_5m")),
            rsi_style(r.get("rsi_15m")),
            rsi_style(r.get("rsi_1h")),
            rsi_style(r.get("rsi_4h")),
            oi_str,
            fund_str,
        ]
        if "ml_probability" in df.columns:
            prob = r.get("ml_probability")
            if prob is not None and not pd.isna(prob):
                prob_style = "bold green" if prob > 0.6 else ("yellow" if prob > 0.4 else "red")
                row_data.append(f"[{prob_style}]{prob:.2f}[/]")
            else:
                row_data.append("[dim]—[/]")

        table.add_row(*row_data)

    console.print(table)

    # Summary alerts across ALL scanned symbols (not just displayed)
    if "rsi_5m" in df.columns:
        oversold = df[df["rsi_5m"] < 30]
        overbought = df[df["rsi_5m"] > 70]
        if len(oversold) > 0:
            syms = ", ".join(oversold["symbol"].str.replace("USDT", "").tolist()[:15])
            extra = f" (+{len(oversold)-15} more)" if len(oversold) > 15 else ""
            console.print(f"  [bold green]🟢 OVERSOLD (RSI 5m < 30):[/] {syms}{extra}")
        if len(overbought) > 0:
            syms = ", ".join(overbought["symbol"].str.replace("USDT", "").tolist()[:15])
            extra = f" (+{len(overbought)-15} more)" if len(overbought) > 15 else ""
            console.print(f"  [bold red]🔴 OVERBOUGHT (RSI 5m > 70):[/] {syms}{extra}")

    # ML high-confidence alerts
    if "ml_probability" in df.columns:
        high_conf = df[df["ml_probability"] > 0.7].sort_values("ml_probability", ascending=False)
        if len(high_conf) > 0:
            syms = ", ".join(
                f"{r['symbol'].replace('USDT','')}({r['ml_probability']:.0%})"
                for _, r in high_conf.head(10).iterrows()
            )
            console.print(f"  [bold magenta]🧠 ML HIGH CONFIDENCE (>70%):[/] {syms}")


def maybe_train_model():
    """Retrain ML model if enough data and it's time."""
    if SCAN_COUNT % RETRAIN_EVERY_N_SCANS != 0 and SCAN_COUNT != 1:
        return

    console.print("\n[bold magenta]🧠 Running feature pipeline & ML training...[/]")
    featured = run_feature_pipeline()
    if featured.empty:
        console.print("  [yellow]No feature data yet.[/]")
        return

    n_labeled = featured["label"].notna().sum()
    n_symbols = featured["symbol"].nunique()
    console.print(f"  Total rows: {len(featured)}, Labeled: {n_labeled}, Unique symbols: {n_symbols}")

    report = train_model()
    status = report.get("status")

    if status == "trained":
        console.print(Panel(
            f"[green]✓ Model trained![/]\n"
            f"  Accuracy: {report['accuracy']:.1%}\n"
            f"  F1 Score: {report['f1_score']:.1%}\n"
            f"  ROC AUC: {report.get('roc_auc', 'N/A')}\n"
            f"  Train: {report['n_train']}  Test: {report['n_test']}\n"
            f"  Symbols in dataset: {report.get('n_symbols', '?')}\n"
            f"  Top features: {', '.join(f[0] for f in report['top_features'][:5])}",
            title="ML Training Report",
            border_style="green",
        ))
    elif status == "not_enough_data":
        console.print(
            f"  [yellow]Need {report['need']} labeled samples, have {report['n_samples']}. "
            f"Keep scanning![/]"
        )
    else:
        console.print(f"  [yellow]Training status: {status}[/]")


def main():
    # Discover symbols count for startup display
    console.print("[dim]Checking exchanges for available pairs...[/]")
    startup_symbols, _ = get_active_symbols()
    n_syms = len(startup_symbols)

    console.print(Panel(
        f"[bold white]AZALYST CRYPTO INTELLIGENCE v2.1[/]\n"
        f"[dim]Full-Market RSI × OI × Price Action ML[/]\n\n"
        f"Mode: [bold green]DYNAMIC[/] — scans KuCoin/Bitget/OKX USDT perps\n"
        f"Active pairs found: [bold cyan]{n_syms}[/]\n"
        f"Volume filter: >${MIN_VOLUME_24H_USDT/1e6:.0f}M daily\n"
        f"Scan interval: {SCAN_INTERVAL_SECONDS}s ({SCAN_INTERVAL_SECONDS//60}min)\n"
        f"Timeframes: 1m, 5m, 15m, 1h, 4h, 1d\n"
        f"Batch size: {BATCH_SIZE} symbols per batch\n"
        f"Auto-retrain every {RETRAIN_EVERY_N_SCANS} scans\n"
        f"Min samples for ML: {MIN_SAMPLES_TO_TRAIN}",
        title="━━━ CONFIG ━━━",
        border_style="blue",
    ))

    while RUNNING:
        try:
            # 1. Discover & scan all active symbols
            scan_df = run_scan()

            if not scan_df.empty:
                # 2. Build features & maybe train model
                maybe_train_model()

                # 3. Run predictions on current scan (if model exists)
                scan_df = predict_current(scan_df)

                # 4. Display results
                display_scan_table(scan_df)

                # 5. Update dashboard JSON payloads
                save_live_signals(scan_df)
                write_runtime_payload(scan_signals=scan_df)

            # 5. Wait for next cycle
            if RUNNING:
                console.print(
                    f"\n[dim]Next scan in {SCAN_INTERVAL_SECONDS}s... "
                    f"(Ctrl+C to stop)[/]"
                )
                for _ in range(SCAN_INTERVAL_SECONDS):
                    if not RUNNING:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]ERROR: {e}[/]")
            traceback.print_exc()
            console.print("[yellow]Retrying in 30s...[/]")
            time.sleep(30)

    # Graceful shutdown
    console.print("\n[bold green]Final feature pipeline run...[/]")
    run_feature_pipeline()
    console.print("[bold green]✓ Scanner stopped. All data saved.[/]")


if __name__ == "__main__":
    main()
