# Coinglass ML Scanner

This project scans USDT perpetual futures with Binance as the primary provider and Bybit as the fallback, builds ML features from RSI, price, volume, funding, and open interest, and trains two separate XGBoost models:

- `Main scanner model`: snapshot-based market scanner that scores current symbols using multi-timeframe RSI, OI, funding, and price features.
- `Hourly candle-pattern model`: 1h continuation model focused on large candle bodies, volume expansion, RSI persistence, and open-interest behavior.

## What Changed

The project is now structured to run on GitHub Actions instead of relying on one forever-running local loop.

- One-shot scan jobs replace the infinite scanner loop for automation.
- Labeling uses an actual time horizon instead of row-count shifts.
- Training saves feature medians so live inference matches training-time imputation.
- Hourly candle-pattern training is separate from the main scanner model.
- Runtime artifacts are persisted to a dedicated `runtime-data` branch.
- Manual workflow runs can reset runtime artifacts for a clean rebuild.

## Project Layout

```text
jobs.py                      CLI entrypoint for one-shot jobs
scanner.py                   Local loop runner
src/collector.py             Binance/Bybit market collection with fallback + proxy support
src/features.py              Main scanner feature engineering and time-based labels
src/trainer.py               Main scanner model training and live prediction
src/hourly_trainer.py        1h candle-pattern dataset, labels, training, reports
src/pipeline.py              Shared one-shot orchestration
.github/workflows/           GitHub Actions automation
```

## Models

### 1. Main Scanner Model

Inputs include:

- RSI on `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- RSI divergence and overbought/oversold counts
- OI change, OI direction, OI spike
- Price change and rolling volatility
- Funding rate

Target:

- Whether price rises at least `0.5%` around `60 minutes` after the scan snapshot

Outputs:

- Current market probabilities in `reports/latest_scan_signals.csv`
- Training reports in `logs/latest_train_report.json`
- Saved model in `models/latest_model.joblib`

### 2. Hourly Candle-Pattern Model

Inputs include:

- Candle body, range, wicks, close position
- Volume change and multi-hour volume expansion ratios
- RSI level and RSI persistence
- Open-interest change and OI expansion ratios
- Price/OI and price/volume alignment
- Relative candle size versus recent average

Target:

- Whether the current 1h candle continues in the same direction over the next few hours

Outputs:

- Hourly live pattern signals in `reports/hourly_live_signals.csv`
- Training reports in `logs/latest_hourly_train_report.json`
- Saved model in `models/latest_hourly_model.joblib`

## GitHub Actions

Two workflows are included:

- `Main Scanner`: runs every 15 minutes, scans the market, rebuilds features, retrains the main model, and writes current signals.
- `Hourly Candle Patterns`: runs hourly, rebuilds the 1h candle-pattern dataset, retrains the continuation model, and updates reports.

Generated runtime data is not kept on the code branch. Instead, workflows restore and persist:

- `data/`
- `logs/`
- `models/`
- `reports/`

through a separate `runtime-data` branch.

If exchanges block GitHub-hosted runners, you can still run on a self-hosted
runner by setting the repository variable `SCANNER_RUNNER` to a JSON runner
label array such as `["self-hosted","scanner"]`.

The workflows also support an outbound proxy for market-data requests. This
repo is configured for a single Oxylabs endpoint on `ddc.oxylabs.io:8001`, and
expects the GitHub secrets `OXYLABS_USERNAME` and `OXYLABS_PASSWORD`. If those
secrets are empty, the scanner will try the proxy first and then fall back to a
direct request.

## Dashboard

The web dashboard is published from `docs/` as:

- `https://gitdhirajs.github.io/coinglass-scanner/`

It surfaces:

- latest workflow status from GitHub Actions
- runtime health from the `runtime-data` branch
- local bootstrap model snapshots for the main and hourly ML models
- known operational issues such as data-provider blocking on hosted runners

## Local Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single automated cycle:

```bash
python jobs.py scheduled
```

Clear generated runtime artifacts and start fresh:

```bash
python jobs.py reset-runtime --yes
```

Run only the main scanner once:

```bash
python jobs.py scan-once --predict
python jobs.py train-main
```

Run only the hourly candle-pattern model:

```bash
python jobs.py train-hourly --force
```

Run the local forever-loop scanner:

```bash
python scanner.py
```

## Key Environment Variables

```bash
MARKET_DATA_PROVIDERS=binance,bybit
MARKET_PROXY_HOST=ddc.oxylabs.io
MARKET_PROXY_PORT=8001
MARKET_PROXY_USERNAME=...
MARKET_PROXY_PASSWORD=...
MARKET_PROXY_FALLBACK_DIRECT=true
MIN_VOLUME_24H_USDT=5000000
MAX_SYMBOLS_PER_SCAN=0
LABEL_HORIZON_MINUTES=60
LABEL_LOOKAHEAD_TOLERANCE_MINUTES=45
HOURLY_SYMBOL_LIMIT=120
HOURLY_KLINE_LIMIT=168
HOURLY_FORWARD_CANDLES=3
HOURLY_CONTINUATION_THRESHOLD_PCT=1.5
```

## Notes

- The scanner uses Binance USD-M public futures endpoints first, then Bybit public futures endpoints as fallback. No exchange API key is required.
- `data/`, `logs/`, `models/`, and `reports/` are intentionally ignored on the code branch.
- The dashboard remains useful locally, but GitHub Actions is now the primary execution path.
