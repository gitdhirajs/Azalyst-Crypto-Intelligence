# Azalyst Crypto Intelligence

A multi-engine USDT-perpetual-futures scanner for institutional-grade crypto signal generation. Runs on **only free public APIs** (KuCoin / Bitget / OKX), works on GitHub Actions free-tier, and combines a calibrated XGBoost ML model with five independent leading-indicator engines fused into Tier-A/B/C consensus signals.

**Live dashboard:** https://gitdhirajs.github.io/Azalyst-Crypto-Intelligence/
**Status:** v2.1 (multi-engine, GitHub-Actions-native, zero paid dependencies)

---

## What it does, in one paragraph

Every 5 minutes it scans ~150 USDT-perpetual contracts on KuCoin (with Bitget/OKX fallback), computes multi-timeframe RSI / open-interest / funding-rate features, runs them through a calibrated XGBoost classifier predicting +0.5% moves over the next 60 minutes, then for the top-12 candidates it runs five additional engines — liquidation-cluster proximity, cross-exchange funding extremes, top-trader vs retail long/short divergence, perp-vs-spot basis, and OI×price quadrant — fuses everything into a consensus tier (A/B/C), pushes Tier-A signals to Discord, and walk-forward backtests the model after every cycle. No Azalyst key. No Binance API key. No Bybit dependency. The data plumbing was specifically built around exchanges that respond to GitHub-runner IPs.

---

## Why this scanner exists (vs the dozens of others on GitHub)

1. **It actually works on GitHub Actions free-tier.** Most public crypto scanners fail silently because Binance and Bybit return HTTP 451 to GCP/AWS runner IPs. This one uses KuCoin Futures as primary, Bitget as backup, OKX as tertiary — all of which answer GitHub-hosted IPs.
2. **No paid APIs.** Azalyst-equivalent data (liquidation heatmaps, cross-exchange funding aggregates, long/short ratios, taker flow) is reconstructed from public endpoints. Total monthly cost: $0.
3. **Multi-engine consensus, not single-model.** A 60% accurate XGBoost model that says LONG at the same time a liquidation cluster, funding squeeze, and L/S divergence all say LONG carries massively more conviction than the model alone.
4. **Walk-forward backtest in every cycle.** Accuracy and F1 don't tell you if the strategy makes money. The backtester does.
5. **Calibrated probabilities.** A `predict_proba` of 0.7 actually means ~70% real-world hit rate after `CalibratedClassifierCV` (isotonic, cv=3).

---

## Architecture

```
                       ┌───────────────────────────────────────────┐
                       │  KuCoin Futures (PRIMARY) ── GHA-friendly │
                       │  Bitget Futures (BACKUP)                  │
                       │  OKX SWAP        (TERTIARY)               │
                       └───────────────────┬───────────────────────┘
                                           │  every 5 min
                                           ▼
                       ┌───────────────────────────────────────────┐
                       │  scan_market_once()  — ~150 USDT perps    │
                       │  • multi-TF RSI (1m/5m/15m/1h/4h/1d)      │
                       │  • OI snapshot                            │
                       │  • funding rate                           │
                       └───────────┬───────────────────────────────┘
                                   │
                                   ├── DerivedDataClient enrichment (top 20)
                                   │   • cross-exchange funding spread
                                   │   • OI-weighted funding
                                   │   • Bitget account+position L/S
                                   │   • OKX rubik taker volume
                                   │   • OKX liquidations + leverage zones
                                   ▼
                       ┌───────────────────────────────────────────┐
                       │  Calibrated XGBoost (grouped split)       │
                       │  • 35 features (26 base + 9 derived)      │
                       │  • scale_pos_weight from class imbalance  │
                       │  • Isotonic calibration cv=3              │
                       │  • Permutation importance                 │
                       └───────────┬───────────────────────────────┘
                                   │ ml_probability (calibrated)
                                   ▼
                       ┌───────────────────────────────────────────┐
                       │  5 LEADING-INDICATOR ENGINES (top 12)     │
                       │  ────────────────────────────────────    │
                       │  1. liq_proximity   — heatmap pull ratio  │
                       │  2. funding_extreme — overheated/squeezed │
                       │  3. ls_extreme      — top vs retail       │
                       │  4. basis           — perp vs spot        │
                       │  5. oi_delta        — OI×price quadrant   │
                       └───────────┬───────────────────────────────┘
                                   │ List[SignalCard] per symbol
                                   ▼
                       ┌───────────────────────────────────────────┐
                       │  CryptoSignalFuser → Tier A/B/C consensus │
                       │  • A: ≥4 engines agree                    │
                       │  • B: 3 engines agree                     │
                       │  • C: 2 engines agree                     │
                       │  • divergent: split → score × 0.7         │
                       └────┬─────────────────────┬────────────────┘
                            │                     │
                            ▼                     ▼
                ┌───────────────────┐  ┌────────────────────────┐
                │  Discord/TG       │  │  Walk-forward backtest │
                │  alerter          │  │  • PnL/Sharpe/MDD      │
                │  (Tier-A only,    │  │  • Threshold sweep     │
                │   30-min dedupe)  │  │  • Equity curve        │
                └───────────────────┘  └────────────────────────┘
```

---

## The 5 engines

| Engine | What it detects | Data source |
|---|---|---|
| `liq_proximity` | Liquidation clusters within ±3% of current price; ratio of short-side vs long-side fuel determines direction | OKX public liquidation orders + cross-exchange OI distributed across leverage tiers |
| `funding_extreme` | Funding > +5bps (longs overheated → fade) or < -3bps (shorts squeezed → squeeze long); cross-exchange spread > 30bps strengthens conviction | KuCoin + Bitget + OKX funding aggregated, OI-weighted |
| `ls_extreme` | Retail crowded long (>2.2 ratio) but top traders flipped short = smart-money fade. Inverse for shorts. | Bitget account-long-short + position-long-short, OKX rubik long-short-account-ratio |
| `basis` | Perp premium > +25bps over spot = leveraged FOMO (fade). Discount > -25bps = institutional spot bid. | KuCoin/Bitget/OKX spot vs perp prices |
| `oi_delta` | Four quadrants: OI↑+Price↑ = healthy long; OI↑+Price↓ = shorts loading; OI↓+Price↑ = short cover (don't chase); OI↓+Price↓ = capitulation (don't catch) | Already in scan rows from `collector.py` |

Each engine emits a `SignalCard{symbol, direction, strength 0-100, reason}`. The fuser combines them with weights `liq_proximity: 0.28, ml_main: 0.22, ls_extreme: 0.16, funding_extreme: 0.14, basis: 0.10, oi_delta: 0.10`.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/gitdhirajs/Azalyst-Crypto-Intelligence.git
cd Azalyst-Crypto-Intelligence

# 2. Install
pip install -r requirements.txt

# 3. (Optional) Set Discord webhook for Tier-A alerts
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/.../..."

# 4. Run a scan
python jobs.py scan-once --show-progress

# 5. Train the model (after a few hours of scans accumulate)
python jobs.py train-main --force

# 6. Backtest the trained model
python jobs.py backtest --threshold 0.62

# 7. Run the multi-engine analysis with Discord alerts
python jobs.py engines --with-alerts

# 8. Test the webhook
python jobs.py test-alert

# 9. Full v2.1 scheduled cycle (matches the GitHub Actions workflow)
python jobs.py scheduled --show-progress
```

---

## Repo structure

```
azalyst-crypto-intelligence/
├── jobs.py                      ← CLI entrypoint
├── scanner.py                   ← Local infinite-loop runner
├── dashboard.py                 ← Local CLI dashboard (data summary + history)
├── requirements.txt
├── README.md                    ← This file
├── UPGRADE_PLAN_V2.md           ← Detailed migration notes from v2.0 → v2.1
│
├── src/
│   ├── config.py                ← Tunables, paths, RSI/labeling params
│   ├── collector.py             ← KuCoin-primary scanner, falls back Bitget→OKX
│   ├── derived_data.py          ← Cross-exchange aggregator (replaces Azalyst)
│   ├── exchange_fallback.py     ← Spot+perp price fetchers for BasisEngine
│   ├── features.py              ← Feature engineering + time-horizon labeling
│   ├── trainer.py               ← Calibrated XGBoost (grouped split, scale_pos_weight)
│   ├── hourly_trainer.py        ← Separate 1h-candle continuation model
│   ├── signal_engines.py        ← The 5 leading-indicator engines
│   ├── signal_fusion.py         ← Tier A/B/C consensus
│   ├── backtester.py            ← Walk-forward PnL/Sharpe/MDD
│   ├── alerter.py               ← Discord + Telegram (with dedupe)
│   └── pipeline.py              ← Scheduled orchestrator
│
├── .github/workflows/
│   ├── main_scanner.yml         ← Every 15min: scan + train + engines + alert + backtest
│   └── hourly_patterns.yml      ← Hourly: 1h candle continuation model
│
├── docs/                        ← GitHub Pages dashboard
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
└── (runtime, ignored on code branch — persisted to runtime-data branch)
    ├── data/raw/                ← scan_YYYYMMDD.csv
    ├── data/features/           ← latest_features.csv
    ├── data/hourly/             ← latest_hourly_market.csv
    ├── models/                  ← latest_model.joblib + feature_columns.joblib + medians
    ├── logs/                    ← train_report_*.json + training_history.jsonl + backtest_history.jsonl + alert_dedupe.json
    └── reports/                 ← latest_scan_signals.{csv,json}, latest_fused_signals.json, latest_backtest.json, latest_runtime_status.json
```

---

## Configuration

All env vars are **optional**. The scanner runs end-to-end with nothing set.

| Variable | Default | Purpose |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | — | Discord webhook for Tier-A alerts |
| `TELEGRAM_BOT_TOKEN` | — | Optional Telegram alerts (also requires chat ID) |
| `TELEGRAM_CHAT_ID` | — | |
| `ALERT_MIN_TIER` | `A` | Minimum tier to alert: `A`, `B`, or `C` |
| `ALERT_DEDUPE_MINUTES` | `30` | Don't repeat the same `(symbol,direction)` within this window |
| `ENGINE_TOP_SYMBOLS` | `12` | How many top-volume symbols get the heavyweight engine pass |
| `FEATURES_DERIVED_CAP` | `20` | How many symbols get cg_* derived-data feature enrichment per cycle |
| `DERIVED_RPS` | `8` | Rate-limit ceiling for the cross-exchange aggregator |
| `MIN_VOLUME_24H_USDT` | `5_000_000` | Minimum 24h USDT volume to include a symbol |
| `MAX_SYMBOLS_PER_SCAN` | `0` (no cap) | Cap how many symbols get scanned each cycle |
| `LABEL_HORIZON_MINUTES` | `60` | Prediction horizon for the main model |
| `PRICE_RISE_THRESHOLD_PCT` | `0.5` | Threshold for label=1 in the snapshot model |
| `MIN_SAMPLES_TO_TRAIN` | `200` | Don't train until this many labelled samples |
| `ALLOW_BLOCKED_EXCHANGES` | `0` | Set to `1` ONLY for local dev — adds Binance/Bybit to fallback chain. On GitHub Actions, leave unset (they 451). |

---

## Models

### Main scanner model (5-min snapshot)

- **Target:** Will price rise ≥0.5% within ~60 minutes of the scan?
- **Features (35):** RSI multi-timeframe (1m/5m/15m/1h/4h/1d) + RSI divergences + OI change/spike/contracts + price/volume rolling stats + funding rate + 9 derived-data columns (`cg_funding_oi_weighted_bps`, `cg_funding_spread_bps`, `cg_top_ls_ratio`, `cg_global_ls_ratio`, `cg_top_minus_global_ls`, `cg_taker_buy_sell_ratio`, `cg_liq_pull_up`, `cg_liq_pull_down`, `cg_liq_pull_ratio`)
- **Algorithm:** XGBoost classifier, depth=5, n_estimators=300, regularized (`reg_alpha=0.1`, `reg_lambda=1.0`) + `scale_pos_weight` auto-computed from class imbalance + isotonic calibration with cv=3
- **Validation:** Primary = grouped split by symbol (no leakage); secondary = time-ordered split (legacy comparison)
- **Output:** Calibrated `predict_proba` saved to `reports/latest_scan_signals.{csv,json}`

### Hourly candle-pattern model (1h continuation)

- **Target:** Will the current 1h candle continue ≥1.5% over the next 3 candles in the same direction?
- **Features:** Candle anatomy (body/wick/range/close-position/direction) + volume expansion ratios + RSI persistence + OI behaviour + price/OI/volume alignment + relative size vs 12h average
- **Output:** Top-25 live signals in `reports/hourly_live_signals.{csv,json}`

---

## Backtester

Walk-forward simulation using the labelled feature dataset. Runs after every training cycle.

| Metric | Definition |
|---|---|
| `n_trades` | Number of signals at `long_threshold ≥ 0.62` |
| `win_rate` | Fraction of trades that closed positive |
| `profit_factor` | Gross winnings ÷ gross losses |
| `expectancy_pct` | Expected per-trade return after stop-loss/take-profit logic |
| `sharpe_annualized` | Annualized using cycle-frequency-adjusted std |
| `max_drawdown_pct` | Peak-to-trough on the equity curve |
| `total_return_pct` | Compound return over the backtest period |
| `by_threshold` | Sensitivity sweep across thresholds 0.55→0.80 |

Trade rules: stop-loss -1.0%, take-profit +2.0%, fixed 1% risk per trade.

> **Realistic expectations:** synthetic-data backtests show high Sharpe (50+). On real Binance data expect Sharpe 1–3 annualized, win rate 52–58%, profit factor 1.2–2.0. Anything above Sharpe 5 in real-data backtests is overfitting.

---

## Discord alerter

Sends rich-embed alerts for Tier-A signals (configurable via `ALERT_MIN_TIER`):

- Color: green for LONG, red for SHORT, orange for divergent
- Fields: each engine's verdict + reason (truncated to 200 chars)
- ML probability + direction
- Timestamp + footer
- Deduplication: same `(symbol, direction, tier)` won't repeat within `ALERT_DEDUPE_MINUTES` (default 30)

Webhook payload limits respected (max 10 embeds per request).

---

## Known limitations (honest)

1. **Liquidation events are OKX-only.** ~10–15% of total derivatives liquidation volume. The rest of the heatmap is synthesized from cross-exchange OI × leverage tiers (10×/25×/50×/100× weights). To close that gap for free, add a websocket listener to KuCoin's free public liquidation stream — ~150 lines of code, but requires a long-running process (won't fit pure cron).
2. **Long/short ratios cap at ~30 days history.** Bitget and OKX both retain ~30 days of public L/S history. For longer windows, log them yourself.
3. **`MIN_SAMPLES_TO_TRAIN=200`** is too low for 35 features. Will overfit on small datasets. Bump to ≥2000 once you've accumulated data, or use the `--force` flag knowing the model is data-starved.
4. **No paper-trading PnL tracker** in this repo (yet). The backtester proves PnL on historical signals; live paper trading is on the roadmap.
5. **No regime conditioning.** BTC dominance and BTC trend aren't features. Performance will vary by market regime.

---

## Roadmap

- [ ] Add KuCoin liquidation websocket listener for the heatmap (closes the 85% real-event gap)
- [ ] Live paper-trading PnL tracker (mirrors the ETF scanner pattern)
- [ ] Regime gates: BTC trend + BTC dominance as global features
- [ ] Multi-class label (up / flat / down) instead of long-only
- [ ] Self-improvement loop (Qwen-driven hyperparameter tuning) — JSON-registry pattern from the ETF scanner
- [ ] SHAP per-prediction explanations in the dashboard

---

## License & disclaimer

This scanner is research software. It is **not financial advice**. The signals it produces are statistical predictions on a 60-minute horizon — they will sometimes be wrong, and they will sometimes be wrong in correlated ways during regime changes. Do not deploy real capital based on its output without your own walk-forward validation, position sizing logic, and a hard maximum drawdown circuit breaker. Past backtest performance does not guarantee future results, especially in crypto where regime shifts can invalidate models in days.
