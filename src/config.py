"""Configuration for the crypto scanner and training jobs."""

import os
from pathlib import Path
from urllib.parse import quote


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    values = [item.strip().lower() for item in raw.split(",")]
    return [item for item in values if item]


def _build_proxy_url() -> str | None:
    explicit = os.getenv("MARKET_PROXY_URL", "").strip()
    if explicit:
        return explicit

    host = os.getenv("MARKET_PROXY_HOST", "").strip()
    port = os.getenv("MARKET_PROXY_PORT", "").strip()
    if not host or not port:
        return None

    scheme = os.getenv("MARKET_PROXY_SCHEME", "http").strip() or "http"
    username = os.getenv("MARKET_PROXY_USERNAME", "").strip()
    password = os.getenv("MARKET_PROXY_PASSWORD", "").strip()

    auth = ""
    if username:
        auth = quote(username, safe="")
        if password:
            auth += f":{quote(password, safe='')}"
        auth += "@"

    return f"{scheme}://{auth}{host}:{port}"


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
HOURLY_DIR = DATA_DIR / "hourly"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

for directory in [
    RAW_DIR,
    FEATURES_DIR,
    HOURLY_DIR,
    LABELS_DIR,
    MODELS_DIR,
    LOGS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# Symbol discovery and filtering
MIN_VOLUME_24H_USDT = _env_float("MIN_VOLUME_24H_USDT", 5_000_000)
MIN_OI_USDT = _env_float("MIN_OI_USDT", 0)
EXCLUDED_SYMBOLS = {
    "USDCUSDT",
    "BUSDUSDT",
    "TUSDUSDT",
    "FDUSDUSDT",
    "EURUSDT",
    "DAIUSDT",
}
MAX_SYMBOLS_PER_SCAN = _env_int("MAX_SYMBOLS_PER_SCAN", 0)


# Timeframes
TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
KLINE_LIMIT = _env_int("KLINE_LIMIT", 100)


# Scanner settings
SCAN_INTERVAL_SECONDS = _env_int("SCAN_INTERVAL_SECONDS", 300)
OI_CHANGE_WINDOW = _env_int("OI_CHANGE_WINDOW", 6)


# Rate limiting
REQUEST_DELAY = _env_float("REQUEST_DELAY", 0.05)
BATCH_DELAY = _env_float("BATCH_DELAY", 0.5)
BATCH_SIZE = _env_int("BATCH_SIZE", 20)


# RSI settings
RSI_PERIOD = _env_int("RSI_PERIOD", 14)
RSI_OVERBOUGHT = _env_int("RSI_OVERBOUGHT", 70)
RSI_OVERSOLD = _env_int("RSI_OVERSOLD", 30)


# Main scanner labeling
LABEL_HORIZON_MINUTES = _env_int("LABEL_HORIZON_MINUTES", 60)
LABEL_LOOKAHEAD_TOLERANCE_MINUTES = _env_int(
    "LABEL_LOOKAHEAD_TOLERANCE_MINUTES", 45
)
PRICE_RISE_THRESHOLD_PCT = _env_float("PRICE_RISE_THRESHOLD_PCT", 0.5)


# Main model settings
MIN_SAMPLES_TO_TRAIN = _env_int("MIN_SAMPLES_TO_TRAIN", 200)
RETRAIN_EVERY_N_SCANS = _env_int("RETRAIN_EVERY_N_SCANS", 50)
ML_TEST_SIZE = _env_float("ML_TEST_SIZE", 0.2)
ML_RANDOM_STATE = _env_int("ML_RANDOM_STATE", 42)


# Hourly candle-pattern model
HOURLY_SYMBOL_LIMIT = _env_int("HOURLY_SYMBOL_LIMIT", 120)
HOURLY_KLINE_LIMIT = _env_int("HOURLY_KLINE_LIMIT", 168)
HOURLY_FORWARD_CANDLES = _env_int("HOURLY_FORWARD_CANDLES", 3)
HOURLY_CONTINUATION_THRESHOLD_PCT = _env_float(
    "HOURLY_CONTINUATION_THRESHOLD_PCT", 1.5
)
HOURLY_MIN_SAMPLES_TO_TRAIN = _env_int("HOURLY_MIN_SAMPLES_TO_TRAIN", 500)
HOURLY_REPORT_TOP_N = _env_int("HOURLY_REPORT_TOP_N", 25)


# GitHub automation
RUNTIME_DATA_BRANCH = os.getenv("RUNTIME_DATA_BRANCH", "runtime-data")


# Market data providers
SUPPORTED_MARKET_PROVIDERS = ("binance", "bybit")
MARKET_DATA_PROVIDERS = [
    provider
    for provider in _env_csv("MARKET_DATA_PROVIDERS", "binance,bybit")
    if provider in SUPPORTED_MARKET_PROVIDERS
]
if not MARKET_DATA_PROVIDERS:
    MARKET_DATA_PROVIDERS = ["binance", "bybit"]


# Optional outbound proxy
MARKET_PROXY_URL = _build_proxy_url()
MARKET_PROXY_FALLBACK_DIRECT = _env_bool("MARKET_PROXY_FALLBACK_DIRECT", True)
MARKET_PROXY_HOST = os.getenv("MARKET_PROXY_HOST", "").strip()
MARKET_PROXY_PORT = os.getenv("MARKET_PROXY_PORT", "").strip()


# Exchange APIs (USDT perpetual futures)
BINANCE_FAPI_BASE = "https://fapi.binance.com"
BYBIT_BASE = "https://api.bybit.com"
BYBIT_V5_BASE = "https://api.bybit.com/v5"
