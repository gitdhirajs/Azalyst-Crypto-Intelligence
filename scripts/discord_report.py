from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import requests

DEFAULT_WEBHOOK_URL = "https://discord.com/api/webhooks/1497644966929760387/DVOa9Ehih3AVGW44g94-vTw-V3WpPVm5-J1M7mtxUzPk7Vow8Dx2KtM9v4e_u9_4VgY_"
DEFAULT_DASHBOARD_URL = "https://gitdhirajs.github.io/coinglass-scanner/"
DEFAULT_REPO_URL = "https://github.com/gitdhirajs/coinglass-scanner"
DEFAULT_PAYLOAD_PATH = Path("reports/latest_dashboard_payload.json")
DEFAULT_SUMMARY_PATH = Path("reports/latest_summary.md")
MAX_EMBED_DESCRIPTION = 4000


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def fmt_num(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_pct(value: Any, digits: int = 1) -> str:
    return fmt_num(value, digits=digits, suffix="%")


def fmt_prob(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def fmt_big(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.2f}K"
    return f"{number:.2f}"


def clean_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def best_scan_signals(rows: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            abs(float(row.get("ml_probability") or 0.0)),
            abs(float(row.get("price_change_pct_24h") or 0.0)),
            abs(float(row.get("oi_change_pct_1h") or 0.0)),
        ),
        reverse=True,
    )[:limit]


def best_hourly_signals(rows: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: float(row.get("continuation_probability") or 0.0),
        reverse=True,
    )[:limit]


def scan_take(signal: Dict[str, Any]) -> str:
    price_change = float(signal.get("price_change_pct_24h") or 0.0)
    oi_change = float(signal.get("oi_change_pct_1h") or 0.0)
    ml_prob = signal.get("ml_probability")
    if ml_prob is not None:
        try:
            if float(ml_prob) >= 0.65:
                return "model-backed upside watch"
            if float(ml_prob) <= 0.35:
                return "model-backed weak setup"
        except (TypeError, ValueError):
            pass
    if price_change >= 8 and oi_change >= 0:
        return "up hard with open interest still building"
    if price_change <= -8 and oi_change >= 0:
        return "down hard while open interest rises, which can mean crowded positioning"
    if price_change >= 5:
        return "strong upside move, but still needs confirmation"
    if price_change <= -5:
        return "heavy downside move, better treated as caution than chase"
    if abs(oi_change) >= 1:
        return "positioning shifted even though price is not moving cleanly"
    return "quiet or mixed tape right now"


def hourly_take(signal: Dict[str, Any]) -> str:
    probability = float(signal.get("continuation_probability") or 0.0)
    candle_return = float(signal.get("candle_return_pct") or 0.0)
    if probability >= 0.85:
        return "hourly pattern model sees strong continuation odds"
    if probability >= 0.65:
        return "hourly continuation watch"
    if candle_return < 0:
        return "sell candle, but continuation confidence is only moderate"
    return "low continuation confidence"


def build_human_summary(payload: Dict[str, Any]) -> str:
    runtime = payload.get("runtime_status") or {}
    scanner = runtime.get("scanner") or {}
    main_model = runtime.get("main_model") or {}
    hourly_model = runtime.get("hourly_model") or {}
    scan_rows = payload.get("latest_scan_signals") or []
    hourly_rows = payload.get("hourly_live_signals") or []

    scanner_status = scanner.get("status") or "unknown"
    scanned = int(scanner.get("symbols_scanned") or 0)
    attempted = int(scanner.get("symbols_attempted") or 0)
    failures = int(scanner.get("symbols_failed") or 0)

    if scanner_status == "healthy":
        line1 = f"Scanner is healthy: {scanned}/{attempted} symbols were scanned successfully, with {failures} failures."
    elif scanner_status == "blocked":
        line1 = "Scanner is blocked right now, so this run did not collect usable live data."
    elif scanner_status == "degraded":
        line1 = "Scanner ran in degraded mode, so treat this update as partial rather than fully live."
    else:
        line1 = f"Scanner status is {scanner_status}, so this run needs a little caution."

    movers = sorted(
        scan_rows,
        key=lambda row: abs(float(row.get("price_change_pct_24h") or 0.0)),
        reverse=True,
    )[:2]
    if movers:
        mover_bits = [
            f"{row.get('symbol', '?')} {fmt_pct(row.get('price_change_pct_24h'), 1)}"
            for row in movers
        ]
        line2 = "Biggest 24h movers in this batch: " + ", ".join(mover_bits) + "."
    else:
        line2 = "No current scan rows were available for a mover summary."

    if main_model.get("status") == "trained":
        line3 = (
            f"Main model is trained with accuracy {fmt_prob(main_model.get('accuracy'))} "
            f"and ROC AUC {fmt_num(main_model.get('roc_auc'), 3)}."
        )
    elif main_model.get("status") == "no_labeled_data":
        line3 = "Main scanner model is still warming up because there is not enough labeled data yet."
    else:
        line3 = f"Main scanner model status: {main_model.get('status', 'missing')}."

    if hourly_model.get("status") == "trained":
        caution = ""
        if float(hourly_model.get("samples") or 0.0) < 500:
            caution = " Treat it as promising research, not automatic truth."
        line4 = (
            f"Hourly pattern model is trained at {fmt_prob(hourly_model.get('accuracy'))} accuracy "
            f"on {int(hourly_model.get('samples') or 0)} samples." + caution
        )
    else:
        line4 = f"Hourly pattern model status: {hourly_model.get('status', 'missing')}."

    if hourly_rows:
        leader = best_hourly_signals(hourly_rows, limit=1)[0]
        line5 = (
            f"Top hourly continuation watch right now is {leader.get('symbol', '?')} "
            f"at {fmt_prob(leader.get('continuation_probability'))}."
        )
    else:
        line5 = "No hourly continuation rows were available yet."

    notes = runtime.get("notes") or []
    if notes:
        line6 = "Process note: " + str(notes[0])
    else:
        line6 = "Process note: runtime payload was generated successfully."

    return "\n".join([line1, line2, line3, line4, line5, line6])


def build_scan_embed(rows: List[Dict[str, Any]]) -> str:
    top_rows = best_scan_signals(rows)
    if not top_rows:
        return "No live scan rows were available."
    blocks = []
    for row in top_rows:
        blocks.append(
            "\n".join(
                [
                    f"**{row.get('symbol', '?')}**",
                    f"Plain-English: {scan_take(row)}.",
                    (
                        f"Tech: price {fmt_num(row.get('price'), 4)} | 24h move {fmt_pct(row.get('price_change_pct_24h'), 2)} | "
                        f"OI 1h {fmt_pct(row.get('oi_change_pct_1h'), 2)} | funding {fmt_num(row.get('funding_rate'), 6)} | "
                        f"RSI 1h {fmt_num(row.get('rsi_1h'), 1)} | volume 24h {fmt_big(row.get('volume_24h'))}"
                    ),
                    f"ML probability: {fmt_prob(row.get('ml_probability'))}",
                ]
            )
        )
    return truncate("\n\n".join(blocks), MAX_EMBED_DESCRIPTION)


def build_hourly_embed(rows: List[Dict[str, Any]]) -> str:
    top_rows = best_hourly_signals(rows)
    if not top_rows:
        return "No hourly continuation rows were available."
    blocks = []
    for row in top_rows:
        blocks.append(
            "\n".join(
                [
                    f"**{row.get('symbol', '?')}**",
                    f"Plain-English: {hourly_take(row)}.",
                    (
                        f"Tech: continuation {fmt_prob(row.get('continuation_probability'), 2)} | "
                        f"prediction {row.get('continuation_prediction', 'n/a')} | candle return {fmt_pct(row.get('candle_return_pct'), 2)} | "
                        f"body {fmt_pct(row.get('body_pct'), 2)} | volume ratio 6h {fmt_num(row.get('volume_ratio_6h'), 2)} | "
                        f"RSI 1h {fmt_num(row.get('rsi_1h'), 1)}"
                    ),
                ]
            )
        )
    return truncate("\n\n".join(blocks), MAX_EMBED_DESCRIPTION)


def build_process_embed(payload: Dict[str, Any]) -> str:
    runtime = payload.get("runtime_status") or {}
    scanner = runtime.get("scanner") or {}
    main_report = payload.get("main_report") or {}
    hourly_report = payload.get("hourly_report") or {}
    lines = [
        f"Generated: {payload.get('generated_at') or 'n/a'}",
        f"Scanner status: {scanner.get('status', 'n/a')}",
        f"Symbols attempted/scanned/failed: {scanner.get('symbols_attempted', 'n/a')} / {scanner.get('symbols_scanned', 'n/a')} / {scanner.get('symbols_failed', 'n/a')}",
        f"Min 24h volume filter: {fmt_big(scanner.get('min_volume_24h_usdt'))}",
        f"Request errors: {((scanner.get('request_errors') or {}).get('count')) or 0}",
        (
            f"Main model: {main_report.get('status', 'missing')} | acc {fmt_prob(main_report.get('accuracy'))} | "
            f"baseline {fmt_prob(main_report.get('baseline_accuracy'))} | auc {fmt_num(main_report.get('roc_auc'), 3)} | "
            f"samples {main_report.get('n_samples', 'n/a')}"
        ),
        (
            f"Hourly model: {hourly_report.get('status', 'missing')} | acc {fmt_prob(hourly_report.get('accuracy'))} | "
            f"baseline {fmt_prob(hourly_report.get('baseline_accuracy'))} | auc {fmt_num(hourly_report.get('roc_auc'), 3)} | "
            f"samples {hourly_report.get('n_samples', 'n/a')}"
        ),
        (
            f"Schedules: main {((runtime.get('workflow_schedules') or {}).get('main_scanner')) or 'n/a'} | "
            f"hourly {((runtime.get('workflow_schedules') or {}).get('hourly_patterns')) or 'n/a'}"
        ),
    ]
    notes = runtime.get("notes") or []
    if notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in notes[:3])
    return truncate("\n".join(lines), MAX_EMBED_DESCRIPTION)


def build_summary_embed(summary_markdown: str) -> str:
    cleaned = clean_markdown(summary_markdown)
    if not cleaned:
        return "No markdown summary was available."
    return truncate(cleaned, MAX_EMBED_DESCRIPTION)


def build_payload(
    runtime_payload: Dict[str, Any],
    summary_markdown: str,
    dashboard_url: str,
    repo_url: str,
    run_url: str,
    update_kind: str,
) -> Dict[str, Any]:
    kind_label = update_kind.strip() if update_kind.strip() else "runtime"
    links = [f"[Dashboard]({dashboard_url})", f"[Repo]({repo_url})"]
    if run_url:
        links.append(f"[Workflow Run]({run_url})")
    content = f"New coinglass-scanner {kind_label} update. " + " | ".join(links)

    return {
        "username": "Coinglass Scanner",
        "allowed_mentions": {"parse": []},
        "content": content,
        "embeds": [
            {
                "title": f"Coinglass Scanner - {kind_label.title()} Summary",
                "color": 0xF97316,
                "description": build_human_summary(runtime_payload),
            },
            {
                "title": "Current Scan Signals",
                "color": 0x2563EB,
                "description": build_scan_embed(runtime_payload.get("latest_scan_signals") or []),
            },
            {
                "title": "Hourly Continuation Signals",
                "color": 0x16A34A,
                "description": build_hourly_embed(runtime_payload.get("hourly_live_signals") or []),
            },
            {
                "title": "Technical + Process Details",
                "color": 0x64748B,
                "description": build_process_embed(runtime_payload),
            },
            {
                "title": "Scanner Summary Report",
                "color": 0x7C3AED,
                "description": build_summary_embed(summary_markdown),
            },
        ],
    }


def send_payload(webhook_url: str, payload: Dict[str, Any]) -> None:
    response = requests.post(webhook_url, json=payload, timeout=30)
    response.raise_for_status()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post coinglass-scanner runtime updates to Discord.")
    parser.add_argument("--webhook-url", default=DEFAULT_WEBHOOK_URL)
    parser.add_argument("--dashboard-url", default=DEFAULT_DASHBOARD_URL)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--update-kind", default="runtime")
    parser.add_argument("--payload-path", default=str(DEFAULT_PAYLOAD_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--payload-out", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_payload = load_json(Path(args.payload_path))
    summary_markdown = load_text(Path(args.summary_path))
    payload = build_payload(
        runtime_payload=runtime_payload,
        summary_markdown=summary_markdown,
        dashboard_url=args.dashboard_url,
        repo_url=args.repo_url,
        run_url=args.run_url,
        update_kind=args.update_kind,
    )

    if args.payload_out:
        Path(args.payload_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    send_payload(args.webhook_url, payload)
    print("Discord update sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
