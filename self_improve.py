"""
self_improve.py — AZALYST CRYPTO AUTONOMOUS IMPROVEMENT ENGINE

Runs via GitHub Actions.
Reads performance data + source code, calls DeepSeek-V4-Pro (fallback to Qwen),
receives a targeted code change, validates it, applies it.
"""

import json
import os
import py_compile
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent

# ── What the AI is ALLOWED to edit ──────────────────────────────────────────
MUTABLE_FILES = {
    "src/trainer.py",
    "src/features.py",
    "src/signal_engines.py",
    "src/signal_fusion.py",
    "src/pipeline.py",
    "src/collector.py",
}

# ── What the AI reads as context (includes read-only files) ─────────────────
SOURCE_FILES = [
    "src/trainer.py",
    "src/features.py",
    "src/signal_engines.py",
    "src/signal_fusion.py",
    "src/pipeline.py",
    "src/collector.py",
    "src/derived_data.py",
    "src/config.py",
]

# ── Performance and History ─────────────────────────────────────────────────
DATA_FILES = [
    "reports/latest_scan_summary.json",
    "logs/latest_train_report.json",
    "reports/latest_backtest.json",
    "improvement_log.jsonl",
]

# ── NVIDIA NIM ───────────────────────────────────────────────────────────────
NIM_URL        = "https://integrate.api.nvidia.com/v1/chat/completions"
PRIMARY_MODEL  = "deepseek-ai/deepseek-v4-pro"
FALLBACK_MODEL = "qwen/qwen3-coder-480b-a35b-instruct"

# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an autonomous improvement agent for Azalyst Crypto, a Python-based
institutional crypto futures scanner and ML signal engine.

Your job:
1. Read the performance data (ML accuracy, backtest win rate, signal tier quality)
2. Read the improvement_log.jsonl to see what has ALREADY been applied — never re-propose these
3. Read the source code
4. Identify the SINGLE highest-impact improvement you can make TODAY
5. Output it as a precise, safe, syntactically valid code change

HARD RULES:
- Output ONLY raw JSON. No markdown, no triple backticks, no prose outside the JSON.
- Propose exactly ONE change per run.
- Check improvement_log.jsonl first. If your proposed change_description closely matches
  any entry where applied=true, pick a DIFFERENT improvement instead.
- old_code must be a VERBATIM exact match of text currently in the file.
- Only edit files in this allowed set: src/trainer.py, src/features.py, src/signal_engines.py,
  src/signal_fusion.py, src/pipeline.py, src/collector.py
- new_code must be syntactically valid Python.
- Keep changes focused and minimal.

OUTPUT FORMAT:
{
  "analysis": "1-2 sentences: what is the problem and how does this fix it",
  "target_metric": "which metric this improves: accuracy / win_rate / signal_quality / risk",
  "confidence": <integer 0-100>,
  "change": {
    "file": "src/filename.py",
    "description": "one line: what this change does",
    "old_code": "exact verbatim code to replace",
    "new_code": "replacement code"
  }
}
"""

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def build_context() -> str:
    parts = []
    parts.append("=" * 60)
    parts.append("PERFORMANCE DATA & IMPROVEMENT HISTORY")
    parts.append("=" * 60)
    for fname in DATA_FILES:
        content = _read(ROOT / fname)
        if content:
            parts.append(f"\n--- {fname} ---")
            parts.append(content)

    parts.append("\n" + "=" * 60)
    parts.append("SOURCE CODE")
    parts.append("=" * 60)
    for fname in SOURCE_FILES:
        content = _read(ROOT / fname)
        if content:
            parts.append(f"\n--- {fname} ---")
            parts.append(content)
    return "\n".join(parts)

def call_nim(context: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    models = [PRIMARY_MODEL, FALLBACK_MODEL]
    last_error = None

    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze and improve Azalyst Crypto:\n\n{context}"},
                ],
                "max_tokens": 4096,
                "temperature": 0.15,
                "top_p": 0.7,
            }
            print(f"  Calling {model} ...")
            resp = requests.post(NIM_URL, headers=headers, json=payload, timeout=600)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:] if lines[-1] != "```" else lines[1:-1])
            return json.loads(raw)
        except Exception as e:
            print(f"  FAILED with {model}: {e}")
            last_error = e
            if model == FALLBACK_MODEL:
                raise last_error
            print("  Retrying with fallback model...")
    raise last_error

def validate_syntax(filepath: Path) -> tuple[bool, str]:
    try:
        py_compile.compile(str(filepath), doraise=True)
        return True, ""
    except py_compile.PyCompileError as exc:
        return False, str(exc)

def apply_change(change: dict) -> bool:
    filename    = (change.get("file") or "").strip()
    old_code    = change.get("old_code", "")
    new_code    = change.get("new_code", "")
    description = change.get("description", "")

    if filename not in MUTABLE_FILES:
        print(f"  BLOCKED: {filename} is not in the mutable file set")
        return False

    filepath = ROOT / filename
    if not filepath.exists():
        print(f"  BLOCKED: {filepath} does not exist")
        return False

    content = filepath.read_text(encoding="utf-8")
    occurrences = content.count(old_code)
    if occurrences != 1:
        print(f"  BLOCKED: old_code found {occurrences} times (must be exactly 1)")
        return False

    new_content = content.replace(old_code, new_code, 1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(new_content)
        tmp_path = Path(tmp.name)

    try:
        ok, err = validate_syntax(tmp_path)
        if not ok:
            print(f"  BLOCKED: syntax error — {err}")
            return False
    finally:
        tmp_path.unlink(missing_ok=True)

    filepath.write_text(new_content, encoding="utf-8")
    print(f"  APPLIED: {description}  →  {filename}")
    return True

def write_log(result: dict, applied: bool):
    log_path = ROOT / "improvement_log.jsonl"
    change = result.get("change") or {}
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis": result.get("analysis", ""),
        "target_metric": result.get("target_metric", ""),
        "confidence": result.get("confidence", 0),
        "change_file": change.get("file"),
        "change_description": change.get("description"),
        "applied": applied,
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

def main() -> int:
    api_key = os.environ.get("NIM_API_KEY", "").strip()
    if not api_key:
        print("ERROR: NIM_API_KEY not set")
        return 1

    print(f"Step 1: Building context...")
    context = build_context()
    
    print(f"Step 2: Calling NIM with fallback...")
    try:
        result = call_nim(context, api_key)
    except Exception as e:
        print(f"API Error: {e}")
        return 1

    change = result.get("change")
    applied = False
    if change:
        print(f"Step 3: Applying change...")
        applied = apply_change(change)
    else:
        print("No change proposed.")

    write_log(result, applied)
    return 0

if __name__ == "__main__":
    sys.exit(main())
