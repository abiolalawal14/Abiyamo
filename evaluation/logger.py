"""
logger.py

Purpose:
    Appends one structured record to data/pilot_logs.jsonl for every
    interaction the chatbot handles. Used during the pilot study to
    track usage patterns, escalation rates, response quality, and
    system performance.

Storage format — JSONL (newline-delimited JSON):
    One JSON object per line. Chosen over a database or CSV because:
    - Appendable with a single file.write() — no locking, no schema
    - Survives process restarts (each line is a complete record)
    - Readable with pandas, jq, or a text editor
    - Easy to stream for large log files

Record schema (one per interaction):
    {
        "timestamp"       : ISO-8601 UTC string
        "user_id"         : SHA-256 of raw phone number, hex-encoded.
                            One-way hash — you can track a user across
                            sessions without storing their real number.
        "language"        : BCP-47 code passed by the caller ("en",
                            "ha", "yo", "ig")
        "message"         : The user's question text. Stored for
                            qualitative review during the pilot.
                            Remove or redact before any public release.
        "answer_preview"  : First 200 chars of the answer — enough to
                            spot obvious failures without storing the
                            full generated text.
        "escalated"       : bool — True if routed to safety escalation
        "chunks_used"     : int — chunks retrieved from knowledge base
        "response_time_ms": float — end-to-end time from message
                            received to answer ready, in milliseconds
        "channel"         : "whatsapp" or "api" — which entry point
                            handled this interaction
    }

Privacy note:
    - user_id is a one-way hash — cannot be reversed to get the phone number
    - message text IS stored as-is for pilot analysis. Inform users in
      the consent statement that anonymised messages are reviewed.
    - Do NOT log to this file in production without a data retention
      and deletion policy in place.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Log file lives in data/ so it stays with the other project data.
# Path is relative to the project root — callers that run from a
# different directory should override LOG_FILE before importing.
LOG_FILE = Path("data/pilot_logs.jsonl")

# Ensure the data directory exists. This is the only place that creates
# it — all other modules assume it already exists from build_knowledge_base.
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _anonymize_user(raw_id: str) -> str:
    """
    Returns a stable, one-way SHA-256 hash of the user identifier
    (phone number, or "api" for direct API calls). The same input
    always produces the same hash, so a user can be tracked across
    sessions in the log without storing their real number.

    Using SHA-256 (not bcrypt) because:
    - We need the same output every time (deterministic) to count
      unique users across log lines
    - Phone numbers are not passwords — the goal is pseudonymisation,
      not password storage. SHA-256 is sufficient and fast.
    """
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]  # 16 hex chars = 64-bit


def _now_utc() -> str:
    """Returns the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def log_interaction(
    user_id: str,
    message: str,
    result: dict,
    response_time_ms: float,
    channel: str = "whatsapp",
) -> None:
    """
    Appends one interaction record to the JSONL log file.

    Parameters:
        user_id          : Raw user identifier — phone number from
                           Twilio (e.g. "whatsapp:+2348012345678") or
                           "api" for HTTP /chat calls. Hashed before
                           writing — the raw value is never stored.
        message          : The user's question text.
        result           : The dict returned by chat_handler.handle_message().
                           Must have keys: answer, language, escalated,
                           chunks_used.
        response_time_ms : End-to-end time in milliseconds, measured
                           by the caller (main.py or webhook.py) by
                           timing the handle_message() call.
        channel          : "whatsapp" or "api". Defaults to "whatsapp"
                           since most interactions come through Twilio.

    Returns nothing. Errors are logged to stdout rather than raised so
    a logging failure never breaks the chatbot's response to the user.
    """
    record = {
        "timestamp": _now_utc(),
        "user_id": _anonymize_user(user_id),
        "language": result.get("language", "en"),
        "message": message,
        "answer_preview": result.get("answer", "")[:200],
        "escalated": result.get("escalated", False),
        "chunks_used": result.get("chunks_used", 0),
        "response_time_ms": round(response_time_ms, 1),
        "channel": channel,
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        # Never raise — a log write failure must not crash the server.
        print(f"[logger] Failed to write log record: {e}")


def read_logs() -> list[dict]:
    """
    Reads and parses all records from the JSONL log file.
    Returns a list of dicts (one per logged interaction).
    Returns an empty list if the log file does not exist yet.

    Used by pilot_report.py to generate summaries.
    """
    if not LOG_FILE.exists():
        return []

    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[logger] Skipping malformed line {line_num}: {e}")

    return records


def count_logs() -> int:
    """
    Returns the number of logged interactions without loading them all
    into memory. Used by the /health endpoint to include log stats.
    """
    if not LOG_FILE.exists():
        return 0
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Quick manual test — runs only when this file is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    # Redirect the log to a temp file so tests never pollute the real
    # pilot_logs.jsonl. We reassign LOG_FILE directly in this module's
    # globals — log_interaction() resolves LOG_FILE at call time from
    # the same globals dict, so it picks up the new path immediately.
    original_log = LOG_FILE
    tmp = tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w", encoding="utf-8"
    )
    tmp.close()
    LOG_FILE = Path(tmp.name)  # redirect all writes to temp file

    print("=== TEST 1: Log a normal interaction ===")
    mock_result = {
        "answer": "Family planning is the practice of controlling the number and spacing of children.",
        "language": "en",
        "escalated": False,
        "chunks_used": 3,
    }
    log_interaction(
        user_id="whatsapp:+2348012345678",
        message="What is family planning?",
        result=mock_result,
        response_time_ms=1240.5,
        channel="whatsapp",
    )
    print(f"Records written: {count_logs()}")

    print("\n=== TEST 2: Log an escalated interaction ===")
    escalated_result = {
        "answer": "This sounds like something a health worker should help with.",
        "language": "en",
        "escalated": True,
        "chunks_used": 0,
    }
    log_interaction(
        user_id="whatsapp:+2347099999999",
        message="I think I might be pregnant and I am scared",
        result=escalated_result,
        response_time_ms=12.0,
        channel="whatsapp",
    )

    print("\n=== TEST 3: Log an API call ===")
    log_interaction(
        user_id="api",
        message="What are the signs of STIs?",
        result={**mock_result, "language": "en"},
        response_time_ms=980.0,
        channel="api",
    )

    print(f"\nTotal records: {count_logs()} (expected: 3)")

    print("\n=== TEST 4: Read logs back ===")
    records = read_logs()
    for r in records:
        print(f"  [{r['channel']}] {r['timestamp'][:19]}  "
              f"lang={r['language']}  escalated={r['escalated']}  "
              f"chunks={r['chunks_used']}  {r['response_time_ms']}ms")

    print("\n=== TEST 5: Anonymisation -- same number -> same hash ===")
    h1 = _anonymize_user("whatsapp:+2348012345678")
    h2 = _anonymize_user("whatsapp:+2348012345678")
    h3 = _anonymize_user("whatsapp:+2347099999999")
    assert h1 == h2, "Same input must give same hash"
    assert h1 != h3, "Different inputs must give different hashes"
    print(f"  +2348012345678 -> {h1}")
    print(f"  +2347099999999 -> {h3}")
    print("PASS")

    # Restore and clean up
    LOG_FILE = original_log
    os.unlink(tmp.name)
    print("\nAll tests passed.")
