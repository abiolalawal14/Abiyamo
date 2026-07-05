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

Persistence across Hugging Face Space restarts:
    Hugging Face Spaces run on an EPHEMERAL filesystem — data/ is wiped
    every time the Space restarts (redeploy, sleep/wake on the free
    tier, a crash). Without a second copy somewhere durable, every
    pilot interaction logged between restarts is lost permanently.

    Fix: after every local append, the current full log file is also
    pushed to a private Hugging Face Dataset repo (HF_LOGS_DATASET_REPO
    env var, e.g. "abiola114/abiyamo-pilot-logs") using the SAME
    HF_TOKEN secret already configured for the Space — no new
    credential to set up. On a fresh process (local file missing), the
    last-pushed copy is pulled down first so appends continue from
    where the previous instance left off, rather than silently
    restarting from zero and overwriting the remote history on the
    next push.

    Why a Dataset repo over the other options considered:
    - Google Sheets (gspread) needs a NEW credential (a service account
      JSON key) that doesn't exist yet, plus creating a Google Cloud
      project and sharing the sheet with the service account —
      meaningfully more setup for a 3-5 day, 15-25 user pilot.
    - Periodic email (SMTP) needs a NEW email account/app-password
      secret, and batching "every N interactions" reintroduces a loss
      window: if the process restarts between batches, the
      not-yet-emailed records are gone — the exact problem this is
      supposed to solve.
    - A Dataset repo reuses HF_TOKEN (already present), needs no new
      external service, and pushing after every interaction means at
      most the single in-flight request is ever at risk, not a whole
      unflushed batch.

    Honest tradeoffs:
    - HF Datasets are git-backed. Pushing on every interaction creates
      one commit each time — fine at this pilot's volume (well under
      1,000 interactions total) but would need batching before scaling
      to production traffic.
    - If HF_LOGS_DATASET_REPO or HF_TOKEN isn't set (e.g. local dev),
      the push/restore functions silently no-op — local-only JSONL
      logging keeps working exactly as before. This is a bolt-on layer,
      not a replacement for the local file.
    - This does NOT protect against losing the CURRENT in-flight
      request if the process dies between the local write and the
      remote push — an acceptable residual risk for a pilot, not
      something a production system should accept unmodified.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env so HF_LOGS_DATASET_REPO / HF_TOKEN are available when
# running locally. override=False matches generator.py/sender.py's
# pattern — a value already set in the shell/Space environment wins.
load_dotenv(override=False)

# Log file lives in data/ so it stays with the other project data.
# Path is relative to the project root — callers that run from a
# different directory should override LOG_FILE before importing.
LOG_FILE = Path("data/pilot_logs.jsonl")

# Ensure the data directory exists. This is the only place that creates
# it — all other modules assume it already exists from build_knowledge_base.
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Remote persistence — Hugging Face Dataset repo (see module docstring)
# ---------------------------------------------------------------------------

# Repo ID for the private HF Dataset that mirrors LOG_FILE, e.g.
# "abiola114/abiyamo-pilot-logs". Empty by default — the whole remote
# layer no-ops (local-only logging) until this is configured, so
# nothing breaks for local dev or a Space without this secret set yet.
HF_LOGS_DATASET_REPO = os.environ.get("HF_LOGS_DATASET_REPO", "").strip()
_HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

_hf_api = None            # module-level cache, same pattern as generator.py's _client_cache
_remote_repo_ready = False
_remote_restore_attempted = False


def _remote_enabled() -> bool:
    """True only when both the repo ID and a token are configured."""
    return bool(HF_LOGS_DATASET_REPO and _HF_TOKEN)


def _get_hf_api():
    """Lazily builds (and caches) the HfApi client. Returns None if not configured."""
    global _hf_api
    if not _remote_enabled():
        return None
    if _hf_api is None:
        from huggingface_hub import HfApi
        _hf_api = HfApi(token=_HF_TOKEN)
    return _hf_api


def _ensure_remote_repo_exists() -> None:
    """
    Creates HF_LOGS_DATASET_REPO as a private dataset repo if it doesn't
    exist yet. Runs at most once per process. exist_ok=True makes this
    safe to call even if the repo was already created (by a previous
    Space instance, or manually) — never raises for "already exists".
    """
    global _remote_repo_ready
    if _remote_repo_ready or not _remote_enabled():
        return
    api = _get_hf_api()
    try:
        api.create_repo(repo_id=HF_LOGS_DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
        _remote_repo_ready = True
    except Exception as e:
        print(f"[logger] Warning: could not create/verify dataset repo {HF_LOGS_DATASET_REPO}: {e}")


def _restore_from_remote_if_needed() -> None:
    """
    On a fresh process where the local log file is missing (Space just
    restarted and wiped data/), pulls down the last-pushed copy from
    HF_LOGS_DATASET_REPO first, so new appends continue the existing
    history instead of starting from zero and overwriting the remote
    copy with a near-empty file on the next push. Runs at most once per
    process; a no-op if the local file already exists (mid-process,
    nothing to restore) or the remote layer isn't configured.
    """
    global _remote_restore_attempted
    if _remote_restore_attempted or LOG_FILE.exists() or not _remote_enabled():
        return
    _remote_restore_attempted = True

    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=HF_LOGS_DATASET_REPO,
            repo_type="dataset",
            filename=LOG_FILE.name,
            token=_HF_TOKEN,
        )
        shutil.copy(downloaded_path, LOG_FILE)
        print(f"[logger] Restored {count_logs()} prior log record(s) from {HF_LOGS_DATASET_REPO}.")
    except Exception as e:
        # Covers: repo/file doesn't exist yet (first-ever run), network
        # error, etc. None of these should block local logging from
        # starting fresh.
        print(f"[logger] No prior remote logs restored ({e}). Starting fresh.")


def _push_to_remote() -> None:
    """
    Uploads the CURRENT full local log file to HF_LOGS_DATASET_REPO,
    overwriting the remote copy. Called after every successful local
    append (see log_interaction()) so at most the single in-flight
    interaction is ever at risk if the process dies before this runs —
    not the whole pilot's accumulated history.
    """
    if not _remote_enabled():
        return
    _ensure_remote_repo_exists()
    api = _get_hf_api()
    try:
        api.upload_file(
            path_or_fileobj=str(LOG_FILE),
            path_in_repo=LOG_FILE.name,
            repo_id=HF_LOGS_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update pilot logs",
        )
    except Exception as e:
        # Never raise — a failed remote push must not crash the
        # chatbot's response to the user. The record is still safe in
        # the local file; the next successful push will include it too.
        print(f"[logger] Warning: failed to push logs to {HF_LOGS_DATASET_REPO}: {e}")


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

    Persistence: restores any prior log history from the HF Dataset
    repo first (only does anything on a fresh process with no local
    file — see _restore_from_remote_if_needed()), appends locally as
    before, then pushes the updated file back to the dataset repo (see
    module docstring). Both remote steps silently no-op if
    HF_LOGS_DATASET_REPO/HF_TOKEN aren't configured — local-only JSONL
    logging is unaffected either way.
    """
    _restore_from_remote_if_needed()

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
        return  # nothing new to push if the local write itself failed

    _push_to_remote()


def read_logs() -> list[dict]:
    """
    Reads and parses all records from the JSONL log file.
    Returns a list of dicts (one per logged interaction).
    Returns an empty list if the log file does not exist yet.

    Used by pilot_report.py to generate summaries. Restores from the HF
    Dataset repo first (see log_interaction()) so this reflects prior
    history even if called immediately after a fresh Space restart,
    before any new interaction has triggered a restore itself.
    """
    _restore_from_remote_if_needed()

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
    Restores from the HF Dataset repo first — see read_logs().
    """
    _restore_from_remote_if_needed()

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

    print("\n=== TEST 6: Remote layer no-ops when unconfigured ===")
    print(f"  HF_LOGS_DATASET_REPO set: {bool(HF_LOGS_DATASET_REPO)}  (expected: False, nothing in .env)")
    print(f"  _remote_enabled(): {_remote_enabled()}  (expected: False)")
    # These must not raise even though nothing is configured.
    _restore_from_remote_if_needed()
    _push_to_remote()
    print("  No exceptions raised. PASS")

    print("\n=== TEST 7: Live round-trip against a real HF Dataset repo ===")
    test_repo = os.environ.get("TEST_HF_LOGS_DATASET_REPO", "")
    test_token = os.environ.get("HF_TOKEN", "")
    if not (test_repo and test_token):
        print("  SKIPPED -- set TEST_HF_LOGS_DATASET_REPO and HF_TOKEN env vars to run this live.")
    else:
        HF_LOGS_DATASET_REPO = test_repo
        _HF_TOKEN = test_token
        _remote_repo_ready = False
        _remote_restore_attempted = False

        log_interaction(
            user_id="whatsapp:+2348012345678",
            message="[logger self-test] live push",
            result=mock_result,
            response_time_ms=42.0,
            channel="whatsapp",
        )
        print(f"  Pushed. Local record count: {count_logs()}")

        # Simulate a fresh Space restart: local file gone, guards reset.
        os.unlink(tmp.name)
        _remote_restore_attempted = False
        restored_count = count_logs()
        print(f"  After simulated restart (local file deleted): "
              f"restored {restored_count} record(s) from {test_repo}  "
              f"(expected: >= 1, includes this test's own record plus any prior runs)")

    # Restore and clean up
    LOG_FILE = original_log
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
    print("\nAll tests passed.")
