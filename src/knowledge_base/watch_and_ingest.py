"""
watch_and_ingest.py

Purpose:
    Periodically scans data/raw_pdfs/ for new or updated PDF files and
    automatically processes them into the knowledge base — without
    requiring someone to manually run build_knowledge_base.py every time
    a new document is added.

Design notes:
    - Uses a SCHEDULED CHECK approach (runs every N minutes), not
      real-time folder watching — this was a deliberate choice for
      reliability and simplicity, discussed and agreed on earlier.
    - Relies on ingestion_log.json as memory: a record of which files
      have already been processed, and when, so this script never
      re-processes an unchanged file, and can detect when a file has
      been replaced with a newer version.
    - Reuses add_document() from build_knowledge_base.py and
      document_exists() from manage_documents.py directly — this file
      does not duplicate any pipeline logic, it only adds the
      "what's new, what changed" decision layer on top.

Requirements:
    No new packages — uses only what pdf_loader.py, text_splitter.py,
    embedder.py, build_knowledge_base.py, and manage_documents.py
    already require, plus Python's built-in json and time modules.
"""

import os
import json
import time
from datetime import datetime

from build_knowledge_base import add_document, remove_document
from manage_documents import document_exists

RAW_PDFS_FOLDER = "data/raw_pdfs"
INGESTION_LOG_PATH = "data/ingestion_log.json"

# How often to check the folder, in minutes. Adjust as needed — this was
# left as a deliberately simple, editable constant since the right
# interval depends on how often new documents are actually expected
# (e.g. hourly is reasonable for occasional updates; daily may be
# enough if updates are rare).
CHECK_INTERVAL_MINUTES = 60


# ---------------------------------------------------------------------------
# Step 1: Load and save the ingestion log (the script's memory)
# ---------------------------------------------------------------------------

def load_ingestion_log() -> dict:
    """
    Loads the record of previously processed files. Returns an empty
    dict if the log doesn't exist yet (e.g. first time this script runs).

    Log format:
        {
            "filename.pdf": {
                "last_modified": 1718600000.0,   (file's modification time)
                "processed_at": "2026-06-17T10:30:00",
                "chunk_count": 137
            }
        }
    """
    if not os.path.exists(INGESTION_LOG_PATH):
        return {}

    try:
        with open(INGESTION_LOG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[watch_and_ingest] Could not read ingestion log, starting fresh: {e}")
        return {}


def save_ingestion_log(log: dict):
    """
    Writes the ingestion log back to disk. Called after every successful
    add or update, so progress is never lost even if the script is
    stopped between files.
    """
    os.makedirs(os.path.dirname(INGESTION_LOG_PATH), exist_ok=True)
    with open(INGESTION_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# Step 2: Figure out what's new or changed since the last check
# ---------------------------------------------------------------------------

def get_pdf_files(folder_path: str) -> list:
    """Returns a list of all .pdf filenames currently in the folder."""
    if not os.path.isdir(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]


def needs_processing(filename: str, full_path: str, log: dict) -> str:
    """
    Decides what action (if any) a given file needs, by comparing it
    against the ingestion log.

    Returns one of:
        "new"        - file has never been processed before
        "updated"    - file was processed before, but has since changed
                        (modification time is newer than what's logged)
        "unchanged"  - file was already processed and hasn't changed
    """
    current_modified_time = os.path.getmtime(full_path)

    if filename not in log:
        return "new"

    logged_modified_time = log[filename].get("last_modified", 0)
    if current_modified_time > logged_modified_time:
        return "updated"

    return "unchanged"


# ---------------------------------------------------------------------------
# Step 3: Run a single check-and-ingest pass
# ---------------------------------------------------------------------------

def run_ingestion_pass():
    """
    Performs one full check: scans the folder, determines what's new or
    updated, processes those files, and updates the ingestion log.

    This is the function that gets called on each scheduled interval —
    it does one pass and returns, rather than looping itself, so the
    looping/scheduling logic stays separate and easy to follow.
    """
    print(f"\n[watch_and_ingest] Running ingestion check at {datetime.now().isoformat()}")

    log = load_ingestion_log()
    pdf_files = get_pdf_files(RAW_PDFS_FOLDER)

    if not pdf_files:
        print(f"[watch_and_ingest] No PDF files found in {RAW_PDFS_FOLDER}")
        return

    new_count = 0
    updated_count = 0
    unchanged_count = 0
    failed_count = 0

    for filename in pdf_files:
        full_path = os.path.join(RAW_PDFS_FOLDER, filename)
        status = needs_processing(filename, full_path, log)

        if status == "unchanged":
            unchanged_count += 1
            continue

        if status == "updated":
            print(f"[watch_and_ingest] '{filename}' has changed — removing old version first.")
            remove_document(filename)

        print(f"[watch_and_ingest] Processing '{filename}' ({status})...")
        success = add_document(full_path, replace_if_exists=False)
        # replace_if_exists=False here is correct because we already
        # called remove_document() above for the "updated" case — we
        # don't want add_document() to try removing it a second time.

        if success:
            log[filename] = {
                "last_modified": os.path.getmtime(full_path),
                "processed_at": datetime.now().isoformat(),
            }
            save_ingestion_log(log)  # save after every file, not just at the end

            if status == "new":
                new_count += 1
            else:
                updated_count += 1
        else:
            failed_count += 1
            print(f"[watch_and_ingest] Failed to process '{filename}' — will retry on next check.")

    print(f"\n[watch_and_ingest] Pass complete. "
          f"New: {new_count}, Updated: {updated_count}, "
          f"Unchanged: {unchanged_count}, Failed: {failed_count}")


# ---------------------------------------------------------------------------
# Step 4: The scheduled loop — runs forever, checking at fixed intervals
# ---------------------------------------------------------------------------

def start_watching():
    """
    Runs run_ingestion_pass() immediately, then repeats every
    CHECK_INTERVAL_MINUTES, indefinitely, until manually stopped.

    This is meant to run as a long-lived background process (e.g. via
    a scheduled task on your deployment platform, as discussed earlier
    for Render's cron job feature) — not something you run once and
    expect to exit.
    """
    print(f"[watch_and_ingest] Starting scheduled ingestion watcher. "
          f"Checking every {CHECK_INTERVAL_MINUTES} minute(s).")

    while True:
        run_ingestion_pass()
        print(f"[watch_and_ingest] Sleeping for {CHECK_INTERVAL_MINUTES} minute(s)...")
        time.sleep(CHECK_INTERVAL_MINUTES * 60)


# ---------------------------------------------------------------------------
# Run this file directly to start watching
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # For manual testing, run a single pass instead of the full loop,
    # so you can verify it works without waiting an hour to see results.
    # Switch to start_watching() once you're ready for continuous,
    # scheduled operation.
    run_ingestion_pass()

    # Uncomment this line when ready for continuous scheduled watching:
    # start_watching()