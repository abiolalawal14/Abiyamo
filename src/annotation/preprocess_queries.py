"""
preprocess_queries.py

Purpose:
    Clean the Google Form CSV export, extract individual SRH queries
    from the query_text column, and produce three output files:
      1. label_studio_import.csv   — for Label Studio Cloud annotation
      2. queries_with_metadata.csv — query + participant demographics
      3. participant_summary.csv   — one row per participant, all
                                     non-query columns, for Chapter 4
                                     demographic/acceptability analysis

    Safe to rerun as new form responses arrive — appends only genuinely
    new data; never overwrites or duplicates existing rows.

Where this sits in the annotation pipeline:
    [Google Form export] -> [THIS FILE] -> label_studio_import.csv
                                        -> queries_with_metadata.csv
                                        -> participant_summary.csv
    label_studio_import.csv and queries_with_metadata.csv feed into
    Label Studio Cloud. participant_summary.csv is Chapter 4 only.
    After annotation, merge_annotations.py takes over.

Design notes:
    - Rerun deduplication uses Timestamp (unique per form submission)
      to detect already-processed participants, and case-insensitive
      query text to detect already-processed queries.
    - participant_id is positional within the full form export (not
      within a single run), so P001 always refers to the same person
      across reruns.
    - Category estimation (crisis/diagnostic/educational) is printed
      in the terminal summary ONLY — never written to any output file.
"""

import os
import re
import pandas as pd
from collections import Counter

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

INPUT_FILE = os.path.join("data", "raw_queries", "form_responses.csv")

OUTPUT_DIR          = os.path.join("data", "raw_queries")
OUTPUT_LABEL_STUDIO = os.path.join(OUTPUT_DIR, "label_studio_import.csv")
OUTPUT_WITH_METADATA = os.path.join(OUTPUT_DIR, "queries_with_metadata.csv")
OUTPUT_PARTICIPANT_SUMMARY = os.path.join(OUTPUT_DIR, "participant_summary.csv")

TIMESTAMP_COL  = "Timestamp"
QUERY_COL      = "query_text"
AGE_COL        = "Age Range "
GENDER_COL     = "Gender "
RELIGION_COL   = "Religion"
LANGUAGE_COL   = "What is your preferred Language "
INFO_SOURCES_COL      = "Where do you usually get information about sexual and reproductive health?\n(Select all that apply) "
UNCOMFORTABLE_COL     = "Have you ever felt uncomfortable asking someone a sexual or reproductive health question? "
UNCOMFORTABLE_WHY_COL = "If yes, why?"
CHATBOT_HELPFUL_COL   = "Do you think a private chatbot could help young people access SRH information safely? "
TRUST_FACTORS_COL     = "What kinds of responses would make you trust a health chatbot?"
HELPLINE_COL          = "If a chatbot detects that a user may be in danger or emotional distress, should it recommend contacting a helpline or healthcare provider? "
TOPICS_NEEDED_COL     = "What sexual and reproductive health topics do you need information about?"
TOPICS_WANTED_COL     = "What SRH topics do you personally want more information about?"

FILLER_ANSWERS = {
    "n/a", "none", "nil", "i don't know", "i do not know",
    "nothing", "no question", "no comment",
}

QUERY_ID_PREFIX       = "SRH"
PARTICIPANT_ID_PREFIX = "P"

# Topics covered by the two knowledge base documents.
# Used to flag gaps between what participants want and what the KB contains.
KB_TOPICS = {
    "family planning", "contraception", "condom", "pill", "injectable",
    "implant", "iud", "safe period", "calendar method",
    "sexually transmitted", "sti", "hiv", "aids", "gonorrhoea",
    "chlamydia", "syphilis", "herpes",
    "menstruation", "menstrual", "period",
    "pregnancy", "pregnant", "antenatal",
    "puberty", "adolescent", "reproductive health",
    "gender", "consent", "abuse", "sexual violence",
    "cervical cancer", "breast", "nutrition",
}

# Percentage threshold above which a missing-from-KB topic triggers a warning.
KB_GAP_THRESHOLD = 0.20

CRISIS_KEYWORDS = [
    "abuse", "rape", "pregnant", "bleeding", "pain", "scared",
    "help me", "suicid", "unsafe", "assault", "forced",
    "emergency", "danger", "distress",
]

DIAGNOSTIC_KEYWORDS = [
    "symptoms", "discharge", "burning", "itching", "missed period",
    "infection", "tested", "doctor", "hurts", "irregular",
    "abnormal", "sore", "rash", "fever",
]


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

_NUMBERED_SPLIT_RE = re.compile(r'(?<!\d)(?=\d+[\.\)]\s)')


def _split_cell(cell: str) -> list[str]:
    text = cell.strip()
    if not text:
        return []

    parts = [p.strip() for p in _NUMBERED_SPLIT_RE.split(text) if p.strip()]
    if len(parts) > 1:
        cleaned = [re.sub(r'^\d+[\.\)]\s*', '', p).strip() for p in parts]
        return [c for c in cleaned if c]

    parts = [p.strip() for p in re.split(r'\r?\n', text) if p.strip()]
    if len(parts) > 1:
        return parts

    parts = text.split("?")
    reassembled = []
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        if i < len(parts) - 1:
            p = p + "?"
        reassembled.append(p)
    if len(reassembled) > 1:
        return reassembled

    if len(text) > 100:
        parts = [p.strip() for p in text.split(". ") if p.strip()]
        if len(parts) > 1:
            parts = [p + ("." if not p.endswith(".") else "") for p in parts[:-1]] + [parts[-1]]
            return parts

    return [text]


def _should_drop(fragment: str, seen_lower: set) -> tuple[bool, str]:
    stripped = fragment.strip()
    if not stripped:
        return True, "empty"
    if stripped.lower() in FILLER_ANSWERS:
        return True, "filler"
    if stripped.lower() in seen_lower:
        return True, "duplicate"
    return False, ""


def _estimate_category(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in CRISIS_KEYWORDS):
        return "crisis"
    if any(kw in lower for kw in DIAGNOSTIC_KEYWORDS):
        return "diagnostic"
    return "educational"


# ---------------------------------------------------------------------------
# Load existing output state (for rerun deduplication)
# ---------------------------------------------------------------------------

def _load_existing_state() -> tuple[set[str], set[str], int, set[str]]:
    """
    Returns:
        existing_timestamps  : Timestamps already in participant_summary.csv
        existing_query_texts : Lowercased query texts already in label_studio_import.csv
        max_query_number     : Highest SRH### number already assigned
        existing_pids        : participant_ids already in participant_summary.csv
    """
    existing_timestamps:  set[str] = set()
    existing_query_texts: set[str] = set()
    existing_pids:        set[str] = set()
    max_query_number = 0

    if os.path.exists(OUTPUT_PARTICIPANT_SUMMARY):
        try:
            ps = pd.read_csv(OUTPUT_PARTICIPANT_SUMMARY, dtype=str)
            if TIMESTAMP_COL in ps.columns:
                existing_timestamps = set(ps[TIMESTAMP_COL].dropna().str.strip())
            if "participant_id" in ps.columns:
                existing_pids = set(ps["participant_id"].dropna().str.strip())
        except Exception:
            pass

    if os.path.exists(OUTPUT_LABEL_STUDIO):
        try:
            ls = pd.read_csv(OUTPUT_LABEL_STUDIO, dtype=str)
            if "text" in ls.columns:
                existing_query_texts = set(ls["text"].dropna().str.strip().str.lower())
            if "id" in ls.columns:
                for qid in ls["id"].dropna():
                    m = re.match(r'[A-Z]+(\d+)$', qid.strip())
                    if m:
                        max_query_number = max(max_query_number, int(m.group(1)))
        except Exception:
            pass

    return existing_timestamps, existing_query_texts, max_query_number, existing_pids


# ---------------------------------------------------------------------------
# Demographic summary helpers (terminal printout only)
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "0%"


def _distribution(series: pd.Series, label: str, total: int):
    counts = series.value_counts(dropna=False)
    print(f"\n  {label}:")
    for val, cnt in counts.items():
        display = val if pd.notna(val) and str(val).strip() else "(no response)"
        print(f"    {display}: {cnt} ({_pct(cnt, total)})")


def _top_keywords(series: pd.Series, n: int, label: str):
    """Tokenise free-text column into words and print top-n by frequency."""
    counter: Counter = Counter()
    for cell in series.dropna():
        words = re.findall(r'\b[a-z]{4,}\b', str(cell).lower())
        counter.update(words)
    # Remove stop words that carry no semantic meaning in this context.
    stop = {"that", "this", "with", "from", "have", "will", "they",
            "them", "their", "your", "more", "also", "about", "would",
            "should", "could", "when", "what", "which", "were", "been",
            "than", "then", "some", "such", "each", "into", "just"}
    for w in stop:
        counter.pop(w, None)
    print(f"\n  {label} (top {n} keywords):")
    for word, cnt in counter.most_common(n):
        print(f"    {word}: {cnt}")


def _kb_gap_check(series: pd.Series, total_participants: int):
    """
    Flags free-text topic mentions that exceed KB_GAP_THRESHOLD and do
    not appear in the KB_TOPICS set.
    """
    counter: Counter = Counter()
    for cell in series.dropna():
        # Use short phrases (2-grams) and single words for topic detection.
        text = str(cell).lower()
        words = re.findall(r'\b[a-z]{4,}\b', text)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        counter.update(words + bigrams)

    gaps = []
    for term, cnt in counter.items():
        if cnt / total_participants >= KB_GAP_THRESHOLD:
            if not any(kb in term or term in kb for kb in KB_TOPICS):
                gaps.append((term, cnt))

    if gaps:
        print("\n  KNOWLEDGE BASE GAP WARNING — topics mentioned by >20% of")
        print("  participants that may not be covered by the current KB:")
        for term, cnt in sorted(gaps, key=lambda x: -x[1]):
            print(f"    '{term}': {cnt} mentions ({_pct(cnt, total_participants)})")
    else:
        print("\n  KB gap check: no gaps detected above 20% threshold.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"[preprocess_queries] ERROR: Input file not found:\n  {INPUT_FILE}")
        print("  Place the Google Form CSV export at that path and re-run.")
        return

    df = pd.read_csv(INPUT_FILE)

    if QUERY_COL not in df.columns:
        print(
            f"[preprocess_queries] ERROR: Column '{QUERY_COL}' not found.\n"
            f"  Columns present: {list(df.columns)}"
        )
        return

    # --- Load existing output state for rerun deduplication ---
    existing_timestamps, existing_query_texts, max_query_number, existing_pids = (
        _load_existing_state()
    )
    is_rerun = bool(existing_timestamps or existing_query_texts)
    prev_participant_count = len(existing_timestamps)
    prev_query_count       = max_query_number

    # seen_lower starts from all queries already in the output files so
    # cross-run duplicates are caught the same way as within-run ones.
    seen_lower: set[str] = set(existing_query_texts)

    drop_counts   = {"empty": 0, "filler": 0, "duplicate": 0}
    query_counter = max_query_number  # continues from last run

    new_label_studio_rows: list[dict] = []
    new_metadata_rows:     list[dict] = []
    new_summary_rows:      list[dict] = []

    zero_query_participants: list[str] = []
    total_participants = len(df)

    # participant_id is positional in the full export so P001 is always
    # the same person across reruns. Row index + 1 gives a stable position.
    for row_idx, row in df.iterrows():
        participant_id = f"{PARTICIPANT_ID_PREFIX}{(row_idx + 1):03d}"
        timestamp      = str(row.get(TIMESTAMP_COL, "")).strip()

        # Skip participants whose Timestamp is already in the output files.
        if timestamp and timestamp in existing_timestamps:
            continue

        age_range  = row.get(AGE_COL, "")
        gender     = row.get(GENDER_COL, "")
        religion   = row.get(RELIGION_COL, "")
        language   = row.get(LANGUAGE_COL, "")
        info_src   = row.get(INFO_SOURCES_COL, "")
        uncomfort  = row.get(UNCOMFORTABLE_COL, "")
        uncomf_why = row.get(UNCOMFORTABLE_WHY_COL, "")
        chatbot_h  = row.get(CHATBOT_HELPFUL_COL, "")
        trust      = row.get(TRUST_FACTORS_COL, "")
        helpline   = row.get(HELPLINE_COL, "")
        topics_n   = row.get(TOPICS_NEEDED_COL, "")
        topics_w   = row.get(TOPICS_WANTED_COL, "")
        raw_cell   = row.get(QUERY_COL, "")

        participant_query_count = 0

        if not (pd.isna(raw_cell) or not str(raw_cell).strip()):
            for fragment in _split_cell(str(raw_cell)):
                drop, reason = _should_drop(fragment, seen_lower)
                if drop:
                    drop_counts[reason] += 1
                    continue

                query_counter += 1
                query_id = f"{QUERY_ID_PREFIX}{query_counter:03d}"

                new_label_studio_rows.append({"id": query_id, "text": fragment.strip()})
                new_metadata_rows.append({
                    "id":             query_id,
                    "text":           fragment.strip(),
                    "participant_id": participant_id,
                    "age_range":      age_range,
                    "gender":         gender,
                    "language":       language,
                })
                seen_lower.add(fragment.strip().lower())
                participant_query_count += 1

        if participant_query_count == 0 and participant_id not in existing_pids:
            zero_query_participants.append(participant_id)

        new_summary_rows.append({
            TIMESTAMP_COL:      timestamp,
            "participant_id":   participant_id,
            "age_range":        age_range,
            "gender":           gender,
            "religion":         religion,
            "language_preference": language,
            "information_sources": info_src,
            "felt_uncomfortable":  uncomfort,
            "uncomfortable_reason": uncomf_why,
            "chatbot_helpful":     chatbot_h,
            "trust_factors":       trust,
            "helpline_support":    helpline,
            "topics_needed":       topics_n,
            "topics_wanted":       topics_w,
            "queries_extracted":   participant_query_count,
        })

    # --- Append to output files ---
    _append_csv(
        OUTPUT_LABEL_STUDIO,
        new_label_studio_rows,
        ["id", "text"],
    )
    _append_csv(
        OUTPUT_WITH_METADATA,
        new_metadata_rows,
        ["id", "text", "participant_id", "age_range", "gender", "language"],
    )
    _append_csv(
        OUTPUT_PARTICIPANT_SUMMARY,
        new_summary_rows,
        [TIMESTAMP_COL, "participant_id", "age_range", "gender", "religion",
         "language_preference", "information_sources", "felt_uncomfortable",
         "uncomfortable_reason", "chatbot_helpful", "trust_factors",
         "helpline_support", "topics_needed", "topics_wanted",
         "queries_extracted"],
    )

    # --- Load full participant summary for demographic stats ---
    full_summary = pd.read_csv(OUTPUT_PARTICIPANT_SUMMARY, dtype=str)
    total_all_participants = len(full_summary)
    new_participants_added = len(new_summary_rows)
    new_queries_added      = len(new_label_studio_rows)
    total_all_queries      = query_counter

    # category counts over NEW queries only (all would need re-reading LS file)
    category_counts = {"educational": 0, "diagnostic": 0, "crisis": 0}
    for r in new_label_studio_rows:
        category_counts[_estimate_category(r["text"])] += 1

    total_extracted_this_run = len(new_label_studio_rows) + sum(drop_counts.values())
    avg_per_participant = (
        total_all_queries / total_all_participants if total_all_participants else 0
    )

    # --- Terminal summary ---
    W = 62
    print()
    print("=" * W)
    print("  preprocess_queries.py -- SUMMARY")
    print("=" * W)

    if is_rerun:
        print("  MODE: RERUN (appending new data only)")
        print()
        print(f"  Participants in form export this run : {total_participants}")
        print(f"  Previously processed participants    : {prev_participant_count}")
        print(f"  NEW participants added this run      : {new_participants_added}")
        print(f"  Previously processed queries         : {prev_query_count}")
        print(f"  NEW queries added this run           : {new_queries_added}")
        print(f"  Combined total participants          : {total_all_participants}")
        print(f"  Combined total queries               : {total_all_queries}")
    else:
        print("  MODE: FIRST RUN")
        print()
        print(f"  Participants processed       : {total_participants}")
        print(f"  Raw query_text cells read    : {df[QUERY_COL].notna().sum()}")
        print(f"  Fragments extracted (total)  : {total_extracted_this_run}")

    print()
    print("  Dropped fragments (this run):")
    print(f"    Duplicate          : {drop_counts['duplicate']}")
    print(f"    Empty / whitespace : {drop_counts['empty']}")
    print(f"    Filler answer      : {drop_counts['filler']}")
    print(f"    Total dropped      : {sum(drop_counts.values())}")
    print()
    print(f"  Final query count (all runs)    : {total_all_queries}")
    print(f"  Average queries/participant     : {avg_per_participant:.1f}")
    print()
    print("  Category distribution this run (ESTIMATE ONLY -- keyword-based,")
    print("  NOT the annotation labels):")
    print(f"    Educational : {category_counts['educational']}")
    print(f"    Diagnostic  : {category_counts['diagnostic']}")
    print(f"    Crisis      : {category_counts['crisis']}")

    # --- Demographic statistics (based on full participant_summary.csv) ---
    print()
    print("-" * W)
    print("  PARTICIPANT SUMMARY STATISTICS (all runs combined)")
    print("-" * W)

    n = len(full_summary)
    _distribution(full_summary["gender"],             "Gender distribution", n)
    _distribution(full_summary["age_range"],          "Age range distribution", n)
    _distribution(full_summary["language_preference"],"Language preference", n)
    _distribution(full_summary["felt_uncomfortable"], "Felt uncomfortable asking SRH question", n)
    _distribution(full_summary["chatbot_helpful"],    "Chatbot would help young people", n)
    _distribution(full_summary["helpline_support"],   "Should chatbot recommend helpline", n)

    _top_keywords(full_summary["trust_factors"],  5, "Top 5 trust factor keywords")
    _top_keywords(full_summary["topics_needed"],  5, "Top 5 SRH topics needed keywords")

    _kb_gap_check(full_summary["topics_needed"], n)

    print()
    print("  Output files written:")
    print(f"    {OUTPUT_LABEL_STUDIO}")
    print(f"    {OUTPUT_WITH_METADATA}")
    print(f"    {OUTPUT_PARTICIPANT_SUMMARY}")
    print()

    if zero_query_participants:
        print(
            f"  WARNING: {len(zero_query_participants)} new participant(s) "
            f"produced 0 usable queries:"
        )
        for pid in zero_query_participants:
            print(f"    {pid}")
        print()

    print("=" * W)
    print()


def _append_csv(path: str, new_rows: list[dict], columns: list[str]):
    """
    Appends new_rows to the CSV at path. Creates the file with a header
    row if it does not yet exist; appends without a header if it does.
    Does nothing if new_rows is empty.
    """
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows, columns=columns)
    if os.path.exists(path):
        new_df.to_csv(path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
