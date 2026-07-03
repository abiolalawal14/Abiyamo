"""
merge_annotations.py

Parses the Label Studio Cloud CSV export — one row per annotator per task —
and produces two output files:
  1. merged_annotations.csv  : wide format (one row per task) for manual review
  2. merged_annotations_for_jmp.csv : long binary format for JMP Cohen's Kappa

The JMP output excludes tasks where one annotator is missing, since Kappa
cannot be calculated from unpaired ratings.
"""

import json
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Constants — update paths here if the export moves
# ---------------------------------------------------------------------------

EXPORT_PATH = "data/raw_queries/annotator_exports.csv"
MERGED_OUTPUT_PATH = "data/raw_queries/merged_annotations.csv"
JMP_OUTPUT_PATH = "data/raw_queries/merged_annotations_for_jmp.csv"
SYNTHETIC_PATH = "data/raw_queries/synthetic_queries.csv"
TRAINING_OUTPUT_PATH = "data/raw_queries/training_dataset.csv"

VALID_LABELS = ["educational", "diagnostic", "crisis"]

# Hardcoded so annotator_1 / annotator_2 columns are consistent regardless
# of row order in the export.
ANNOTATOR_1_EMAIL = "ajiboyepaul8@gmail.com"
ANNOTATOR_2_EMAIL = "aishah002.lawal@gmail.com"


# ---------------------------------------------------------------------------
# Step 2: Parse intent labels
# ---------------------------------------------------------------------------

def parse_intent(raw_intent) -> set:
    """
    Label Studio exports single labels as plain strings ("educational") and
    multi-label annotations as a JSON dict string
    ('{"choices":["educational","crisis"]}').

    Both formats must be handled. We warn on unexpected values rather than
    raising so that one malformed row never aborts the whole merge.
    """
    if pd.isna(raw_intent) or not str(raw_intent).strip():
        return set()

    raw = str(raw_intent).strip()

    if raw.startswith("{"):
        # Label Studio's multi-label format: {"choices":["educational","crisis"]}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "choices" in parsed:
                labels = {str(c).strip().lower() for c in parsed["choices"] if str(c).strip()}
            else:
                print(f"[merge_annotations] Warning: unexpected JSON structure in intent: {raw!r}")
                labels = {raw.lower()}
        except json.JSONDecodeError:
            # Malformed JSON — fall back to comma-split so we lose nothing
            labels = {part.strip().lower() for part in raw.split(",") if part.strip()}
    else:
        # Plain string ("educational") or comma-separated ("educational,diagnostic")
        labels = {part.strip().lower() for part in raw.split(",") if part.strip()}

    unexpected = labels - set(VALID_LABELS)
    if unexpected:
        print(f"[merge_annotations] Warning: unexpected label value(s) {unexpected!r} in {raw!r}")

    return labels


# ---------------------------------------------------------------------------
# Step 1: Load the CSV export
# ---------------------------------------------------------------------------

def load_export(path: str) -> pd.DataFrame:
    """
    Loads the Label Studio Cloud CSV. The export has one row per annotator
    per task, so a task annotated by two people appears twice.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Export file not found: {path}\n"
            f"Download the CSV export from Label Studio and save it at this path."
        )

    df = pd.read_csv(path)

    required_cols = {"id", "text", "intent", "annotator", "agreement"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Required columns missing from export: {missing_cols}. "
            f"Found columns: {list(df.columns)}. "
            f"Ensure the file is a Label Studio Cloud CSV export."
        )

    df["intent_set"] = df["intent"].apply(parse_intent)
    return df


# ---------------------------------------------------------------------------
# Step 3: Identify the two annotators
# ---------------------------------------------------------------------------

def identify_annotators(df: pd.DataFrame):
    """
    Returns (ann1_email, ann2_email). Uses the hardcoded email constants when
    both are present. Falls back to annotation-count ranking if unexpected
    annotators appear, so the script never crashes on a third reviewer's rows.
    """
    found = df["annotator"].unique().tolist()

    if len(found) == 1:
        raise ValueError(
            f"Only one annotator found in the export: {found[0]!r}.\n"
            f"Cohen's Kappa requires exactly two annotators.\n"
            f"Check that both annotators have submitted labels in Label Studio."
        )

    if len(found) > 2:
        counts = df["annotator"].value_counts()
        top2 = counts.index[:2].tolist()
        extras = [a for a in found if a not in top2]
        print(
            f"[merge_annotations] Warning: {len(found)} annotators found (expected 2). "
            f"Using top 2 by annotation count: {top2}. Ignoring: {extras}"
        )
        found = top2

    if ANNOTATOR_1_EMAIL in found and ANNOTATOR_2_EMAIL in found:
        return ANNOTATOR_1_EMAIL, ANNOTATOR_2_EMAIL

    # Emails don't match constants — warn and assign by first appearance
    print(
        f"[merge_annotations] Warning: annotator emails do not match expected constants.\n"
        f"  Expected : {ANNOTATOR_1_EMAIL!r}, {ANNOTATOR_2_EMAIL!r}\n"
        f"  Found    : {found[0]!r}, {found[1]!r}\n"
        f"  Assigning by first appearance in the export."
    )
    return found[0], found[1]


# ---------------------------------------------------------------------------
# Step 4: Pivot to one row per task (wide format)
# ---------------------------------------------------------------------------

def pivot_to_wide(df: pd.DataFrame, ann1: str, ann2: str) -> pd.DataFrame:
    """
    Converts long format to wide format. If Label Studio produced multiple
    rows for the same annotator + task (e.g. after re-submission), we keep
    only the most recent annotation so each annotator contributes exactly
    one rating per task.
    """
    if "created_at" in df.columns:
        df = df.sort_values("created_at")
    df = df.drop_duplicates(subset=["id", "annotator"], keep="last")

    rows = []
    for task_id, group in df.groupby("id", sort=False):
        text = group["text"].iloc[0]

        # Label Studio's pre-calculated score (100 = full agreement).
        # We keep it as a cross-check; our own boolean `agreement` column
        # is the authoritative one used downstream.
        ls_score = group["agreement"].dropna().iloc[0] if not group["agreement"].dropna().empty else None

        row1 = group[group["annotator"] == ann1]
        row2 = group[group["annotator"] == ann2]

        labels1 = row1["intent_set"].iloc[0] if not row1.empty else set()
        labels2 = row2["intent_set"].iloc[0] if not row2.empty else set()
        missing = row1.empty or row2.empty

        rows.append({
            "query_id": task_id,
            "text": text,
            "annotator_1_labels": labels1,
            "annotator_2_labels": labels2,
            "agreement_score": ls_score,
            "missing_annotator": missing,
        })

    wide = pd.DataFrame(rows)
    wide = wide.sort_values("query_id").reset_index(drop=True)

    # Step 5: Calculate agreement
    wide["agreement"] = wide["annotator_1_labels"] == wide["annotator_2_labels"]
    wide["needs_review"] = ~wide["agreement"]

    missing_ids = wide[wide["missing_annotator"]]["query_id"].tolist()
    if missing_ids:
        print(
            f"\n[merge_annotations] Warning: {len(missing_ids)} task(s) labeled by "
            f"only one annotator — excluded from JMP Kappa output:\n"
            f"  {missing_ids}"
        )

    return wide


# ---------------------------------------------------------------------------
# Step 6a: Save merged_annotations.csv
# ---------------------------------------------------------------------------

def save_merged(wide: pd.DataFrame, path: str):
    out = wide.copy()
    out["annotator_1_labels"] = out["annotator_1_labels"].apply(
        lambda s: ",".join(sorted(s)) if isinstance(s, set) else ""
    )
    out["annotator_2_labels"] = out["annotator_2_labels"].apply(
        lambda s: ",".join(sorted(s)) if isinstance(s, set) else ""
    )
    cols = [
        "query_id", "text", "annotator_1_labels", "annotator_2_labels",
        "agreement", "needs_review", "missing_annotator",
    ]
    _makedirs(path)
    out[cols].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Step 6b: Save merged_annotations_for_jmp.csv (long binary per category)
# ---------------------------------------------------------------------------

def save_jmp(wide: pd.DataFrame, path: str) -> pd.DataFrame:
    # Tasks with a missing annotator cannot contribute a paired rating,
    # so excluding them is required for Kappa to be meaningful.
    complete = wide[~wide["missing_annotator"]]

    rows = []
    for _, task in complete.iterrows():
        for cat in VALID_LABELS:
            rows.append({
                "query_id": task["query_id"],
                "category": cat,
                "annotator_1": 1 if cat in task["annotator_1_labels"] else 0,
                "annotator_2": 1 if cat in task["annotator_2_labels"] else 0,
            })

    long = pd.DataFrame(rows)
    _makedirs(path)
    long.to_csv(path, index=False)
    return long


def _makedirs(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 7: Print comprehensive summary
# ---------------------------------------------------------------------------

def print_summary(wide: pd.DataFrame, ann1: str, ann2: str):
    total = len(wide)
    both = wide[~wide["missing_annotator"]]
    missing_count = int(wide["missing_annotator"].sum())
    missing_ids = wide[wide["missing_annotator"]]["query_id"].tolist()

    full_agree = int(both["agreement"].sum())
    disagree = int((~both["agreement"]).sum())
    agree_pct = (full_agree / len(both) * 100) if len(both) > 0 else 0.0
    disagree_pct = (disagree / len(both) * 100) if len(both) > 0 else 0.0

    def _n_labeled(col):
        return int(wide[col].apply(lambda s: isinstance(s, set) and len(s) > 0).sum())

    def _label_dist(col):
        n = _n_labeled(col)
        dist = {}
        for cat in VALID_LABELS:
            count = int(wide[col].apply(lambda s: isinstance(s, set) and cat in s).sum())
            dist[cat] = (count, count / n * 100 if n > 0 else 0.0)
        multi = int(wide[col].apply(lambda s: isinstance(s, set) and len(s) > 1).sum())
        dist["multi-label"] = (multi, multi / n * 100 if n > 0 else 0.0)
        return dist, n

    a1_dist, a1_total = _label_dist("annotator_1_labels")
    a2_dist, a2_total = _label_dist("annotator_2_labels")

    print("\n" + "=" * 50)
    print("  ANNOTATION MERGE SUMMARY")
    print("=" * 50)
    print(f"  Total tasks in export         : {total}")
    print(f"  Tasks with both annotators    : {len(both)}")
    if missing_count:
        print(f"  Tasks with only 1 annotator   : {missing_count} (listed below)")
        print(f"    {missing_ids}")
    else:
        print(f"  Tasks with only 1 annotator   : 0")
    print()
    print(f"  ANNOTATOR 1: {ann1}")
    print(f"    Tasks labeled: {a1_total}")
    print()
    print(f"  ANNOTATOR 2: {ann2}")
    print(f"    Tasks labeled: {a2_total}")
    print()
    print(f"  AGREEMENT STATISTICS:")
    print(f"  Full agreement (both identical) : {full_agree} ({agree_pct:.1f}%)")
    print(f"  Disagreement (needs review)     : {disagree} ({disagree_pct:.1f}%)")
    print()
    print(f"  LABEL DISTRIBUTION:")
    print()
    print(f"  Annotator 1:")
    for cat in VALID_LABELS:
        count, pct = a1_dist[cat]
        print(f"    {cat:<12}: {count} ({pct:.1f}%)")
    count, pct = a1_dist["multi-label"]
    print(f"    {'multi-label':<12}: {count} ({pct:.1f}%)")
    print()
    print(f"  Annotator 2:")
    for cat in VALID_LABELS:
        count, pct = a2_dist[cat]
        print(f"    {cat:<12}: {count} ({pct:.1f}%)")
    count, pct = a2_dist["multi-label"]
    print(f"    {'multi-label':<12}: {count} ({pct:.1f}%)")
    print()

    disagree_rows = wide[wide["needs_review"] & ~wide["missing_annotator"]].head(5)
    if not disagree_rows.empty:
        print(f"  DISAGREEMENT EXAMPLES (first {len(disagree_rows)}):")
        for _, row in disagree_rows.iterrows():
            a1_str = ",".join(sorted(row["annotator_1_labels"])) or "(none)"
            a2_str = ",".join(sorted(row["annotator_2_labels"])) or "(none)"
            text_preview = str(row["text"])[:50]
            print(f"    {row['query_id']}: \"{text_preview}\"")
            print(f"      Ann1: {a1_str}")
            print(f"      Ann2: {a2_str}")
        print()

    print(f"  Output files written:")
    print(f"    {MERGED_OUTPUT_PATH}")
    print(f"    {JMP_OUTPUT_PATH}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Step 8: Resolve disagreements via union (produces one gold label set/task)
# ---------------------------------------------------------------------------

def resolve_disagreements(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a `resolved_labels` column: one gold label set per task, for use as
    training data for the Phase 2 intent classifier.

    - Full agreement: the shared label set is used as-is.
    - Single-annotator tasks: the one available label set is used as-is
      (there is nothing to reconcile).
    - Disagreements (both annotators present, sets differ): resolved via
      UNION — a category is included if EITHER annotator flagged it. This
      project's escalation logic is safety-critical, so under-labeling a
      crisis/diagnostic query is worse than over-including one.
    """
    out = wide.copy()

    def _resolve(row):
        if row["missing_annotator"]:
            return row["annotator_1_labels"] or row["annotator_2_labels"]
        if row["agreement"]:
            return row["annotator_1_labels"]
        return row["annotator_1_labels"] | row["annotator_2_labels"]

    out["resolved_labels"] = out.apply(_resolve, axis=1)
    return out


# ---------------------------------------------------------------------------
# Step 9: Build training_dataset.csv (resolved real annotations + synthetic)
# ---------------------------------------------------------------------------

def _labels_to_binary_cols(label_set) -> dict:
    return {cat: (1 if cat in label_set else 0) for cat in VALID_LABELS}


def build_training_dataset(resolved: pd.DataFrame, synthetic_path: str = SYNTHETIC_PATH) -> pd.DataFrame:
    """
    Combines resolved real annotations with approved synthetic queries
    (data/raw_queries/synthetic_queries.csv) into one flat table for the
    Phase 2 DistilBERT classifier.

    gold_label is the comma-joined resolved category set for real queries
    (e.g. "crisis,educational") and the single approved category for
    synthetic queries, which are single-label by design.
    """
    real_rows = []
    for _, row in resolved.iterrows():
        real_rows.append({
            "query_id": row["query_id"],
            "text": row["text"],
            **_labels_to_binary_cols(row["resolved_labels"]),
            "source": "real",
            "gold_label": ",".join(sorted(row["resolved_labels"])),
        })
    real_df = pd.DataFrame(real_rows)

    synthetic_rows = []
    if os.path.exists(synthetic_path):
        syn = pd.read_csv(synthetic_path)
        for i, srow in syn.iterrows():
            category = str(srow["category"]).strip().lower()
            synthetic_rows.append({
                "query_id": f"SYN{i + 1:03d}",
                "text": srow["text"],
                **_labels_to_binary_cols({category}),
                "source": "synthetic",
                "gold_label": category,
            })
    else:
        print(
            f"[merge_annotations] Warning: synthetic queries file not found "
            f"at {synthetic_path} — training dataset will contain real queries only."
        )
    synthetic_df = pd.DataFrame(synthetic_rows)

    training = pd.concat([real_df, synthetic_df], ignore_index=True)
    cols = ["query_id", "text", "educational", "diagnostic", "crisis", "source", "gold_label"]
    return training[cols]


def save_training_dataset(training: pd.DataFrame, path: str = TRAINING_OUTPUT_PATH):
    _makedirs(path)
    training.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Step 10: Print training dataset summary
# ---------------------------------------------------------------------------

def print_training_summary(training: pd.DataFrame):
    total = len(training)
    source_counts = training["source"].value_counts()

    print("\n" + "=" * 50)
    print("  TRAINING DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total training examples (annotated + synthetic) : {total}")
    print()
    for cat in VALID_LABELS:
        count = int(training[cat].sum())
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"    {cat:<12}=1 : {count} ({pct:.1f}%)")
    print()
    print(f"  SOURCE BREAKDOWN:")
    for source in ["real", "synthetic"]:
        count = int(source_counts.get(source, 0))
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"    {source:<10}: {count} ({pct:.1f}%)")
    print()
    print(f"  Output file written:")
    print(f"    {TRAINING_OUTPUT_PATH}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point — run directly to merge the real export file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[merge_annotations] Loading export: {EXPORT_PATH}")
    df = load_export(EXPORT_PATH)
    print(f"[merge_annotations] Loaded {len(df)} rows from export.")

    ann1, ann2 = identify_annotators(df)
    print(f"[merge_annotations] Annotator 1: {ann1}")
    print(f"[merge_annotations] Annotator 2: {ann2}")

    wide = pivot_to_wide(df, ann1, ann2)

    save_merged(wide, MERGED_OUTPUT_PATH)
    print(f"[merge_annotations] Saved merged annotations -> {MERGED_OUTPUT_PATH}")

    save_jmp(wide, JMP_OUTPUT_PATH)
    jmp_rows = len(wide[~wide["missing_annotator"]]) * len(VALID_LABELS)
    print(
        f"[merge_annotations] Saved JMP long-format -> {JMP_OUTPUT_PATH} "
        f"({jmp_rows} rows: {len(wide[~wide['missing_annotator']])} tasks x {len(VALID_LABELS)} categories)"
    )

    print_summary(wide, ann1, ann2)

    resolved = resolve_disagreements(wide)
    training = build_training_dataset(resolved)
    save_training_dataset(training, TRAINING_OUTPUT_PATH)
    print(f"[merge_annotations] Saved training dataset -> {TRAINING_OUTPUT_PATH}")
    print_training_summary(training)
