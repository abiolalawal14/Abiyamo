"""
merge_annotations.py

Purpose:
    Combine two annotators' Label Studio CSV exports into a single file,
    and flag queries where the two annotators disagree, so those can be
    reviewed (by a third reviewer, or discussed between the two
    annotators) before finalizing the labeled dataset.

Design notes:
    - Labels are MULTI-LABEL by design: a single query can genuinely be
      tagged with more than one of educational / diagnostic / crisis
      (e.g. a question that is both informational and symptom-related).
    - Expects Label Studio's CSV export with labels stored as a single
      column containing comma-separated values, e.g. "educational" or
      "educational,diagnostic". If your actual export format differs
      (e.g. separate True/False columns per category), only the
      parse_labels() function below needs to change — everything else
      stays the same.
    - This file does NOT calculate Cohen's Kappa itself. Kappa will be
      calculated in JMP, using this file's long-format export
      (see export_for_jmp()). This file's job is purely to merge,
      clean, and reshape the data so it's ready for that step.

Expected input format (adjust column names below if yours differ):
    Each annotator's CSV should have at minimum:
        - a column identifying the query (e.g. "query" or "text")
        - a column with that annotator's labels (comma-separated)

Requirements (install via pip):
    pip install pandas
"""

import pandas as pd
import os

# Column name settings — adjust these if your actual CSV column names differ
QUERY_ID_COLUMN = "id"          # unique identifier for each query/row
QUERY_TEXT_COLUMN = "query"      # the actual question text
LABEL_COLUMN = "label"           # the column containing comma-separated labels

VALID_CATEGORIES = ["educational", "diagnostic", "crisis"]


# ---------------------------------------------------------------------------
# Step 1: Parse a raw label string into a clean set of categories
# ---------------------------------------------------------------------------

def parse_labels(raw_label_string) -> set:
    """
    Converts a raw label string like "educational, diagnostic" into a
    clean set: {"educational", "diagnostic"}.

    Handles common messiness: extra spaces, inconsistent capitalization,
    and missing/empty values (returns an empty set rather than failing).

    If your Label Studio export uses a different format (e.g. separate
    boolean columns instead of one comma-separated column), this is the
    only function that needs to change — everything downstream expects
    a set of lowercase category strings, regardless of how it got built.
    """
    if pd.isna(raw_label_string) or not str(raw_label_string).strip():
        return set()

    raw_parts = str(raw_label_string).split(",")
    cleaned = {part.strip().lower() for part in raw_parts if part.strip()}

    # Flag anything that doesn't match expected categories, rather than
    # silently dropping it — typos in labeling are common and worth
    # catching early, before they quietly corrupt the dataset.
    unexpected = cleaned - set(VALID_CATEGORIES)
    if unexpected:
        print(f"[merge_annotations] Warning: unexpected label(s) found: {unexpected}")

    return cleaned


# ---------------------------------------------------------------------------
# Step 2: Load one annotator's file and parse their labels
# ---------------------------------------------------------------------------

def load_annotator_file(file_path: str, annotator_name: str) -> pd.DataFrame:
    """
    Loads a single annotator's Label Studio CSV export and returns a
    cleaned DataFrame with parsed label sets.

    Parameters:
        file_path: path to the annotator's exported CSV.
        annotator_name: a short label for this annotator (e.g. "annotator_1"),
                          used to name columns so the two annotators'
                          labels don't overwrite each other when merged.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Annotator file not found: {file_path}")

    df = pd.read_csv(file_path)

    missing_columns = [col for col in [QUERY_ID_COLUMN, QUERY_TEXT_COLUMN, LABEL_COLUMN]
                        if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Expected columns missing from {file_path}: {missing_columns}. "
            f"Found columns: {list(df.columns)}. "
            f"Update QUERY_ID_COLUMN / QUERY_TEXT_COLUMN / LABEL_COLUMN at "
            f"the top of this file to match your actual export."
        )

    df[f"{annotator_name}_labels"] = df[LABEL_COLUMN].apply(parse_labels)

    return df[[QUERY_ID_COLUMN, QUERY_TEXT_COLUMN, f"{annotator_name}_labels"]]


# ---------------------------------------------------------------------------
# Step 3: Merge both annotators' files and flag disagreements
# ---------------------------------------------------------------------------

def merge_annotations(annotator_1_path: str, annotator_2_path: str) -> pd.DataFrame:
    """
    Loads both annotators' files, merges them on the query ID, and adds
    a column indicating whether the two annotators agreed.

    Returns:
        A DataFrame with columns:
            - query id and text
            - annotator_1_labels, annotator_2_labels (sets)
            - agreement (True if both sets are identical, False otherwise)
            - needs_review (True if agreement is False — these are the
               rows a third reviewer should look at)
    """
    df1 = load_annotator_file(annotator_1_path, "annotator_1")
    df2 = load_annotator_file(annotator_2_path, "annotator_2")

    merged = pd.merge(df1, df2, on=[QUERY_ID_COLUMN, QUERY_TEXT_COLUMN], how="outer")

    # Rows that only appear in one annotator's file mean something went
    # wrong upstream (e.g. one annotator skipped a query) — flag rather
    # than silently dropping, since this affects how many usable queries
    # you actually end up with.
    missing_either = merged[
        merged["annotator_1_labels"].isna() | merged["annotator_2_labels"].isna()
    ]
    if not missing_either.empty:
        print(f"[merge_annotations] Warning: {len(missing_either)} quer(y/ies) "
              f"are missing from one annotator's file. Check the queries below:")
        print(missing_either[[QUERY_ID_COLUMN, QUERY_TEXT_COLUMN]].to_string(index=False))

    merged["agreement"] = merged.apply(
        lambda row: row["annotator_1_labels"] == row["annotator_2_labels"],
        axis=1
    )
    merged["needs_review"] = ~merged["agreement"]

    agreement_rate = merged["agreement"].mean() * 100
    print(f"\n[merge_annotations] Raw agreement rate: {agreement_rate:.1f}% "
          f"({merged['agreement'].sum()}/{len(merged)} queries)")
    print(f"[merge_annotations] {merged['needs_review'].sum()} quer(y/ies) need third-reviewer attention.")

    return merged


# ---------------------------------------------------------------------------
# Step 4: Save the merged result for use by calculate_kappa.py and for
# manual review of disagreements
# ---------------------------------------------------------------------------

def save_merged_annotations(merged_df: pd.DataFrame, output_path: str):
    """
    Saves the merged annotations to CSV. Label sets are converted to
    comma-separated strings for readability in the saved file (sets
    don't save cleanly to CSV otherwise).
    """
    output_df = merged_df.copy()
    output_df["annotator_1_labels"] = output_df["annotator_1_labels"].apply(
        lambda s: ",".join(sorted(s)) if isinstance(s, set) else ""
    )
    output_df["annotator_2_labels"] = output_df["annotator_2_labels"].apply(
        lambda s: ",".join(sorted(s)) if isinstance(s, set) else ""
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"[merge_annotations] Saved merged annotations to {output_path}")


# ---------------------------------------------------------------------------
# Step 5: Export in long format, for use in JMP's Agreement/Kappa analysis
# ---------------------------------------------------------------------------

def export_for_jmp(merged_df: pd.DataFrame, output_path: str):
    """
    Exports the merged annotations in LONG format for use in JMP's
    Cohen's Kappa / Agreement Statistic analysis.

    NOTE: I am not fully certain this exact column layout matches what
    JMP expects — JMP's required format can vary depending on which
    specific analysis platform you use (e.g. Analyze > Agreement
    Statistic, or Fit Y by X). Treat this as a reasonable starting
    point. Please confirm the exact expected format in JMP's own
    documentation or with your statistics professor, and adjust this
    function if the shape needs to differ.

    Output format — one row per query, per category:
        query_id | category      | annotator_1 | annotator_2
        ---------|----------------|-------------|------------
        1        | educational    | 1           | 1
        1        | diagnostic     | 0           | 1
        1        | crisis         | 0           | 0
        2        | educational    | 1           | 1
        2        | diagnostic     | 0           | 0
        2        | crisis         | 0           | 0
        ...

    Each row represents a single yes/no judgment by both annotators on
    whether one specific category applies to one specific query. This
    "long" shape is the common format expected by most statistical
    software (including JMP, in most of its agreement analysis tools)
    for calculating Kappa — as opposed to the "wide" shape used in the
    main merged CSV, which keeps both annotators' full label sets in a
    single column per query.

    Parameters:
        merged_df: the DataFrame returned by merge_annotations(), with
                    annotator_1_labels and annotator_2_labels as sets.
        output_path: where to save the long-format CSV.
    """
    rows = []

    for _, row in merged_df.iterrows():
        query_id = row[QUERY_ID_COLUMN]
        annotator_1_set = row["annotator_1_labels"] if isinstance(row["annotator_1_labels"], set) else set()
        annotator_2_set = row["annotator_2_labels"] if isinstance(row["annotator_2_labels"], set) else set()

        for category in VALID_CATEGORIES:
            rows.append({
                "query_id": query_id,
                "category": category,
                "annotator_1": 1 if category in annotator_1_set else 0,
                "annotator_2": 1 if category in annotator_2_set else 0,
            })

    long_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    long_df.to_csv(output_path, index=False)
    print(f"[merge_annotations] Saved long-format export for JMP to {output_path}")
    print(f"[merge_annotations] {len(long_df)} rows ({len(merged_df)} queries x {len(VALID_CATEGORIES)} categories)")


# ---------------------------------------------------------------------------
# Run this file directly to merge your two annotators' exports
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Update these paths to your actual annotator export files when ready.
    annotator_1_file = "data/raw_queries/annotator_1_export.csv"
    annotator_2_file = "data/raw_queries/annotator_2_export.csv"
    output_file = "data/raw_queries/merged_annotations.csv"
    jmp_export_file = "data/raw_queries/merged_annotations_for_jmp.csv"

    merged = merge_annotations(annotator_1_file, annotator_2_file)
    save_merged_annotations(merged, output_file)
    export_for_jmp(merged, jmp_export_file)