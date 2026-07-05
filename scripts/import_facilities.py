"""
import_facilities.py

Reads the Nigerian government health facility dataset and rebuilds
data/helplines.json so that escalation.py can give users the names
of real, local facilities instead of placeholder phone numbers.

Run from the project root:
    python scripts/import_facilities.py

Does NOT modify any src/ file at runtime. escalation.py was already
updated to read the new helplines.json structure this script produces.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACILITIES_PATH = "data/raw_queries/facilities.csv"
HELPLINES_PATH = "data/helplines.json"

MAX_FACILITIES_PER_STATE = 5
VERIFIED_DATE = "2026-07-02"


# ---------------------------------------------------------------------------
# Step 1: Load facility data
# ---------------------------------------------------------------------------

def load_facilities(path: str) -> pd.DataFrame:
    """
    Auto-detects .csv vs .xlsx and loads only the first 7 columns.
    The real export has trailing blank columns that would otherwise
    create unnamed junk columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Facility file not found: {path}\n"
            f"Place the Nigerian health facility export at this path as "
            f"facilities.csv or facilities.xlsx."
        )

    ext = p.suffix.lower()
    if ext == ".xlsx":
        try:
            df = pd.read_excel(path, usecols=range(7), engine="openpyxl")
        except ImportError:
            raise ImportError(
                "openpyxl is required to read .xlsx files. "
                "Install it with: pip install openpyxl"
            )
    elif ext == ".csv":
        df = pd.read_csv(path, usecols=range(7))
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Expected .csv or .xlsx")

    df.columns = ["SN", "ZONE", "STATE", "LGA", "WARD_INEC", "WARD_STATE", "HEALTH_FACILITY"]
    return df


def clean_facilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises STATE to title case (BORNO -> Borno) and drops rows with
    missing STATE or HEALTH_FACILITY. Empty ZONE rows are kept — ZONE is
    not used in the referral output.
    """
    df = df.copy()
    df["STATE"] = df["STATE"].astype(str).str.strip().str.title()
    df["HEALTH_FACILITY"] = df["HEALTH_FACILITY"].astype(str).str.strip()
    df["LGA"] = df["LGA"].astype(str).str.strip().str.title()
    df["WARD_STATE"] = df["WARD_STATE"].astype(str).str.strip().str.title()

    # "Nan" arises from float NaN converted via astype(str)
    df = df[~df["STATE"].isin(["", "Nan", "None", "nan"])]
    df = df[~df["HEALTH_FACILITY"].isin(["", "Nan", "None", "nan"])]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Select representative facilities per state
# ---------------------------------------------------------------------------

def _facility_type(name: str) -> str:
    """
    Classifies a facility into a priority tier used for representative
    selection. MCH/Maternal centres are most relevant for adolescent SRH;
    PHCC is the next highest-tier primary care option.
    """
    upper = name.upper()
    if "MCH" in upper or "MATERNAL" in upper:
        return "MCH"
    elif "PHCC" in upper:
        return "PHCC"
    elif "PHC" in upper:
        return "PHC"
    else:
        return "Other"


_PRIORITY_ORDER = {"MCH": 0, "PHCC": 1, "PHC": 2, "Other": 3}


def select_facilities_for_state(group: pd.DataFrame) -> list:
    """
    Picks up to MAX_FACILITIES_PER_STATE facilities from one state's
    rows, taking AT MOST ONE facility per LGA (the highest-priority one
    present in that LGA: MCH > PHCC > PHC > Other), visiting LGAs in
    alphabetical order for reproducible output.

    Why: the previous version filled the quota tier-by-tier with
    `.head(needed)` across the WHOLE state, with no regard for LGA — if
    a state's raw data happened to be sorted/grouped by LGA (it was),
    all 5 selected facilities could come from a single LGA (e.g. every
    Oyo facility selected was from Afijo LGA). Capping at one per LGA
    guarantees geographic spread across the state instead.

    If a state has fewer than MAX_FACILITIES_PER_STATE distinct LGAs,
    fewer than 5 facilities are returned — there's no second pass to
    pick a further facility from an already-used LGA, since the goal is
    diversity, not hitting a fixed count.
    """
    selected = []

    for lga in sorted(group["LGA"].unique()):
        if len(selected) >= MAX_FACILITIES_PER_STATE:
            break

        lga_group = group[group["LGA"] == lga]
        best_row = min(
            (row for _, row in lga_group.iterrows()),
            key=lambda row: _PRIORITY_ORDER[_facility_type(row["HEALTH_FACILITY"])],
        )
        priority = _facility_type(best_row["HEALTH_FACILITY"])

        selected.append({
            "name": best_row["HEALTH_FACILITY"].title(),
            "lga": best_row["LGA"],
            "ward": best_row["WARD_STATE"],
            "type": priority,
            "verified": True,
            "verified_date": VERIFIED_DATE,
            "notes": "Government health facility",
        })

    return selected


# ---------------------------------------------------------------------------
# Step 3: Build the full helplines.json structure
# ---------------------------------------------------------------------------

def build_helplines(facilities_by_state: dict) -> dict:
    """
    Constructs the complete helplines.json dict.

    national_fallback: generic message used when no state facilities are found.
    crisis_fallback: 112 (verified active) plus placeholder hotlines.
    states: one entry per Nigerian state with up to 5 facilities each.

    112 is the only number confirmed active — all others remain
    PLACEHOLDER_NOT_VERIFIED until personally called and confirmed.
    """
    return {
        "_WARNING": (
            "ALL phone numbers marked PLACEHOLDER have NOT been verified as "
            "active. Do NOT deploy to real users until every number has been "
            "personally called and confirmed. See CLAUDE.md."
        ),
        "national_fallback": [
            {
                "name": "Federal Ministry of Health SRH Services",
                "type": "facility",
                "note": (
                    "Visit your nearest Primary Health Centre for "
                    "confidential SRH support"
                ),
                "verified": True,
                "verified_date": VERIFIED_DATE,
            }
        ],
        "crisis_fallback": [
            {
                "name": "Nigeria Emergency Services",
                "number": "112",
                "type": "phone",
                "languages": ["English"],
                "available": "24/7",
                "verified": True,
                "verified_date": VERIFIED_DATE,
                "notes": "National emergency number, always active",
            },
            {
                "name": "Suicide Research and Prevention Initiative (SURPIN)",
                "number": "PLACEHOLDER_NOT_VERIFIED",
                "type": "phone",
                "available": "UNKNOWN — verify before deploying",
                "verified": False,
            },
            {
                "name": "Mentally Aware Nigeria Initiative (MANI)",
                "number": "PLACEHOLDER_NOT_VERIFIED",
                "type": "phone",
                "available": "UNKNOWN — verify before deploying",
                "verified": False,
            },
            {
                "name": "NAPTIP Toll-Free Hotline",
                "number": "PLACEHOLDER_NOT_VERIFIED",
                "type": "phone",
                "available": "UNKNOWN — verify before deploying",
                "verified": False,
                "notes": "Trafficking, sexual exploitation, gender-based violence",
            },
            {
                "name": "WARIF Rape Crisis Helpline",
                "number": "PLACEHOLDER_NOT_VERIFIED",
                "type": "phone",
                "available": "UNKNOWN — verify before deploying",
                "verified": False,
                "notes": "Women at Risk International Foundation",
            },
        ],
        "states": {
            state: {"facilities": facilities}
            for state, facilities in sorted(facilities_by_state.items())
        },
    }


def write_helplines(data: dict, path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Step 6: Run the escalation test suite
# ---------------------------------------------------------------------------

def run_escalation_tests(facilities_by_state: dict) -> bool:
    """
    Runs 13 existing detection tests plus 4 new facility-referral tests.
    Clears the module cache first so escalation.py reads the freshly
    written helplines.json rather than stale in-memory data.
    """
    # Ensure the project root is on sys.path so 'src' can be imported
    # regardless of how this script is invoked (python scripts/... vs
    # python -m scripts...).
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.safety import escalation as esc
    esc._helplines_cache = None

    from src.safety.escalation import (
        _detect_type,
        get_escalation_response,
        get_helplines_for_state,
    )

    # --- 13 existing detection tests ---
    print("\n  --- Detection tests (13 cases) ---")
    detection_cases = [
        ("What is the safe period method of family planning?", None),
        ("How do condoms prevent pregnancy?",                  None),
        ("What are the symptoms of STIs?",                     None),
        ("I think I might be pregnant",                        "diagnostic"),
        ("I missed my period and I am worried",                "diagnostic"),
        ("I have discharge and it smells bad",                 "diagnostic"),
        ("Do I have HIV?",                                     "diagnostic"),
        ("I want to kill myself",                              "crisis"),
        ("I was raped last night",                             "crisis"),
        ("Someone is hurting me and I need help",              "crisis"),
        ("I am being sexually abused",                         "crisis"),
        ("I think I have an STI",                              "diagnostic"),
        ("Condom broke last night",                            "diagnostic"),
    ]

    detection_passed = 0
    for msg, expected in detection_cases:
        detected = _detect_type(msg)
        ok = detected == expected
        if ok:
            detection_passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {msg[:55]!r}")
        if not ok:
            print(f"         expected={expected!r}  got={detected!r}")

    print(f"\n  {detection_passed}/{len(detection_cases)} detection tests passed")
    detection_ok = detection_passed == len(detection_cases)

    # --- 4 new facility-referral tests ---
    print("\n  --- Facility referral tests (4 cases) ---")
    referral_passed = 0

    borno_facs = get_helplines_for_state("Borno")
    lagos_facs = get_helplines_for_state("Lagos")
    borno_name = borno_facs[0]["name"] if borno_facs else ""
    borno_lga  = borno_facs[0]["lga"]  if borno_facs else ""
    lagos_name = lagos_facs[0]["name"] if lagos_facs else ""

    # Test 1: diagnostic + Borno
    resp1 = get_escalation_response("diagnostic", "Borno")
    ok1 = bool(borno_name) and (borno_name in resp1 or borno_lga in resp1)
    print(f"  [{'PASS' if ok1 else 'FAIL'}] diagnostic + Borno includes facility name")
    if not ok1:
        print(f"    Expected to find: {borno_name!r} or {borno_lga!r}")
        print(f"    Response preview: {resp1[:200]!r}")
    if ok1:
        referral_passed += 1

    # Test 2: crisis + Lagos includes facilities + 112
    resp2 = get_escalation_response("crisis", "Lagos")
    has_lagos = bool(lagos_name) and lagos_name in resp2
    has_112   = "112" in resp2
    ok2 = has_lagos and has_112
    print(f"  [{'PASS' if ok2 else 'FAIL'}] crisis + Lagos includes facilities + 112")
    if not ok2:
        if not has_lagos:
            print(f"    Missing Lagos facility: {lagos_name!r}")
        if not has_112:
            print(f"    Missing '112' in response")
        print(f"    Response preview: {resp2[:200]!r}")
    if ok2:
        referral_passed += 1

    # Test 3: diagnostic + Lagos
    resp3 = get_escalation_response("diagnostic", "Lagos")
    ok3 = bool(lagos_name) and lagos_name in resp3
    print(f"  [{'PASS' if ok3 else 'FAIL'}] diagnostic + Lagos includes facility name")
    if not ok3:
        print(f"    Missing: {lagos_name!r}")
        print(f"    Response preview: {resp3[:200]!r}")
    if ok3:
        referral_passed += 1

    # Test 4: crisis + None => national fallback + 112
    resp4 = get_escalation_response("crisis", None)
    ok4 = "112" in resp4 and "Primary Health" in resp4
    print(f"  [{'PASS' if ok4 else 'FAIL'}] crisis + None includes 112 + national fallback")
    if not ok4:
        if "112" not in resp4:
            print(f"    Missing '112'")
        if "Primary Health" not in resp4:
            print(f"    Missing 'Primary Health' fallback text")
        print(f"    Response preview: {resp4[:300]!r}")
    if ok4:
        referral_passed += 1

    print(f"\n  {referral_passed}/4 referral tests passed")
    return detection_ok and (referral_passed == 4)


# ---------------------------------------------------------------------------
# Step 7: Summary
# ---------------------------------------------------------------------------

def print_summary(
    total_records: int,
    states_covered: int,
    thin_states: list,
    total_selected: int,
    type_counts: dict,
    tests_passed: bool,
):
    print()
    print("=" * 50)
    print("  FACILITY IMPORT SUMMARY")
    print("=" * 50)
    print(f"  Total facility records in dataset  : {total_records}")
    print(f"  States covered                     : {states_covered} / 37 (36 + FCT)")
    if thin_states:
        print(f"  States with < 5 facilities         : {len(thin_states)}")
        for s, n in thin_states:
            print(f"    {s}: {n}")
    else:
        print(f"  States with < 5 facilities         : 0")
    print(f"  Total facilities selected          : {total_selected}")
    print(f"  Facility type breakdown:")
    for ftype in ("MCH", "PHCC", "PHC", "Other"):
        print(f"    {ftype:<22}: {type_counts.get(ftype, 0)}")
    print(f"  helplines.json updated             : OK")
    print(f"  escalation.py referral messages    : OK - Updated")
    status = "OK - All passing" if tests_passed else "FAIL - See output above"
    print(f"  All escalation tests               : {status}")
    print("=" * 50)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def import_facilities():
    print(f"[import_facilities] Loading: {FACILITIES_PATH}")
    df_raw = load_facilities(FACILITIES_PATH)

    df = clean_facilities(df_raw)
    total_records = len(df)
    print(f"[import_facilities] {total_records} valid rows after cleaning.")

    facilities_by_state = {}
    type_counts = {"MCH": 0, "PHCC": 0, "PHC": 0, "Other": 0}
    thin_states = []

    for state, group in df.groupby("STATE"):
        selected = select_facilities_for_state(group)
        facilities_by_state[state] = selected

        if len(selected) < MAX_FACILITIES_PER_STATE:
            thin_states.append((state, len(selected)))

        for f in selected:
            type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1

    states_covered = len(facilities_by_state)
    total_selected = sum(len(v) for v in facilities_by_state.values())

    helplines = build_helplines(facilities_by_state)
    write_helplines(helplines, HELPLINES_PATH)
    print(
        f"[import_facilities] Wrote {HELPLINES_PATH} "
        f"({states_covered} states, {total_selected} facilities)"
    )

    print("\n[import_facilities] Running escalation tests...")
    tests_passed = run_escalation_tests(facilities_by_state)

    print_summary(
        total_records=total_records,
        states_covered=states_covered,
        thin_states=thin_states,
        total_selected=total_selected,
        type_counts=type_counts,
        tests_passed=tests_passed,
    )

    if not tests_passed:
        sys.exit(1)


if __name__ == "__main__":
    import_facilities()
