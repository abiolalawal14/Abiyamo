"""
enrich_facility_contacts.py

Adds an "officer_contact" block to each facility already listed in
data/helplines.json, pulled from NPHCDA's public facility-survey API
(api.nphcda.gov.ng -- undocumented, discovered by reading the network
calls made by phc.nphcda.gov.ng, not an official published API).

For each facility, resolves state -> LGA -> ward -> facility record via
the API's boundary endpoints, fuzzy-matches on facility name (survey
names use different abbreviation conventions than our dataset -- "Phc"
vs "Primary Health Centre" -- so a stripped-token match is used, not an
exact string match), then fetches that facility's detail record for
officer_in_charge / officer_phone_number / officer_email.

Additive only: existing helplines.json fields (name/lga/ward/type/
verified/notes) are untouched. New fields are clearly unverified
(officer_contact_verified: false) and nothing in src/ reads them yet --
wiring them into escalation.py's user-facing responses is a separate,
deliberate decision, not made by this script. See CLAUDE.md item 9
(is_placeholder gate) for why unverified numbers must not silently
reach users.

Run from the project root (must use the project venv -- thefuzz/httpx
are not on the base interpreter):
    .venv/Scripts/python.exe scripts/enrich_facility_contacts.py
    .venv/Scripts/python.exe scripts/enrich_facility_contacts.py --states Abia,Lagos   # test on a subset first
    .venv/Scripts/python.exe scripts/enrich_facility_contacts.py --limit 20            # smoke test

Safe to interrupt (Ctrl-C) and re-run: progress is cached to
data/.facility_contact_cache.json and flushed periodically, so a
re-run skips facilities already resolved (match or confirmed no-match)
instead of re-querying the API for them.

This hits a real government server ~1-3 times per facility (more for
the ~37% of facilities missing a ward name, which requires scanning an
LGA's wards). ~789 facilities => a few thousand requests. A delay is
enforced between requests (--delay, default 0.2s) to stay polite; a
full run takes roughly 15-35 minutes depending on how many facilities
need the ward-scan fallback.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from thefuzz import fuzz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://api.nphcda.gov.ng/"
HELPLINES_PATH = "data/helplines.json"
CACHE_PATH = "data/.facility_contact_cache.json"

DEFAULT_DELAY_SECONDS = 0.2
REQUEST_TIMEOUT = 15.0
MAX_RETRIES = 3

# Post-strip name match must clear this to be accepted at all.
MATCH_THRESHOLD = 55
# A match at or above this is treated as confident enough to stop an
# LGA-wide ward scan early (missing-ward facilities only).
STRONG_MATCH = 85
# LGA/ward name resolution against the API's admin lists (cleaner data
# than facility names) can use a stricter bar.
ADMIN_NAME_THRESHOLD = 70

CACHE_FLUSH_EVERY = 10

# Generic facility-type words stripped before fuzzy name comparison --
# our dataset abbreviates ("Phc"), the API spells out ("Primary Health
# Centre"), so comparing raw strings scores low even for the same place.
_GENERIC_WORDS = {
    "PHC", "PHCC", "MCH", "MCHC", "HC", "HP", "PHCS",
    "HEALTH", "CENTRE", "CENTER", "CLINIC", "POST", "POSTS",
    "DISPENSARY", "PRIMARY", "MATERNAL", "CHILD", "CARE",
    "COMPREHENSIVE", "STATION", "OPERATIONAL", "BASE", "MODEL",
    "BASIC", "AND", "OF", "THE",
}

_client = httpx.Client(
    base_url=API_BASE,
    headers={"User-Agent": "Mozilla/5.0 (srh-chatbot facility enrichment; one-time research script)"},
    timeout=REQUEST_TIMEOUT,
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None, delay: float = DEFAULT_DELAY_SECONDS) -> dict | None:
    """GET with retry/backoff. Returns parsed JSON, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _client.get(path, params=params)
            if resp.status_code == 200:
                time.sleep(delay)
                return resp.json()
            if resp.status_code == 404:
                time.sleep(delay)
                return None
        except (httpx.TimeoutException, httpx.TransportError):
            pass
        time.sleep(min(2 ** attempt, 8))
    return None


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

def _core_name(name: str) -> str:
    upper = re.sub(r"[^A-Z0-9 ]", " ", name.upper())
    tokens = [t for t in upper.split() if t not in _GENERIC_WORDS]
    return " ".join(tokens) if tokens else upper.strip()


def _best_admin_match(target: str, candidates: dict) -> str | None:
    """candidates: {name_upper: id}. Returns the best-matching id, or None."""
    if not candidates:
        return None
    best_name, best_score = None, -1
    target_upper = target.upper().strip()
    for name in candidates:
        score = fuzz.token_sort_ratio(target_upper, name)
        if score > best_score:
            best_name, best_score = name, score
    if best_score >= ADMIN_NAME_THRESHOLD:
        return candidates[best_name]
    return None


def _best_facility_match(target_name: str, facility_records: list) -> tuple[dict | None, int]:
    """Returns (best record, score) among facility_records, or (None, 0)."""
    target_core = _core_name(target_name)
    best_record, best_score = None, -1
    for rec in facility_records:
        candidate_core = _core_name(rec["health_facility_full_name"])
        score = fuzz.token_sort_ratio(target_core, candidate_core)
        if score > best_score:
            best_record, best_score = rec, score
    if best_record is not None and best_score >= MATCH_THRESHOLD:
        return best_record, best_score
    return None, best_score if best_record is not None else 0


# ---------------------------------------------------------------------------
# API lookups (with in-process caching)
# ---------------------------------------------------------------------------

_states_cache: dict[str, str] = {}
_lgas_by_state_cache: dict[str, dict[str, str]] = {}
_wards_by_lga_cache: dict[str, dict[str, str]] = {}
_facilities_by_ward_cache: dict[str, list] = {}


def load_states() -> dict:
    if _states_cache:
        return _states_cache
    data = _get("boundary/states/") or {}
    for s in data.get("data", []):
        _states_cache[s["name"].upper()] = s["id"]
    return _states_cache


def load_lgas_for_state(state_id: str) -> dict:
    if state_id in _lgas_by_state_cache:
        return _lgas_by_state_cache[state_id]
    data = _get("boundary/lgas/", params={"state": state_id}) or {}
    result = {l["name"].upper(): l["id"] for l in data.get("data", [])}
    _lgas_by_state_cache[state_id] = result
    return result


def load_wards_for_lga(lga_id: str) -> dict:
    if lga_id in _wards_by_lga_cache:
        return _wards_by_lga_cache[lga_id]
    data = _get("boundary/wards/", params={"lga": lga_id}) or {}
    result = {w["name"].upper(): w["id"] for w in data.get("data", [])}
    _wards_by_lga_cache[lga_id] = result
    return result


def load_facilities_for_ward(ward_id: str) -> list:
    if ward_id in _facilities_by_ward_cache:
        return _facilities_by_ward_cache[ward_id]
    data = _get("boundary/facility/", params={"ward": ward_id}) or {}
    result = data.get("data", [])
    _facilities_by_ward_cache[ward_id] = result
    return result


def fetch_facility_detail(facility_id: int) -> dict | None:
    data = _get(f"boundary/facility/{facility_id}/")
    if not data:
        return None
    features = data.get("data", {}).get("features", [])
    if not features:
        return None
    return features[0].get("properties", {})


# ---------------------------------------------------------------------------
# Phone / field cleanup
# ---------------------------------------------------------------------------

def _clean_phone(raw) -> str | None:
    if raw is None:
        return None
    digits = re.sub(r"\D", "", str(raw))
    if not digits or digits in ("0", "00"):
        return None
    if digits.startswith("234") and len(digits) == 13:
        digits = "0" + digits[3:]
    elif len(digits) == 10:
        digits = "0" + digits
    if len(digits) != 11 or not digits.startswith("0"):
        return None  # doesn't look like a real Nigerian number -- skip rather than guess
    return digits


def _clean_email(raw) -> str | None:
    if not raw or "@" not in str(raw):
        return None
    return str(raw).strip()


# ---------------------------------------------------------------------------
# Resolution for a single facility entry
# ---------------------------------------------------------------------------

def resolve_facility_contact(state_name: str, facility: dict, states_map: dict) -> dict:
    """
    Returns a result dict:
      {"status": "matched", "officer_name":..., "officer_phone":...,
       "officer_email":..., "as_of":..., "match_score": int}
      {"status": "no_phone", "match_score": int}
      {"status": "no_match"}
      {"status": "api_error"}
    """
    state_id = states_map.get(state_name.upper())
    if not state_id:
        return {"status": "api_error", "reason": "state not found in API"}

    lgas = load_lgas_for_state(state_id)
    lga_id = _best_admin_match(facility["lga"], lgas)
    if not lga_id:
        return {"status": "no_match", "reason": "lga not resolved"}

    wards = load_wards_for_lga(lga_id)
    ward_name = facility.get("ward")
    ward_candidates_ids = []

    if isinstance(ward_name, str) and ward_name.strip():
        ward_id = _best_admin_match(ward_name, wards)
        if ward_id:
            ward_candidates_ids = [ward_id]

    if not ward_candidates_ids:
        # Missing or unresolved ward -- scan the LGA's wards, closest-named first.
        target_upper = str(ward_name).upper().strip() if isinstance(ward_name, str) else ""
        scored = sorted(
            wards.items(),
            key=lambda kv: fuzz.token_sort_ratio(target_upper, kv[0]) if target_upper else 0,
            reverse=True,
        )
        ward_candidates_ids = [wid for _, wid in scored]

    best_record, best_score = None, 0
    for ward_id in ward_candidates_ids:
        records = load_facilities_for_ward(ward_id)
        record, score = _best_facility_match(facility["name"], records)
        if record and score > best_score:
            best_record, best_score = record, score
        if best_score >= STRONG_MATCH:
            break

    if not best_record:
        return {"status": "no_match", "reason": "no facility name match in resolved ward(s)"}

    detail = fetch_facility_detail(best_record["id"])
    if not detail:
        return {"status": "api_error", "reason": "detail fetch failed"}

    phone = _clean_phone(detail.get("officer_phone_number"))
    officer_name = detail.get("officer_in_charge")
    officer_name = officer_name.strip() if isinstance(officer_name, str) and officer_name.strip() else None

    if not phone:
        return {"status": "no_phone", "match_score": best_score}

    return {
        "status": "matched",
        "officer_name": officer_name,
        "officer_phone": phone,
        "officer_email": _clean_email(detail.get("officer_email")),
        "as_of": (detail.get("updated_at") or "")[:10] or None,
        "match_score": best_score,
    }


# ---------------------------------------------------------------------------
# Cache (resumability)
# ---------------------------------------------------------------------------

def _facility_key(state_name: str, facility: dict) -> str:
    return f"{state_name}|{facility['lga']}|{facility['name']}"


def load_cache() -> dict:
    p = Path(CACHE_PATH)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict):
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(CACHE_PATH).write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def enrich(states_filter: list | None, limit: int | None, delay: float):
    global DEFAULT_DELAY_SECONDS
    DEFAULT_DELAY_SECONDS = delay

    helplines = json.loads(Path(HELPLINES_PATH).read_text(encoding="utf-8"))
    cache = load_cache()

    print("[enrich] Loading NPHCDA state list...")
    states_map = load_states()
    print(f"[enrich] {len(states_map)} states loaded from API.")

    all_states = list(helplines["states"].items())
    if states_filter:
        wanted = {s.strip().upper() for s in states_filter}
        all_states = [(name, v) for name, v in all_states if name.upper() in wanted]

    total_facilities = sum(len(v["facilities"]) for _, v in all_states)
    if limit:
        total_facilities = min(total_facilities, limit)

    print(f"[enrich] Processing {total_facilities} facilities across {len(all_states)} states.")
    print(f"[enrich] Delay between requests: {delay}s\n")

    stats = {"matched": 0, "no_phone": 0, "no_match": 0, "api_error": 0, "cached": 0}
    processed = 0
    unmatched_log = []

    for state_name, state_data in all_states:
        for facility in state_data["facilities"]:
            if limit and processed >= limit:
                break
            processed += 1

            key = _facility_key(state_name, facility)
            if key in cache:
                result = cache[key]
                stats["cached"] += 1
            else:
                result = resolve_facility_contact(state_name, facility, states_map)
                cache[key] = result
                if processed % CACHE_FLUSH_EVERY == 0:
                    save_cache(cache)

            status = result["status"]
            stats[status] = stats.get(status, 0) + 1

            if status == "matched":
                facility["officer_contact"] = {
                    "officer_name": result.get("officer_name"),
                    "phone": result["officer_phone"],
                    "email": result.get("officer_email"),
                    "as_of": result.get("as_of"),
                    "source": "NPHCDA facility survey (api.nphcda.gov.ng)",
                    "verified": False,
                    "notes": (
                        "Personal contact of the facility's officer-in-charge from "
                        "an NPHCDA facility survey, not an official facility hotline. "
                        "NOT phone-verified -- do not surface to users until confirmed."
                    ),
                }
            else:
                unmatched_log.append({"state": state_name, "facility": facility["name"], "status": status})

            if processed % 25 == 0 or processed == total_facilities:
                print(
                    f"[enrich] {processed}/{total_facilities} -- "
                    f"matched={stats['matched']} no_phone={stats['no_phone']} "
                    f"no_match={stats['no_match']} api_error={stats['api_error']} "
                    f"cached={stats['cached']}"
                )
        if limit and processed >= limit:
            break

    save_cache(cache)
    Path(HELPLINES_PATH).write_text(
        json.dumps(helplines, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_path = "data/facility_contact_enrichment_report.json"
    Path(report_path).write_text(json.dumps(unmatched_log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 50)
    print("  FACILITY CONTACT ENRICHMENT SUMMARY")
    print("=" * 50)
    print(f"  Facilities processed   : {processed}")
    print(f"  Matched with phone     : {stats['matched']}")
    print(f"  Matched, no phone data : {stats['no_phone']}")
    print(f"  No facility match      : {stats['no_match']}")
    print(f"  API errors             : {stats['api_error']}")
    print(f"  From cache (this run)  : {stats['cached']}")
    print(f"  {HELPLINES_PATH} updated with officer_contact blocks.")
    print(f"  Unmatched/no-phone report: {report_path}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=str, default=None,
                         help="Comma-separated state names to process (default: all 37).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap total facilities processed (for a quick smoke test).")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                         help=f"Delay in seconds between API requests (default {DEFAULT_DELAY_SECONDS}).")
    args = parser.parse_args()

    states_filter = args.states.split(",") if args.states else None
    enrich(states_filter=states_filter, limit=args.limit, delay=args.delay)


if __name__ == "__main__":
    main()
