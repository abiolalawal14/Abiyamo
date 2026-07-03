"""
escalation.py

Purpose:
    Detects whether an incoming message describes a crisis situation or
    a personal diagnostic/medical concern, and returns a scripted
    response with facility referrals instead of passing the query to the
    LLM. This module enforces one of the project's core safety rules:
    diagnostic and crisis queries NEVER reach the RAG pipeline or the
    Gemini generator.

Where this fits in the pipeline (chat_handler.py):
    [message received]
        |
        v
    should_escalate()  <- THIS MODULE
        |
    YES -> get_escalation_response()  <- scripted text + facility referrals
        |
    NO  -> retrieve_relevant_chunks() -> generate_answer()  (RAG path)

Detection strategy - Phase 3 (keyword matching):
    The Phase 2 intent classifier (DistilBERT, 3-class: educational /
    diagnostic / crisis) will replace this once the annotation data is
    complete and the model is trained. For now, a curated keyword list
    is the safest available option.

    Design decisions for the keyword lists:
    - Err on the side of escalation: a false positive (escalating an
      educational question) is far less harmful than a false negative
      (sending a crisis or diagnostic query through the LLM).
    - Crisis keywords cover direct self-harm expressions AND
      abuse/assault disclosures -- both require human support, not a
      chatbot response.
    - Diagnostic keywords cover personal symptom descriptions ("I have
      discharge", "I think I'm pregnant"), not general health questions
      ("what causes discharge?" is educational and should NOT escalate).
    - Language: English only for now. Once translator.py is active,
      translation happens BEFORE escalation, so the escalation check
      always sees English text.

Replacing keyword detection with the Phase 2 classifier:
    When DistilBERT is trained and exported (Phase 2), replace the
    _detect_type() function body with a call to the classifier.
    The public interface (should_escalate, get_escalation_response,
    get_helplines_for_state) does not need to change -- only
    _detect_type() changes internally.

Helplines / Facilities:
    Loaded from data/helplines.json. Run scripts/import_facilities.py
    once to populate the "states" section with real Nigerian health
    facilities. The 112 emergency number is the only phone number
    confirmed active; all others remain placeholders until verified.

Response format:
    Returns a plain string -- warm acknowledgement, brief guidance, and
    up to 3 named facilities in the user's state. Kept short for WhatsApp.
    Goal is to hand off to a human resource, not counsel directly.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Helplines data
# ---------------------------------------------------------------------------

_HELPLINES_PATH = Path("data/helplines.json")

# Module-level cache -- same pattern used throughout this project.
# helplines.json is small and rarely changes; loading once at first call
# avoids repeated file I/O on every incoming message.
_helplines_cache = None


def _load_helplines() -> dict:
    """
    Loads and caches data/helplines.json. Returns an empty dict on missing
    file so the module still works (with a generic fallback) even without it.
    """
    global _helplines_cache
    if _helplines_cache is None:
        if not _HELPLINES_PATH.exists():
            print(
                f"[escalation] Warning: {_HELPLINES_PATH} not found -- "
                "facility referrals will not appear in escalation responses."
            )
            _helplines_cache = {}
        else:
            with open(_HELPLINES_PATH, "r", encoding="utf-8") as f:
                _helplines_cache = json.load(f)
    return _helplines_cache


# ---------------------------------------------------------------------------
# Detection: keyword lists
# ---------------------------------------------------------------------------

# CRISIS_PHRASES -- phrases that indicate the user may be in immediate
# danger or acute emotional distress. Any match escalates to "crisis".
# Deliberately broad: false positives are acceptable; false negatives
# (missing a real crisis) are not.
CRISIS_PHRASES = [
    # suicidal ideation / self-harm
    "kill myself",
    "want to die",
    "end my life",
    "take my own life",
    "suicide",
    "no reason to live",
    "can't go on",
    "cannot go on",
    "don't want to live",
    "do not want to live",
    "hurt myself",
    "harm myself",
    # sexual violence / abuse disclosure
    "i was raped",
    "someone raped me",
    "he raped me",
    "they raped me",
    "i am being abused",
    "i am being hurt",
    "he is hurting me",
    "someone is hurting me",
    "sexual abuse",
    "sexually abused",
    "sexual assault",
    "sexually assaulted",
    "forced me to have sex",
    "forced sex",
    "he forced me",
    # assault disclosures -- "i was assaulted" is a recognised first-disclosure
    "i was assaulted",
    "assaulted by",
    "assaulted me",
    # vague disclosures -- survivors often can't name what happened directly
    "something happened to me",
    "what happened to me",
    # non-consensual touch -- covers grooming and inappropriate contact
    "he touched me",
    "they touched me",
    "someone touched me",
    "touching me inappropriately",
    "touching me in ways",
    # coercion and non-consent
    "i said no but",
    "forcing himself",
    "without my consent",
    "i did not agree to",
    "i didn't agree to",
    "pressuring me to",
    # physical abuse in intimate relationships
    "he beats me",
    "he hit me",
    "he hits me",
    # self-harm -- "hurting myself" distinct from "hurt myself" (verb form)
    "hurting myself",
    "thinking about ending",
    # drink spiking
    "something in my drink",
    # trafficking / exploitation
    "being trafficked",
    "they won't let me leave",
    "they took my phone",
    "i need help to escape",
]

# DIAGNOSTIC_PHRASES -- phrases that describe personal symptoms or
# health conditions requiring clinical assessment. Any match escalates
# to "diagnostic". These cover PERSONAL descriptions, not general
# educational questions about symptoms.
DIAGNOSTIC_PHRASES = [
    # pregnancy concerns
    "i think i'm pregnant",
    "i think i am pregnant",
    "i might be pregnant",
    "could i be pregnant",
    "am i pregnant",
    "missed my period",
    "i missed my period",
    "late period",
    "pregnancy test",
    "i took a pregnancy test",
    # STI / genital symptoms (personal)
    "i have discharge",
    "i have a discharge",
    "unusual discharge",
    "i have sores",
    "i have a sore",
    "blisters down there",
    "sores down there",
    "itching down there",
    "burning when i pee",
    "burns when i pee",
    "burning when i urinate",
    "pain when i pee",
    "pain when i urinate",
    "smells bad down there",
    "bad smell down there",
    "i have a rash",
    "rash on my private",
    "rash on my genitals",
    # general personal symptom framing
    "i think i have",
    "do i have",
    "what do i have",
    "what is wrong with me",
    "i've been feeling sick",
    "i have been feeling sick",
    "i feel sick",
    "something is wrong with me",
    # HIV/STI personal concern
    "i think i have hiv",
    "i might have hiv",
    "do i have hiv",
    "i think i have an sti",
    "i might have an sti",
    "i think i have an infection",
    # post-exposure / emergency
    "i need emergency contraception",
    "i need the morning after pill",
    "we had unprotected sex",
    "condom broke",
    "condom burst",
]


def _detect_type(message: str) -> str | None:
    """
    Returns the escalation type for a message, or None if no escalation
    is needed.

    Returns:
        "crisis"     -- message contains crisis/self-harm/abuse signals
        "diagnostic" -- message contains personal symptom descriptions
        None         -- message appears to be an educational question

    Priority: crisis takes precedence over diagnostic. If both match,
    "crisis" is returned -- the user gets crisis resources first.

    TODO (Phase 2): Replace this function body with a call to the
    trained DistilBERT intent classifier once it is available. The
    classifier will handle multi-label cases, short ambiguous messages,
    and non-literal phrasing that keyword matching misses. The return
    values and None-for-educational convention must stay the same so
    the callers don't need to change.
    """
    lower = message.lower().strip()

    for phrase in CRISIS_PHRASES:
        if phrase in lower:
            return "crisis"

    for phrase in DIAGNOSTIC_PHRASES:
        if phrase in lower:
            return "diagnostic"

    return None


# ---------------------------------------------------------------------------
# State-aware facility lookup
# ---------------------------------------------------------------------------

# Variants a user or the session layer might pass for the FCT.
# The facility dataset normalises FCT to "Fct" after title-case.
_FCT_VARIANTS = {"fct", "abuja", "fct abuja", "federal capital territory", "abuja fct"}


def _normalize_state_name(state: str | None) -> str | None:
    """
    Normalises a state name so it matches the title-cased keys in
    helplines.json. Handles FCT/Abuja variants explicitly because "Fct"
    is an unusual title-case result that users would never type.
    """
    if not state or not state.strip():
        return None
    if state.strip().lower() in _FCT_VARIANTS:
        return "Fct"
    return state.strip().title()


def get_helplines_for_state(state: str | None) -> list:
    """
    Returns the list of facility dicts for the given Nigerian state from
    data/helplines.json. Returns an empty list when the state is unknown,
    not provided, or the file has not been populated yet.

    Case-insensitive matching so callers can pass "borno" or "Borno".
    """
    if not state:
        return []

    data = _load_helplines()
    normalized = _normalize_state_name(state)
    if not normalized:
        return []

    states = data.get("states", {})
    # Case-insensitive scan so minor spelling differences don't miss a state
    normalized_lower = normalized.lower()
    for key, val in states.items():
        if key.lower() == normalized_lower:
            return val.get("facilities", [])

    return []


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _crisis_response(state: str | None = None) -> str:
    """
    Scripted crisis response. When a state is known, names real local
    facilities so the user has a concrete place to go. Always adds 112.

    Follows safe messaging: acknowledge warmth, validate, give resources,
    encourage action. Short by design -- WhatsApp users disengage with
    long messages.
    """
    facilities = get_helplines_for_state(state)

    if facilities:
        state_display = _normalize_state_name(state) or state
        lines = "\n".join(
            f"- {f['name']} - {f['lga']} LGA"
            for f in facilities[:3]
        )
        return (
            "I am really glad you reached out and I want to make sure you "
            "get the right support right now. Please visit one of these "
            f"health facilities in {state_display} as soon as you can - "
            "they provide confidential support:\n\n"
            f"{lines}\n\n"
            "You can also call Nigeria's emergency line: 112 (available 24/7)"
        )

    # No state info or state not yet in the dataset -- fall back to
    # generic message that still gives the user a concrete action (112).
    data = _load_helplines()
    fallback_entries = data.get("national_fallback", [{}])
    fallback_note = fallback_entries[0].get(
        "note", "Visit your nearest Primary Health Centre for confidential SRH support"
    ) if fallback_entries else "Visit your nearest Primary Health Centre for confidential SRH support"

    return (
        "I hear you, and I want you to know you are not alone.\n\n"
        "What you are going through sounds very difficult, and you deserve "
        "real support from a person who can help - not just a chatbot.\n\n"
        f"{fallback_note}\n\n"
        "You can also call Nigeria's emergency line: 112 (available 24/7)\n\n"
        "You are important, and help is available."
    )


def _diagnostic_response(state: str | None = None) -> str:
    """
    Scripted diagnostic response. Encourages visiting a named health
    facility rather than diagnosing through a chatbot. Falls back to
    generic PHC guidance when no state is known.
    """
    facilities = get_helplines_for_state(state)

    if facilities:
        state_display = _normalize_state_name(state) or state
        lines = "\n".join(
            f"- {f['name']} - {f['lga']} LGA"
            for f in facilities[:3]
        )
        return (
            "Thank you for sharing this with me. This sounds like something "
            "a qualified health professional should look at directly. Here "
            f"are verified health facilities in {state_display} where you "
            "can get confidential support:\n\n"
            f"{lines}\n\n"
            "Please visit any of these facilities and ask for the SRH or "
            "reproductive health unit."
        )

    data = _load_helplines()
    fallback_entries = data.get("national_fallback", [{}])
    fallback_note = fallback_entries[0].get(
        "note", "Visit your nearest Primary Health Centre for confidential SRH support"
    ) if fallback_entries else "Visit your nearest Primary Health Centre for confidential SRH support"

    return (
        "It sounds like you may have a personal health concern that needs "
        "attention from a trained health worker.\n\n"
        "I can share general information about sexual and reproductive health, "
        "but I am not able to assess symptoms or give you a diagnosis - only "
        "a health professional can do that safely.\n\n"
        f"{fallback_note}\n\n"
        "Your health matters. Please do not wait if something feels wrong."
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def should_escalate(message: str) -> bool:
    """
    Returns True if the message should be routed to escalation rather
    than the RAG pipeline. Called first in chat_handler.handle_message()
    before any retrieval or generation happens.

    A True result means the LLM must NOT be called for this message.
    """
    return _detect_type(message) is not None


def get_escalation_response(message: str, state: str | None = None) -> str:
    """
    Returns the appropriate scripted response for an escalated message.
    Should only be called after should_escalate() has returned True.

    Parameters:
        message : the user's message text (used to detect escalation type),
                  OR one of "crisis" / "diagnostic" as a test shorthand.
        state   : optional Nigerian state name (e.g. "Borno", "Lagos").
                  When provided, the response includes named local facilities.
                  When None, falls back to generic national guidance.

    Backward compatible: callers that pass only message (no state) continue
    to work -- they receive the national-fallback response.
    """
    # Allow passing the type directly for testing ("crisis", "diagnostic")
    # without needing a real message that triggers keyword detection.
    if message in ("crisis", "diagnostic"):
        escalation_type = message
    else:
        escalation_type = _detect_type(message)

    if escalation_type == "crisis":
        return _crisis_response(state)
    else:
        # Covers both "diagnostic" and the defensive fallback for None
        return _diagnostic_response(state)


# ---------------------------------------------------------------------------
# Quick manual test -- runs only when this file is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # (message, expected_type)
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

    print("=== TEST 1: Detection accuracy ===\n")
    passed = 0
    for msg, expected in test_cases:
        detected = _detect_type(msg)
        status = "PASS" if detected == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] '{msg[:50]}'")
        if status == "FAIL":
            print(f"         expected={expected!r}  got={detected!r}")

    print(f"\n{passed}/{len(test_cases)} passed")

    print("\n=== TEST 2: should_escalate() ===")
    print(f"  Educational: {should_escalate('What is family planning?')} (expected False)")
    print(f"  Diagnostic:  {should_escalate('I think I am pregnant')} (expected True)")
    print(f"  Crisis:      {should_escalate('I want to kill myself')} (expected True)")

    print("\n=== TEST 3: Crisis response (no state) ===\n")
    print(get_escalation_response("I want to kill myself"))

    print("\n=== TEST 4: Diagnostic response (no state) ===\n")
    print(get_escalation_response("I think I might be pregnant"))

    print("\n=== TEST 5: State-aware responses ===")
    borno = get_helplines_for_state("Borno")
    print(f"  Borno facilities loaded: {len(borno)}")
    if borno:
        print(f"  First: {borno[0]['name']} - {borno[0]['lga']} LGA")
    print()
    print(get_escalation_response("diagnostic", "Borno"))

    print("\n=== TEST 6: Helplines file keys ===")
    data = _load_helplines()
    print(f"  Keys: {list(data.keys())}")
    print(f"  States in file: {len(data.get('states', {}))}")
