"""
escalation.py

Purpose:
    Detects whether an incoming message describes a crisis situation or
    a personal diagnostic/medical concern, and returns a scripted
    response with helpline numbers instead of passing the query to the
    LLM. This module enforces one of the project's core safety rules:
    diagnostic and crisis queries NEVER reach the RAG pipeline or the
    Gemini generator.

Where this fits in the pipeline (chat_handler.py):
    [message received]
        |
        v
    should_escalate()  ← THIS MODULE
        |
    YES → get_escalation_response()  ← scripted text + helplines
        |
    NO  → retrieve_relevant_chunks() → generate_answer()  (RAG path)

Detection strategy — Phase 3 (keyword matching):
    The Phase 2 intent classifier (DistilBERT, 3-class: educational /
    diagnostic / crisis) will replace this once the annotation data is
    complete and the model is trained. For now, a curated keyword list
    is the safest available option.

    Design decisions for the keyword lists:
    - Err on the side of escalation: a false positive (escalating an
      educational question) is far less harmful than a false negative
      (sending a crisis or diagnostic query through the LLM).
    - Crisis keywords cover direct self-harm expressions AND
      abuse/assault disclosures — both require human support, not a
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
    The public interface (should_escalate, get_escalation_response)
    does not need to change — only _detect_type() changes internally.

Helplines:
    Loaded from data/helplines.json. All numbers in that file are
    PLACEHOLDERS until personally verified as active. The formatting
    logic in this file is correct — only the JSON data needs updating.
    See CLAUDE.md and helplines.json for the verification requirement.

Response format:
    Returns a plain string — warm acknowledgement, brief guidance, and
    formatted helpline numbers. Kept short intentionally (WhatsApp
    users disengage with long messages). The goal is to hand off to a
    human resource, not to counsel the user directly.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Helplines data
# ---------------------------------------------------------------------------

_HELPLINES_PATH = Path("data/helplines.json")

# Module-level cache — same pattern used throughout this project.
# helplines.json is small and rarely changes; loading it once at first
# call avoids repeated file I/O on every incoming message.
_helplines_cache = None


def _load_helplines() -> dict:
    """
    Loads and caches data/helplines.json. Returns an empty dict if the
    file is missing rather than raising, so the escalation module still
    works (with a "contact a health worker" fallback) even if the file
    is absent.
    """
    global _helplines_cache
    if _helplines_cache is None:
        if not _HELPLINES_PATH.exists():
            print(f"[escalation] Warning: {_HELPLINES_PATH} not found — "
                  "helpline numbers will not appear in escalation responses.")
            _helplines_cache = {}
        else:
            with open(_HELPLINES_PATH, "r", encoding="utf-8") as f:
                _helplines_cache = json.load(f)
    return _helplines_cache


# ---------------------------------------------------------------------------
# Detection: keyword lists
# ---------------------------------------------------------------------------

# CRISIS_PHRASES — phrases that indicate the user may be in immediate
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
    "forced me to have sex",
    "forced sex",
    "he forced me",
    # trafficking / exploitation
    "being trafficked",
    "they won't let me leave",
    "they took my phone",
    "i need help to escape",
]

# DIAGNOSTIC_PHRASES — phrases that describe personal symptoms or
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
        "crisis"     — message contains crisis/self-harm/abuse signals
        "diagnostic" — message contains personal symptom descriptions
        None         — message appears to be an educational question

    Priority: crisis takes precedence over diagnostic. If both match,
    "crisis" is returned — the user gets crisis resources first.

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
# Response builders
# ---------------------------------------------------------------------------

def _format_helplines(category: str) -> str:
    """
    Formats the helplines for the given category ("crisis",
    "sexual_abuse", or "general_health") into a short string for
    inclusion in a WhatsApp message.

    Returns an empty string if no verified helplines are available yet,
    rather than listing placeholder numbers to real users.
    """
    data = _load_helplines()
    entries = data.get("national", {}).get(category, [])

    lines = []
    for entry in entries:
        number = entry.get("number", "")
        name = entry.get("name", "")
        # Only include numbers that are not placeholders. During the
        # pilot, PLACEHOLDER numbers must never reach a real user.
        if "PLACEHOLDER" in number or not number:
            continue
        lines.append(f"  {name}: {number}")

    return "\n".join(lines)


def _crisis_response() -> str:
    """
    Returns the scripted crisis response. Follows safe messaging
    guidelines: acknowledge warmth, validate feelings, provide
    resources, encourage action. Kept short for WhatsApp.
    """
    crisis_lines = _format_helplines("crisis")
    abuse_lines = _format_helplines("sexual_abuse")

    helpline_block = ""
    if crisis_lines:
        helpline_block += f"Crisis support:\n{crisis_lines}\n"
    if abuse_lines:
        helpline_block += f"\nSexual violence support:\n{abuse_lines}\n"

    if not helpline_block:
        helpline_block = (
            "  Please contact a trusted adult, health worker, "
            "or go to your nearest health facility or police station.\n"
        )

    return (
        "I hear you, and I want you to know you are not alone.\n\n"
        "What you are going through sounds very difficult, and you "
        "deserve real support from a person who can help — not just a "
        "chatbot.\n\n"
        "Please reach out to one of these resources right away:\n\n"
        f"{helpline_block}\n"
        "If you are in immediate danger, call 112 (Nigeria emergency).\n\n"
        "You are important, and help is available."
    )


def _diagnostic_response() -> str:
    """
    Returns the scripted diagnostic response. Encourages visiting a
    health facility rather than diagnosing through a chatbot.
    """
    health_lines = _format_helplines("general_health")

    helpline_block = ""
    if health_lines:
        helpline_block = f"Health helplines:\n{health_lines}\n\n"

    return (
        "It sounds like you may have a personal health concern that "
        "needs attention from a trained health worker.\n\n"
        "I can share general information about sexual and reproductive "
        "health, but I am not able to assess symptoms or give you a "
        "diagnosis — only a health professional can do that safely.\n\n"
        "Please visit your nearest:\n"
        "  - Primary Health Care (PHC) centre\n"
        "  - General hospital\n"
        "  - Family planning clinic\n\n"
        f"{helpline_block}"
        "Your health matters. Please don't wait if something feels wrong."
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


def get_escalation_response(message: str) -> str:
    """
    Returns the appropriate scripted response for an escalated message.
    Should only be called after should_escalate() has returned True.

    Returns the crisis response for crisis-type messages, and the
    diagnostic response for diagnostic-type messages.
    """
    escalation_type = _detect_type(message)

    if escalation_type == "crisis":
        return _crisis_response()
    elif escalation_type == "diagnostic":
        return _diagnostic_response()
    else:
        # Fallback — should not be reached if called correctly (after
        # should_escalate() returned True), but handled defensively.
        return _diagnostic_response()


# ---------------------------------------------------------------------------
# Quick manual test — runs only when this file is executed directly
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

    print("\n=== TEST 3: Crisis response ===\n")
    print(get_escalation_response("I want to kill myself"))

    print("\n=== TEST 4: Diagnostic response ===\n")
    print(get_escalation_response("I think I might be pregnant"))

    print("\n=== TEST 5: Helplines file loaded ===")
    data = _load_helplines()
    print(f"  Keys: {list(data.keys())}")
    print(f"  National categories: {list(data.get('national', {}).keys())}")
