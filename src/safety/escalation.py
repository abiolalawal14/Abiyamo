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

Detection strategy - Phase 2 (DistilBERT classifier, active):
    _detect_type() checks CRISIS_PHRASES / DIAGNOSTIC_PHRASES FIRST,
    unconditionally -- not only as a fallback. Only messages that match
    no keyword phrase fall through to the trained DistilBERT intent
    classifier (models/intent_classifier/). This changed after live
    testing showed the classifier's confidence on some topics (e.g.
    abortion) sits right at the 0.5 decision boundary regardless of
    context, giving inconsistent results for near-identical wording.
    Explicit phrases give deterministic behaviour for known
    personal-distress framings. The crisis label uses a lowered 0.35
    threshold (vs 0.5 for the others) when the classifier IS consulted --
    missing a crisis query is worse than a false positive, so crisis
    recall is favoured over precision. Thresholds and label names come
    from models/intent_classifier/classifier_config.json, not hardcoded
    here.

    CRISIS_PHRASES / DIAGNOSTIC_PHRASES (below) also serve as the
    fallback path when the trained model is not present locally (e.g.
    it was trained on Colab and never copied down) or fails to load.
    See _load_classifier() / _detect_type_keywords().

    Design decisions for the keyword lists (fallback path):
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

_detect_type() return contract (classifier and keyword fallback alike):
    The classifier is multi-label internally (a query can score above
    threshold on more than one class), but _detect_type() still returns
    a single str | None -- "crisis" > "diagnostic" > None priority --
    so should_escalate() and get_escalation_response() did not need to
    change at all when the classifier replaced keyword matching.

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

from thefuzz import fuzz
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch

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
    "i need help urgently",
    "i need medical help",
    "it hurts when i urinate",
    "it hurts inside",
    # abortion in a personal-distress context -- NOT general topic
    # questions like "what is abortion?" or "what about abortion?",
    # which must stay educational and go through the RAG pipeline.
    "i need an abortion",
    "i had an abortion",
    # facility-seeking -- a personal request for somewhere to go, not a
    # general topic question. Routes to escalation so the user gets
    # real facility names (get_helplines_for_state()) instead of a
    # RAG-generated answer with no facility data to draw from.
    "which phc",
    "which primary health",
    "nearest clinic",
    "nearest hospital",
    "nearest health centre",
    "where can i go",
    "where should i go",
    "which hospital",
    "find a clinic",
    "health facility near",
    # personal medical follow-up -- distinct from a generic "follow up
    # on <topic>" request for information (see SAFE_EDUCATIONAL_TERMS
    # below), these name an ongoing personal relationship with a real
    # health worker/facility about the user's own condition.
    "following up with my doctor",
    "follow up with my doctor",
    "following up with the doctor",
    "following up with a doctor",
    "follow up with the hospital",
]


# Terms that must always resolve to educational (None) UNLESS a
# CRISIS_PHRASES/DIAGNOSTIC_PHRASES phrase also matches above (e.g. "I
# think I have an addiction to masturbation and I want to kill myself"
# still escalates on the crisis phrase, checked first). Added after
# live testing with a translated Yoruba query -- see translator.py's
# _KNOWN_TRANSLATION_OVERRIDES for the fuller story, but independent of
# that: a bare mention of "masturbation" is a normal, common
# educational SRH topic on its own and should never need the
# classifier's judgment call.
#
# "follow up"/"follow-up": live testing showed "I need follow up on
# STI" being misclassified as diagnostic by the classifier (0.531,
# right at the usual ~0.5 boundary -- see abortion/masturbation notes
# above for the recurring pattern). "Follow up on <topic>" is commonly
# just "give me more information", not a personal health concern.
# Genuinely personal follow-up ("following up with my doctor about my
# discharge") is caught by the more specific DIAGNOSTIC_PHRASES entries
# above FIRST, so those still escalate -- this bare-term fallback only
# fires when no such phrase already matched.
SAFE_EDUCATIONAL_TERMS = [
    "masturbation",
    "follow up",
    "follow-up",
]

# ---------------------------------------------------------------------------
# Short-message classifier gate
# ---------------------------------------------------------------------------
# Live testing showed short, ambiguous messages with no keyword-phrase
# match (e.g. "What are the symptoms of STIs?", 6 words) reaching the
# classifier and tripping its DIAGNOSTIC threshold (0.528, just over
# the normal 0.5 bar) purely from being short and generic -- the same
# ~0.5-boundary unreliability documented for CRISIS_PHRASES above, just
# on the classifier's diagnostic label instead of crisis.
#
# Fix: for a message under _SHORT_MESSAGE_WORD_THRESHOLD words that
# doesn't ALSO contain an explicit danger signal, the classifier is
# only trusted for its CRISIS probability, at a RAISED bar (0.6, not
# the normal 0.35/0.5) -- diagnostic-via-classifier is not considered
# for such messages at all.
#
# IMPORTANT SAFETY SCOPE -- this gate applies ONLY to the classifier
# fallback step, never to CRISIS_PHRASES or DIAGNOSTIC_PHRASES keyword
# matching above, which always run in full regardless of message
# length. An earlier draft of this gate would have also skipped keyword
# matching for short messages, which breaks CLAUDE.md's explicit design
# principle ("crisis recall favoured over precision, missing a crisis
# query is worse than a false positive") -- e.g. "I want to kill
# myself" (5 words) contains none of the override keywords below, so it
# would have been downgraded to needing classifier crisis probability
# > 0.6 instead of a deterministic "kill myself" phrase match. Keyword
# phrases are checked unconditionally first specifically so this can
# never happen; this gate only narrows what happens AFTER both keyword
# lists have already found no match.
_SHORT_MESSAGE_WORD_THRESHOLD = 10
_SHORT_MESSAGE_CRISIS_PROBABILITY_THRESHOLD = 0.6

# Presence of any of these allows the classifier to be trusted normally
# (both crisis and diagnostic, at the usual thresholds) even for a
# short message -- these signal enough real danger/urgency that the
# extra caution above isn't warranted. Deliberately not exhaustive
# safety-critical wording (that's what CRISIS_PHRASES/DIAGNOSTIC_PHRASES
# above already cover, unconditionally) -- this is only about whether
# the classifier fallback gets the normal or the stricter treatment.
_EXPLICIT_CRISIS_OVERRIDE_KEYWORDS = [
    "rape", "assault", "forced", "abuse", "pregnant and scared", "help me urgently",
]

CLASSIFIER_DIR = "models/intent_classifier"
_clf_config = None
_clf_tokenizer = None
_clf_model = None
_clf_load_failed = False  # set once loading fails so we don't retry every message


def _load_classifier() -> bool:
    """
    Lazily loads the trained DistilBERT classifier. Returns True if it is
    ready to use, False if it should not be used -- missing directory,
    missing classifier_config.json, or corrupt artifacts all fall back
    to keyword detection rather than crashing should_escalate().
    """
    global _clf_config, _clf_tokenizer, _clf_model, _clf_load_failed
    if _clf_model is not None:
        return True
    if _clf_load_failed:
        return False

    config_path = Path(CLASSIFIER_DIR) / "classifier_config.json"
    if not config_path.exists():
        _clf_load_failed = True
        return False

    try:
        with open(config_path) as f:
            _clf_config = json.load(f)
        _clf_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
        _clf_model = DistilBertForSequenceClassification.from_pretrained(CLASSIFIER_DIR)
        _clf_model.eval()
        return True
    except Exception as e:
        print(
            f"[escalation] Warning: failed to load DistilBERT classifier "
            f"({e}). Falling back to keyword-based detection."
        )
        _clf_model = None
        _clf_load_failed = True
        return False


def _detect_type_keywords(message: str) -> str | None:
    """
    Original Phase 3 keyword matching. Used as the fallback when the
    trained DistilBERT classifier is not available (see _load_classifier).
    """
    lower = message.lower().strip()

    for phrase in CRISIS_PHRASES:
        if phrase in lower:
            return "crisis"

    for phrase in DIAGNOSTIC_PHRASES:
        if phrase in lower:
            return "diagnostic"

    return None


def _classifier_probs(message: str) -> dict[str, float]:
    """
    Runs the trained DistilBERT classifier and returns raw per-label
    probabilities (before any decision threshold is applied). Shared by
    both branches of _detect_type()'s classifier fallback below so the
    tokenize/forward-pass code isn't duplicated.
    """
    inputs = _clf_tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=_clf_config["max_length"],
        padding=True,
    )
    with torch.no_grad():
        logits = _clf_model(**inputs).logits[0]
    probs = torch.sigmoid(logits).numpy()
    labels = _clf_config["labels"]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


def _detect_type(message: str) -> str | None:
    """
    Returns the escalation type for a message, or None if no escalation
    is needed.

    Returns:
        "crisis"     -- message scores >= crisis_threshold on "crisis"
        "diagnostic" -- message scores >= 0.5 on "diagnostic"
        None         -- message appears to be an educational question

    Priority: crisis takes precedence over diagnostic. If both match,
    "crisis" is returned -- the user gets crisis resources first. This
    matches the previous keyword-based contract exactly, so
    should_escalate() and get_escalation_response() need no changes.

    Keyword phrases (CRISIS_PHRASES / DIAGNOSTIC_PHRASES) are checked
    FIRST, unconditionally, for EVERY message regardless of length --
    not just as a fallback when the classifier is unavailable. Live
    testing showed the trained classifier's confidence on abortion-
    related text sits right at the 0.5 decision boundary (~0.46-0.52)
    regardless of context, meaning near-identical wording ("what about
    abortion?" vs "I need an abortion help me") could flip either way.
    Explicit phrases give deterministic behaviour for known personal-
    distress framings while general topic questions (no phrase match)
    still fall through to the classifier/RAG pipeline.

    After the keyword check, SAFE_EDUCATIONAL_TERMS is checked next --
    also unconditionally -- for the same reason: a translated Yoruba
    query ("Kini Atosi" -> mistranslated to "What is the Root") was
    getting misclassified as diagnostic by the classifier on nonsense
    input. "masturbation" is checked directly rather than trusting the
    classifier's judgment on it.

    Only falls through to the classifier when NEITHER of the above
    matches. At that point, a SHORT message (< _SHORT_MESSAGE_WORD_
    THRESHOLD words) with no explicit danger signal present
    (_EXPLICIT_CRISIS_OVERRIDE_KEYWORDS) gets extra caution: the
    classifier is trusted only for a crisis call, at a raised bar
    (_SHORT_MESSAGE_CRISIS_PROBABILITY_THRESHOLD), never for diagnostic
    -- see the module-level comment above SAFE_EDUCATIONAL_TERMS for why
    (short generic text was tripping the classifier's diagnostic
    threshold). This gate never touches the keyword-phrase step above,
    which is what keeps it safe -- see that comment block for the
    safety reasoning in full.
    """
    keyword_result = _detect_type_keywords(message)
    if keyword_result is not None:
        return keyword_result

    lower = message.lower()
    if any(term in lower for term in SAFE_EDUCATIONAL_TERMS):
        return None

    if not _load_classifier():
        return None

    word_count = len(message.split())
    has_explicit_danger_signal = any(kw in lower for kw in _EXPLICIT_CRISIS_OVERRIDE_KEYWORDS)

    if word_count < _SHORT_MESSAGE_WORD_THRESHOLD and not has_explicit_danger_signal:
        crisis_prob = _classifier_probs(message).get("crisis", 0.0)
        return "crisis" if crisis_prob > _SHORT_MESSAGE_CRISIS_PROBABILITY_THRESHOLD else None

    probs = _classifier_probs(message)
    crisis_threshold = _clf_config["crisis_threshold"]
    detected = set()
    for label, prob in probs.items():
        thresh = crisis_threshold if label == "crisis" else 0.5
        if prob >= thresh:
            detected.add(label)

    if "crisis" in detected:
        return "crisis"
    if "diagnostic" in detected:
        return "diagnostic"
    return None


# ---------------------------------------------------------------------------
# State-aware facility lookup
# ---------------------------------------------------------------------------

# Variants a user or the session layer might pass for the FCT.
# The facility dataset normalises FCT to "Fct" after title-case.
_FCT_VARIANTS = {"fct", "abuja", "fct abuja", "federal capital territory", "abuja fct"}

# Below this fuzz.partial_ratio score, an LGA is not considered a match.
# 70 handles common real-world mismatches between what a user types
# during onboarding and the exact string in the facility dataset:
# "Ibadan South West" vs "Ibadan S/W", "Bwari" vs "Bwari LGA", "Amac"
# vs "AMAC".
#
# KNOWN LIMITATION: partial_ratio finds the best-matching substring
# window, so two DIFFERENT LGAs that share a long common prefix can
# also score above 70 -- e.g. "Ibadan South West" vs "Ibadan North"
# scores 83. Oyo state alone has five "Ibadan ..." LGAs, so a user in
# one may get facilities prioritised from a neighbouring one instead.
# Tried fuzz.ratio/token_sort_ratio as alternatives -- neither fully
# solves this (e.g. "Ibadan South West" vs "Ibadan South East" still
# scores 94 on plain ratio); it's an inherent limit of fuzzy-matching
# near-identical short strings, not a one-line fix. Accepted here
# because this only affects which facilities are LISTED FIRST, never
# which are shown at all -- get_helplines_for_state() falls back to the
# rest of the state's facilities either way (see docstring below).
_FUZZY_LGA_MATCH_THRESHOLD = 70


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


def get_helplines_for_state(state: str | None, lga: str | None = None) -> list:
    """
    Returns the list of facility dicts for the given Nigerian state from
    data/helplines.json. Returns an empty list when the state is unknown,
    not provided, or the file has not been populated yet.

    Case-insensitive matching so callers can pass "borno" or "Borno".

    Parameters:
        lga : Optional LGA name captured during onboarding. When
              provided, facilities are ranked by fuzz.partial_ratio
              against this value (see _FUZZY_LGA_MATCH_THRESHOLD) and
              those scoring >= threshold are moved to the front, highest
              score first -- callers slice to the first few results
              (see _crisis_response/_diagnostic_response), so this
              surfaces the closest-matching-LGA facilities first
              without ever hiding the rest of the state's facilities if
              nothing scores high enough (graceful fallback, not a
              filter). Exact matches always score 100, so this is a
              strict superset of the old exact-match behaviour -- fuzzy
              matching was added because user-typed LGA names
              (onboarding free text) rarely match the facility
              dataset's exact spelling ("Ibadan South West" vs the
              dataset's "Ibadan S/W", etc).
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
    facilities = []
    for key, val in states.items():
        if key.lower() == normalized_lower:
            facilities = val.get("facilities", [])
            break

    if not facilities or not lga:
        return facilities

    lga_lower = lga.strip().lower()
    scored = [
        (fuzz.partial_ratio(lga_lower, f.get("lga", "").lower()), f)
        for f in facilities
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    matching = [f for score, f in scored if score >= _FUZZY_LGA_MATCH_THRESHOLD]
    other = [f for score, f in scored if score < _FUZZY_LGA_MATCH_THRESHOLD]
    return matching + other


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def is_placeholder(entry: dict) -> bool:
    """
    Returns True if a helpline entry must NOT be shown to a real user --
    either it has no number at all, its number is still the literal
    "PLACEHOLDER_NOT_VERIFIED" sentinel written by
    scripts/import_facilities.py, or it has not been marked
    verified: true. This is the gate CLAUDE.md has always described as
    existing; it previously did not, and crisis_fallback in
    helplines.json was written but never actually read anywhere in
    src/ -- this function and _get_verified_crisis_hotlines() below are
    what closes that gap. A number only reaches a user once someone has
    personally called it, confirmed it is live, and flipped "verified"
    to true in helplines.json.
    """
    number = (entry.get("number") or "").strip()
    if not number or number.upper() == "PLACEHOLDER_NOT_VERIFIED":
        return True
    return entry.get("verified") is not True


def _get_verified_crisis_hotlines() -> list[dict]:
    """
    Returns the crisis_fallback entries from helplines.json that have
    passed is_placeholder() -- i.e. genuinely phone-verified numbers --
    excluding the national emergency line (112), which is already
    always included in the response text separately below.
    """
    data = _load_helplines()
    entries = data.get("crisis_fallback", [])
    return [
        e for e in entries
        if not is_placeholder(e) and e.get("name") != "Nigeria Emergency Services"
    ]


def _format_verified_hotlines_block(hotlines: list[dict]) -> str:
    """Renders verified crisis_fallback entries as extra reference lines, or "" if none."""
    if not hotlines:
        return ""
    lines = "\n".join(f"- {h['name']}: {h['number']}" for h in hotlines)
    return f"\n\nYou can also reach these support lines:\n{lines}"


def _crisis_response(state: str | None = None, lga: str | None = None) -> str:
    """
    Scripted crisis response. When a state is known, names real local
    facilities so the user has a concrete place to go. Always adds 112,
    plus any other crisis_fallback hotlines that have cleared
    is_placeholder() (see above) -- currently none, until each one is
    personally called and confirmed.

    Follows safe messaging: acknowledge warmth, validate, give resources,
    encourage action. Short by design -- WhatsApp users disengage with
    long messages.
    """
    facilities = get_helplines_for_state(state, lga)
    extra_hotlines = _format_verified_hotlines_block(_get_verified_crisis_hotlines())

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
            f"{extra_hotlines}"
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
        "You can also call Nigeria's emergency line: 112 (available 24/7)"
        f"{extra_hotlines}\n\n"
        "You are important, and help is available."
    )


def _diagnostic_response(state: str | None = None, lga: str | None = None) -> str:
    """
    Scripted diagnostic response. Encourages visiting a named health
    facility rather than diagnosing through a chatbot. Falls back to
    generic PHC guidance when no state is known.
    """
    facilities = get_helplines_for_state(state, lga)

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


def get_escalation_response(message: str, state: str | None = None, lga: str | None = None) -> str:
    """
    Returns the appropriate scripted response for an escalated message.
    Should only be called after should_escalate() has returned True.

    Parameters:
        message : the user's message text (used to detect escalation type),
                  OR one of "crisis" / "diagnostic" as a test shorthand.
        state   : optional Nigerian state name (e.g. "Borno", "Lagos").
                  When provided, the response includes named local facilities.
                  When None, falls back to generic national guidance.
        lga     : optional LGA name captured during onboarding. When
                  provided alongside state, facilities from that LGA
                  are prioritised first (see get_helplines_for_state()).

    Backward compatible: callers that pass only message (no state/lga)
    continue to work -- they receive the national-fallback response.
    """
    # Allow passing the type directly for testing ("crisis", "diagnostic")
    # without needing a real message that triggers keyword detection.
    if message in ("crisis", "diagnostic"):
        escalation_type = message
    else:
        escalation_type = _detect_type(message)

    if escalation_type == "crisis":
        return _crisis_response(state, lga)
    else:
        # Covers both "diagnostic" and the defensive fallback for None
        return _diagnostic_response(state, lga)


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
        # Abortion: topic questions must stay educational; personal
        # distress framing must still escalate (found in live testing).
        ("What is abortion?",                                  None),
        ("What about abortion?",                                None),
        ("I need an abortion help me",                         "diagnostic"),
        ("I had an abortion and I am bleeding",                "diagnostic"),
        # Facility-seeking: personal request for a nearby facility must
        # escalate (real facility names) rather than go through RAG.
        ("Which primary health center can I visit",            "diagnostic"),
        # Follow-up: generic "give me more info" must stay educational;
        # a genuinely personal follow-up must still escalate.
        ("I need follow up on STI",                             None),
        ("I have been following up with my doctor about my discharge", "diagnostic"),
        # New specific diagnostic phrases (replacing the overly-generic
        # "i need"/"i am hurt"/standalone "hurts" that were requested
        # for removal but never actually existed in this list).
        ("I need help urgently",                                "diagnostic"),
        ("I need medical help",                                 "diagnostic"),
        ("It hurts when I urinate",                             "diagnostic"),
        ("It hurts inside",                                     "diagnostic"),
        # Short-message classifier gate: "What are the symptoms of
        # STIs?" (test case #3 above) is this same scenario in
        # practice -- a short, ambiguous message with no keyword-phrase
        # match and no explicit danger signal, where the classifier's
        # diagnostic score alone (0.528) used to cross 0.5. That test
        # case was failing on every run before this fix; it's the
        # clearest evidence the gate solves the real reported problem.
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

    print("\n=== TEST 7: Fuzzy LGA matching ===")
    print("  Fct/Bwari (exact match present in data):")
    fct_bwari = get_helplines_for_state("Fct", "Bwari")
    print(f"    Top result: {fct_bwari[0]['name']} - {fct_bwari[0]['lga']}  (expected LGA: Bwari)")

    print("  Oyo/Ibadan South West (all 33 of Oyo's LGAs are now represented):")
    oyo_ibadan = get_helplines_for_state("Oyo", "Ibadan South West")
    print(f"    Top result: {oyo_ibadan[0]['name']} - {oyo_ibadan[0]['lga']}  (expected LGA: Ibadan South West)")
    print(
        "    NOTE: MAX_FACILITIES_PER_STATE was raised from 5 to 50 in "
        "scripts/import_facilities.py so every state's LGAs are covered, not "
        "just an alphabetical first few. A real Ibadan South West entry now "
        "exists and scores 100 on fuzz.partial_ratio, so it is always ranked "
        "ahead of any coincidental false-positive match like 'Atiba'."
    )
