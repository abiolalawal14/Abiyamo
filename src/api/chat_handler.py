"""
chat_handler.py

Purpose:
    Central logic module that processes a user's question through the
    full RAG pipeline and returns a structured result. This is the
    single place where all routing decisions are made — escalation,
    translation, retrieval, generation. Both the HTTP API (main.py)
    and the WhatsApp layer (webhook.py) call handle_message() here
    rather than each wiring up the pipeline themselves.

Why centralise here rather than in each entry point:
    Both the /chat HTTP endpoint and the WhatsApp webhook do the same
    thing: take a user's message, run it through the pipeline, return
    an answer. Keeping that logic in one place means there is exactly
    one function to update when translator.py and escalation.py are
    finished — not two.

Current routing (Phase 3 state):
    All messages → RAG pipeline (retrieval + generation)

    Two TODOs mark where escalation and translation plug in once their
    modules are built. The structure is already correct — only the
    stub code needs replacing.

Returns from handle_message():
    {
        "answer"      : str   — the response text to send to the user
        "language"    : str   — BCP-47 language code of the response
                                (mirrors input for now; will differ once
                                translation is active)
        "escalated"   : bool  — True if query was routed to escalation
                                rather than the RAG pipeline
        "chunks_used" : int   — number of knowledge base chunks that
                                were retrieved (0 if escalated or empty
                                knowledge base; useful for evaluation)
    }
"""

import hashlib
import json
import re
from pathlib import Path

from src.rag_pipeline.retriever import retrieve_relevant_chunks
from src.rag_pipeline.generator import answer_from_chunks
from src.rag_pipeline.translator import from_english, to_english
from src.safety.escalation import should_escalate, get_escalation_response

# Fallback reply sent when the pipeline raises an unexpected exception.
# Kept here (not in generator.py) because chat_handler is the layer
# that decides what the user sees — generator.py's fallback covers
# API failures, this one covers anything else that can go wrong.
_PIPELINE_ERROR_REPLY = (
    "I'm sorry, something went wrong. Please try again, or speak "
    "to a health worker at a nearby health facility."
)

# ---------------------------------------------------------------------------
# Onboarding — first-contact language selection
# ---------------------------------------------------------------------------

# Session file lives in data/ alongside pilot_logs.jsonl and helplines.json.
_SESSIONS_FILE = Path("data/user_sessions.json")

# Maps the WhatsApp numeric reply to the plain-name keys used throughout
# translator.py's SUPPORTED_LANGUAGES dict, so a saved session language
# is directly usable wherever that dict is consulted.
_LANGUAGE_CHOICES = {
    "1": "english",
    "2": "hausa",
    "3": "yoruba",
    "4": "igbo",
}

# translator.py's to_english()/from_english() take BCP-47 codes
# ("en"/"ha"/"yo"/"ig") via LANGUAGE_CODES, NOT the plain names stored
# in user_sessions.json (SUPPORTED_LANGUAGES uses plain names as keys).
# Passing "yoruba" straight to source_lang/target_lang silently no-ops
# (LANGUAGE_CODES.get("yoruba") is None -> falls into the "unsupported
# code" branch) -- this mapping is what makes a stored session language
# actually drive translation.
_PLAIN_TO_BCP47 = {
    "english": "en",
    "hausa": "ha",
    "yoruba": "yo",
    "igbo": "ig",
}

_WELCOME_MESSAGE = (
    "Welcome to Abiyamo — your safe space for sexual "
    "and reproductive health information. 🌿\n\n"
    "Before we begin, what language do you prefer?\n"
    "Reply with:\n"
    "1 for English\n"
    "2 for Hausa (Hausa)\n"
    "3 for Yoruba (Yoruba)\n"
    "4 for Igbo (Igbo)"
)

_STATE_PROMPT = (
    "One more question — which state are you in? "
    "This helps me show you nearby health facilities "
    "if you ever need support.\n\n"
    "Reply with your state name e.g: Lagos, Kano, Abuja, "
    "Rivers, Oyo, Borno etc."
)

_ONBOARDED_CONFIRMATION = (
    "Thank you! You are all set. Ask me anything about sexual and "
    "reproductive health. 🌿"
)

# ---------------------------------------------------------------------------
# Session reset — lets a user escape back to onboarding at any time
# ---------------------------------------------------------------------------

_RESET_TRIGGER_PHRASES = [
    "reset", "start over", "start again", "restart",
    "begin again", "new session", "/reset", "/start",
]

_RESET_CONFIRMATION_MESSAGE = (
    "Your session has been reset. 🌿\n\n"
    "Welcome to Abiyamo — your safe space for sexual "
    "and reproductive health information.\n\n"
    "What language do you prefer?\n"
    "Reply with:\n"
    "1 for English\n"
    "2 for Hausa\n"
    "3 for Yoruba\n"
    "4 for Igbo"
)

_RESET_ERROR_MESSAGE = (
    "Sorry, I could not reset your session right now. "
    "Please try again in a moment."
)

# ---------------------------------------------------------------------------
# Language switching — menu + direct "speak X" phrases for any user,
# onboarded or not, without touching their saved state preference
# ---------------------------------------------------------------------------

_LANGUAGE_MENU_TRIGGER_PHRASES = [
    "language", "change language", "language options", "switch language",
]

_LANGUAGE_MENU_MESSAGE = (
    "Which language would you prefer?\n"
    "Reply with:\n"
    "1 for English\n"
    "2 for Hausa\n"
    "3 for Yoruba\n"
    "4 for Igbo"
)

# Phrase -> plain-name language, same keys _LANGUAGE_CHOICES already uses.
_DIRECT_SWITCH_TRIGGERS = {
    "change to english": "english", "speak english": "english", "english please": "english",
    "change to hausa": "hausa", "speak hausa": "hausa", "hausa please": "hausa",
    "change to yoruba": "yoruba", "speak yoruba": "yoruba", "yoruba please": "yoruba",
    "change to igbo": "igbo", "speak igbo": "igbo", "igbo please": "igbo",
}

# NOTE: not fully confident these read as natural, idiomatic phrasing in
# each language (especially Hausa/Igbo) -- flagged for native-speaker
# review before pilot deployment, per the build instructions.
_LANGUAGE_SWITCH_CONFIRMATIONS = {
    "english": "Language changed to English. 🌿",
    "hausa": "An canza harshe zuwa Hausa. 🌿",
    "yoruba": "Ede ti yipada si Yoruba. 🌿",
    "igbo": "Asụsụ agbanwela na Igbo. 🌿",
}


def _normalize_for_trigger_match(text: str) -> str:
    """
    Lowercases and strips punctuation so a trigger phrase matches
    regardless of capitalisation or punctuation -- "/reset", "Reset!",
    and "reset" must all be recognised the same way. Substring matching
    (not exact-match) is used throughout this module's trigger
    detection, consistent with how CRISIS_PHRASES/DIAGNOSTIC_PHRASES
    are matched in escalation.py.
    """
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


# Normalized once at import time rather than on every incoming message.
_RESET_TRIGGERS_NORMALIZED = [_normalize_for_trigger_match(p) for p in _RESET_TRIGGER_PHRASES]
_LANGUAGE_MENU_TRIGGERS_NORMALIZED = [
    _normalize_for_trigger_match(p) for p in _LANGUAGE_MENU_TRIGGER_PHRASES
]
_DIRECT_SWITCH_TRIGGERS_NORMALIZED = {
    _normalize_for_trigger_match(phrase): lang for phrase, lang in _DIRECT_SWITCH_TRIGGERS.items()
}


def _is_reset_request(message: str) -> bool:
    normalized = _normalize_for_trigger_match(message)
    return any(trigger in normalized for trigger in _RESET_TRIGGERS_NORMALIZED)


def _is_language_menu_request(message: str) -> bool:
    normalized = _normalize_for_trigger_match(message)
    return any(trigger in normalized for trigger in _LANGUAGE_MENU_TRIGGERS_NORMALIZED)


def _detect_direct_language_switch(message: str) -> str | None:
    """Returns the plain-name language a message directly requested, or None."""
    normalized = _normalize_for_trigger_match(message)
    for phrase, lang in _DIRECT_SWITCH_TRIGGERS_NORMALIZED.items():
        if phrase in normalized:
            return lang
    return None


def _hash_user_id(raw_id: str) -> str:
    """
    Same SHA-256 pattern as evaluation/logger.py's _anonymize_user() —
    a one-way, deterministic hash so a phone number is never stored in
    plain text, while the same number always maps to the same session.
    """
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]


def _load_sessions() -> dict:
    """
    Loads data/user_sessions.json. Returns an empty dict if the file
    doesn't exist yet (first run) or is corrupt, so onboarding always
    degrades to "treat as new user" rather than crashing the pipeline.
    """
    if not _SESSIONS_FILE.exists():
        return {}
    try:
        with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[chat_handler] Warning: could not read {_SESSIONS_FILE}: {e}")
        return {}


def _save_sessions(sessions: dict) -> bool:
    """
    Writes the full sessions dict back to data/user_sessions.json.
    Returns True on success, False on failure. Never raises — a
    session-save failure must not crash the user's reply. The boolean
    return (added for the reset feature) lets a caller distinguish
    "saved fine" from "silently didn't persist" instead of always
    assuming success, per the reset feature's explicit requirement to
    tell the user when a reset couldn't actually be completed.
    """
    _SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2)
        return True
    except OSError as e:
        print(f"[chat_handler] Warning: could not write {_SESSIONS_FILE}: {e}")
        return False


def _onboarding_reply(answer: str, language: str) -> dict:
    """Shapes an onboarding-step answer into a handle_message()-style result."""
    return {
        "answer": answer,
        "language": language,
        "escalated": False,
        "chunks_used": 0,
    }


def _handle_onboarding(
    phone_hash: str, sessions: dict, session: dict | None, message: str, language: str
) -> dict | None:
    """
    Runs the onboarding state machine for a WhatsApp user: language
    selection, then state capture (for facility lookups in
    escalation.py), then done. Mutates `sessions` in place and persists
    it via _save_sessions() whenever onboarding advances a step.

    Three sub-states, distinguished without an extra "stage" field:
        session is None                        -> brand new user
        onboarded=False, language is None       -> awaiting language reply
        onboarded=False, language is set        -> awaiting state reply
        onboarded=True                          -> done, normal pipeline runs

    Returns a complete handle_message()-shaped result dict if onboarding
    should intercept this message, or None if the user is already
    onboarded and the normal pipeline should run instead.
    """
    if session is None:
        # Brand-new user — record a pending session so their next
        # message is treated as a language reply, not another "new
        # user" welcome, then send the welcome message.
        sessions[phone_hash] = {"language": None, "state": None, "onboarded": False}
        _save_sessions(sessions)
        return _onboarding_reply(_WELCOME_MESSAGE, language)

    if session.get("onboarded"):
        # Already onboarded — let the normal pipeline handle this message.
        return None

    if session.get("language") is None:
        # Awaiting language reply.
        choice = message.strip()
        if choice in _LANGUAGE_CHOICES:
            session["language"] = _LANGUAGE_CHOICES[choice]
            sessions[phone_hash] = session
            _save_sessions(sessions)
            return _onboarding_reply(_STATE_PROMPT, language)
        # Didn't reply with a valid option — re-send the prompt
        # rather than guessing.
        return _onboarding_reply(_WELCOME_MESSAGE, language)

    # Language already chosen — awaiting state reply.
    state_name = message.strip()
    if state_name:
        session["state"] = state_name
        session["onboarded"] = True
        sessions[phone_hash] = session
        _save_sessions(sessions)
        return _onboarding_reply(_ONBOARDED_CONFIRMATION, language)
    # Empty reply — re-send the state prompt rather than guessing.
    return _onboarding_reply(_STATE_PROMPT, language)


def _handle_reset(phone_hash: str, sessions: dict, language: str) -> dict:
    """
    Deletes the user's session entry entirely, clearing language AND
    state together (there is no partial reset), then immediately
    re-primes a pending onboarding entry — same shape _handle_onboarding
    creates for a brand-new user — so the very next message (expected
    to be a 1-4 language reply, since the reset message already asks
    for it) is captured correctly instead of producing a second,
    redundant welcome message.

    A missing entry (user was never onboarded) is not an error —
    dict.pop(..., None) is a no-op — so this always reaches the same
    "send welcome" outcome without needing to special-case it.
    """
    sessions.pop(phone_hash, None)
    sessions[phone_hash] = {"language": None, "state": None, "onboarded": False}
    if _save_sessions(sessions):
        return _onboarding_reply(_RESET_CONFIRMATION_MESSAGE, language)
    # _save_sessions() already logs the underlying OSError — this is
    # purely about giving the user an honest answer instead of a false
    # "your session was reset" when the write didn't actually happen.
    return _onboarding_reply(_RESET_ERROR_MESSAGE, language)


def _handle_language_switch(
    phone_hash: str, sessions: dict, session: dict | None, message: str, language: str
) -> dict | None:
    """
    Handles the language menu trigger, a direct "speak X" switch, or a
    pending numeric reply after the menu was shown. Returns a result
    dict if this message was consumed by the language-switch feature,
    or None if it wasn't (caller proceeds to the onboarding check).

    State is deliberately left untouched here — only "language" and the
    "pending_language_change" marker are ever written — matching the
    build instructions: switching language must not reset state.
    """
    if session is None:
        # No session yet at all -- a brand-new user's very first
        # message shouldn't be half-captured by a language command; let
        # onboarding create the session and show the standard welcome
        # first, same as it would for any other first message.
        return None

    direct_lang = _detect_direct_language_switch(message)
    if direct_lang is not None:
        session["language"] = direct_lang
        sessions[phone_hash] = session
        _save_sessions(sessions)
        return _onboarding_reply(
            _LANGUAGE_SWITCH_CONFIRMATIONS[direct_lang],
            _PLAIN_TO_BCP47[direct_lang],
        )

    if _is_language_menu_request(message):
        session["pending_language_change"] = True
        sessions[phone_hash] = session
        _save_sessions(sessions)
        return _onboarding_reply(_LANGUAGE_MENU_MESSAGE, language)

    choice = message.strip()
    if choice in _LANGUAGE_CHOICES:
        # A bare 1-4 reply only belongs to this feature once the menu
        # was explicitly requested, OR for a user who is fully done
        # with onboarding (language AND state already captured) --
        # otherwise it's the onboarding flow's own language-selection
        # digit and must fall through to _handle_onboarding untouched.
        fully_onboarded = (
            session.get("onboarded") is True
            and session.get("language") is not None
            and session.get("state") is not None
        )
        if session.get("pending_language_change") or fully_onboarded:
            new_lang = _LANGUAGE_CHOICES[choice]
            session["language"] = new_lang
            session["pending_language_change"] = False
            sessions[phone_hash] = session
            _save_sessions(sessions)
            return _onboarding_reply(
                _LANGUAGE_SWITCH_CONFIRMATIONS[new_lang],
                _PLAIN_TO_BCP47[new_lang],
            )
        return None

    return None


def handle_message(message: str, language: str = "en", user_id: str | None = None) -> dict:
    """
    Processes a single user message through the full pipeline.

    Parameters:
        message  : The user's question as plain text. If the user wrote
                   in Hausa, Yoruba, or Igbo, the caller is responsible
                   for passing the already-translated English text here
                   once translator.py is built. For now all messages
                   are treated as English.
        language : BCP-47 language code of the user's message
                   (e.g. "en", "ha", "yo", "ig"). Stored in the result
                   so the caller knows which language to translate back
                   to. Defaults to "en".
        user_id  : The user's WhatsApp phone number (raw, e.g.
                   "whatsapp:+2348012345678"), used to run the
                   first-contact onboarding flow (language selection),
                   session reset ("reset"/"/start"/etc.), and language
                   switching ("language", "speak yoruba", etc.) via
                   data/user_sessions.json. Defaults to None, which
                   skips all of that — the direct /chat HTTP endpoint
                   has no phone number to key a session off, so it
                   always goes straight to the normal pipeline.

    Returns:
        A dict with keys: answer, language, escalated, chunks_used.
        See module docstring for full description.
    """
    # Reset, then language-switch, then onboarding — all run before
    # anything else, including the empty-message check below and
    # translation/classification further down. Reset must come first so
    # it always works even if a user is stuck mid-onboarding or their
    # session is in some confusing state; language-switch comes next so
    # it works for onboarded AND mid-onboarding users alike, without
    # ever touching whether they're onboarded or what state they saved.
    session_state = None
    if user_id is not None:
        phone_hash = _hash_user_id(user_id)
        sessions = _load_sessions()
        session = sessions.get(phone_hash)

        if _is_reset_request(message):
            return _handle_reset(phone_hash, sessions, language)

        language_switch_result = _handle_language_switch(phone_hash, sessions, session, message, language)
        if language_switch_result is not None:
            return language_switch_result

        onboarding_result = _handle_onboarding(phone_hash, sessions, session, message, language)
        if onboarding_result is not None:
            return onboarding_result

        # Reaching here means the session exists and is onboarded (the
        # branches above return early otherwise) — use the language and
        # state the user chose during onboarding. webhook.py has no
        # BCP-47 code to pass in; user_sessions.json is the only place
        # that knows what a returning WhatsApp user actually selected.
        session = sessions[phone_hash]
        plain_lang = session.get("language")
        if plain_lang in _PLAIN_TO_BCP47:
            language = _PLAIN_TO_BCP47[plain_lang]
        session_state = session.get("state")

    if not message or not message.strip():
        return {
            "answer": (
                "Hi! I'm Abiyamo. Send me a question about sexual "
                "and reproductive health and I'll do my best to help."
            ),
            "language": language,
            "escalated": False,
            "chunks_used": 0,
        }

    # Translate to English BEFORE classification or retrieval. Escalation
    # detection (keyword phrases + the DistilBERT classifier) is
    # English-only — running it on untranslated Hausa/Yoruba/Igbo text
    # means a real crisis message would never be recognised as one. This
    # was a safety-critical gap: translation used to happen only for the
    # RAG path, after the escalation check had already run on raw text.
    english_message = to_english(message, source_lang=language)

    # Escalation check — MUST run before any retrieval or generation.
    # Diagnostic and crisis queries return scripted text + helplines and
    # never reach the LLM. This is a hard safety rule, not a soft preference.
    if should_escalate(english_message):
        answer = get_escalation_response(english_message, state=session_state)
        if language != "en":
            answer = from_english(answer, target_lang=language)
        return {
            "answer":       answer,
            "language":     language,
            "escalated":    True,
            "chunks_used":  0,
        }

    try:
        chunks = retrieve_relevant_chunks(english_message)
        answer = answer_from_chunks(english_message, chunks)
        if language != "en":
            answer = from_english(answer, target_lang=language)
    except Exception as e:
        print(f"[chat_handler] Pipeline error: {e}")
        return {
            "answer": _PIPELINE_ERROR_REPLY,
            "language": language,
            "escalated": False,
            "chunks_used": 0,
        }

    return {
        "answer": answer,
        "language": language,
        "escalated": False,
        "chunks_used": len(chunks),
    }


# ---------------------------------------------------------------------------
# Quick manual test — runs only when this file is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== TEST 1: Normal educational question ===")
    result = handle_message("What is the safe period method of family planning?")
    print(f"Answer      : {result['answer']}")
    print(f"Language    : {result['language']}")
    print(f"Escalated   : {result['escalated']}")
    print(f"Chunks used : {result['chunks_used']}")

    print("\n=== TEST 2: Empty message ===")
    result = handle_message("")
    print(f"Answer      : {result['answer']}")
    print(f"Chunks used : {result['chunks_used']}")

    print("\n=== TEST 3: Language code preserved ===")
    result = handle_message("Kisa ake nufi da hana haihuwa?", language="ha")
    print(f"Language in result: {result['language']}  (expected: ha)")
    print(f"Chunks used: {result['chunks_used']}")

    print("\n=== TEST 4: Onboarding flow ===")
    # Redirect session storage to a temp file so this test never
    # touches the real data/user_sessions.json — same pattern used by
    # evaluation/logger.py's test block for pilot_logs.jsonl.
    import tempfile
    import os as _os

    def _safe(text: str) -> str:
        # Windows consoles are often cp1252, which can't print some
        # non-ASCII characters (Yoruba diacritics, emoji) -- encode
        # defensively so tests never crash on display alone. This is a
        # terminal limitation, not a pipeline bug.
        return text.encode("ascii", errors="backslashreplace").decode("ascii")

    original_sessions_file = _SESSIONS_FILE
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    tmp.write("{}")
    tmp.close()
    _SESSIONS_FILE = Path(tmp.name)

    test_number = "whatsapp:+2348011111111"

    result = handle_message("hi", user_id=test_number)
    print(f"  [new user 'hi']       escalated={result['escalated']}  "
          f"welcome_shown={'language do you prefer' in result['answer']}")

    result = handle_message("1", user_id=test_number)
    print(f"  [reply '1']           escalated={result['escalated']}  "
          f"state_prompt_shown={'which state are you in' in result['answer']}")

    result = handle_message("Lagos", user_id=test_number)
    print(f"  [reply 'Lagos']       escalated={result['escalated']}  "
          f"all_set_shown={'You are all set' in result['answer']}")

    sessions_after = json.loads(Path(tmp.name).read_text(encoding="utf-8"))
    saved_session = sessions_after[_hash_user_id(test_number)]
    print(f"  Saved session         : {saved_session}  (expected: language=english, state=Lagos, onboarded=true)")

    result = handle_message("What is the safe period method?", user_id=test_number)
    print(f"  [returning user]      escalated={result['escalated']}  "
          f"chunks_used={result['chunks_used']}  "
          f"(expected: normal pipeline ran, not onboarding)")

    print("\n=== TEST 5: Escalation uses stored state (real facility names) ===")
    result = handle_message("I was assaulted", user_id=test_number)
    print(f"  Escalated: {result['escalated']}")
    print(f"  Answer   : {result['answer']}")
    print(f"  Contains a Lagos facility name (not generic PHC fallback): "
          f"{'Phc' in result['answer'] or 'PHC' in result['answer']}")

    print("\n=== TEST 6: Response translated to stored language (Yoruba) ===")
    yoruba_number = "whatsapp:+2348022222222"
    sessions = _load_sessions()
    sessions[_hash_user_id(yoruba_number)] = {
        "language": "yoruba", "state": "Lagos", "onboarded": True,
    }
    _save_sessions(sessions)
    result = handle_message("What is pregnancy?", user_id=yoruba_number)
    print(f"  Language in result: {result['language']}  (expected: yo)")
    print(f"  Answer (Yoruba)   : {_safe(result['answer'][:200])}")

    print("\n=== TEST 7: Facility-seeking phrase escalates, not RAG ===")
    result = handle_message("Which primary health center can I visit", user_id=test_number)
    print(f"  Escalated  : {result['escalated']}  (expected: True)")
    print(f"  Chunks used: {result['chunks_used']}  (expected: 0 -- never reached RAG)")
    print(f"  Answer     : {result['answer'][:200]}")

    print("\n=== TEST 8: Reset for an already-onboarded user ===")
    # test_number is fully onboarded (language=english, state=Lagos) at
    # this point from TESTs 4-7.
    result = handle_message("reset", user_id=test_number)
    print(f"  Reset reply mentions language menu: "
          f"{'what language do you prefer' in result['answer'].lower()}")
    sessions_after_reset = _load_sessions()
    reset_session = sessions_after_reset[_hash_user_id(test_number)]
    print(f"  Session after reset: {reset_session}  (expected: language=None, state=None, onboarded=False)")

    result = handle_message("2", user_id=test_number)
    print(f"  Next message '2' resumes onboarding: "
          f"{'which state are you in' in result['answer'].lower()}  (expected: True)")

    print("\n=== TEST 9: Reset for a user with no session at all ===")
    never_contacted_number = "whatsapp:+2348044444444"
    result = handle_message("reset", user_id=never_contacted_number)
    print(f"  No crash, welcome shown: "
          f"{'what language do you prefer' in result['answer'].lower()}")

    print("\n=== TEST 10: Direct language switch for an onboarded user ===")
    yoruba_switch_number = "whatsapp:+2348055555555"
    sessions = _load_sessions()
    sessions[_hash_user_id(yoruba_switch_number)] = {
        "language": "english", "state": "Oyo", "onboarded": True,
    }
    _save_sessions(sessions)

    result = handle_message("speak yoruba", user_id=yoruba_switch_number)
    print(f"  Language in result : {result['language']}  (expected: yo)")
    print(f"  Confirmation shown : {_safe(result['answer'])!r}")
    sessions = _load_sessions()
    updated = sessions[_hash_user_id(yoruba_switch_number)]
    print(f"  State untouched    : {updated['state']}  (expected: Oyo -- unchanged by language switch)")

    print("\n=== TEST 11: Language menu, then digit switch ===")
    hausa_switch_number = "whatsapp:+2348066666666"
    sessions = _load_sessions()
    sessions[_hash_user_id(hausa_switch_number)] = {
        "language": "english", "state": "Kano", "onboarded": True,
    }
    _save_sessions(sessions)

    result = handle_message("language", user_id=hausa_switch_number)
    print(f"  Menu shown: {'1 for english' in result['answer'].lower()}")

    result = handle_message("2", user_id=hausa_switch_number)
    print(f"  Switched to Hausa: {result['answer'] == _LANGUAGE_SWITCH_CONFIRMATIONS['hausa']}")

    print("\n=== TEST 12: Mid-onboarding direct switch doesn't derail onboarding ===")
    mid_onboarding_number = "whatsapp:+2348077777777"
    handle_message("hi", user_id=mid_onboarding_number)          # creates pending session
    handle_message("2", user_id=mid_onboarding_number)            # picks Hausa, now awaiting state

    result = handle_message("speak english", user_id=mid_onboarding_number)
    print(f"  Switch confirmation shown: {result['answer'] == _LANGUAGE_SWITCH_CONFIRMATIONS['english']}")
    sessions = _load_sessions()
    mid_session = sessions[_hash_user_id(mid_onboarding_number)]
    print(f"  Still mid-onboarding (onboarded=False, language=english): {mid_session}")

    result = handle_message("Kano", user_id=mid_onboarding_number)
    print(f"  Onboarding completes with new language: "
          f"{'You are all set' in result['answer']}")
    sessions = _load_sessions()
    final_session = sessions[_hash_user_id(mid_onboarding_number)]
    print(f"  Final session: {final_session}  (expected: language=english, state=Kano, onboarded=True)")

    print("\n=== TEST 13: Full reset -> onboarding cycle saves correctly ===")
    handle_message("reset", user_id=test_number)
    handle_message("3", user_id=test_number)          # Yoruba
    result = handle_message("Rivers", user_id=test_number)
    sessions = _load_sessions()
    cycle_session = sessions[_hash_user_id(test_number)]
    print(f"  Final session after reset + full onboarding: {cycle_session}")
    print(f"  (expected: language=yoruba, state=Rivers, onboarded=True)")

    _SESSIONS_FILE = original_sessions_file
    _os.unlink(tmp.name)
