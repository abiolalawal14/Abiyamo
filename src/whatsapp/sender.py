"""
sender.py

Purpose:
    Sends WhatsApp messages to users via the Twilio REST API.
    Called by webhook.py after processing a user's question in a
    background task, once the RAG pipeline has produced a reply.

Why a separate sender module instead of inline TwiML only:
    Twilio's webhook has a 15-second response timeout. If the RAG
    pipeline (retrieval + Gemini generation) takes longer than that,
    Twilio drops the connection and the user gets no reply. The fix is
    to return an empty TwiML acknowledgement immediately, process the
    question in a FastAPI background task, then send the answer via
    this module. webhook.py owns the split; this module owns the send.

Why httpx instead of the Twilio Python SDK:
    httpx is already installed as a project dependency (ChromaDB and
    google-genai both pull it in). Adding the full Twilio SDK just to
    make one REST call would bloat the dependency tree — Twilio's
    Messages endpoint is a single POST with Basic auth and form-encoded
    data, which httpx handles cleanly.

Credentials required in .env:
    TWILIO_ACCOUNT_SID    — Account SID from Twilio Console
    TWILIO_AUTH_TOKEN     — Auth Token from Twilio Console
    TWILIO_WHATSAPP_FROM  — Your Twilio WhatsApp sender number,
                            e.g. +14155238886 (without "whatsapp:"
                            prefix — this module adds it automatically)

    These are placeholders until the Twilio sandbox is configured.
    Do NOT deploy to real users until all three are verified active.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv(override=False)

# Twilio Messages API endpoint — {sid} is filled from the environment
# at call time, never hardcoded.
TWILIO_MESSAGES_URL = (
    "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
)

# Timeout for the outbound Twilio API call. Twilio typically responds
# in under 2 seconds; 10 seconds is generous but avoids hanging a
# background task indefinitely on a slow network.
REQUEST_TIMEOUT_SECONDS = 10.0


def _get_credentials() -> tuple:
    """
    Reads Twilio credentials from the environment.

    Returns (sid, token, from_number) on success, or (None, None, None)
    if any value is missing. Returning None rather than raising keeps
    the background task that calls send_whatsapp_message() from
    crashing the FastAPI worker on a misconfiguration.
    """
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "").strip()

    if not all([sid, token, from_number]):
        missing = [
            name for name, val in [
                ("TWILIO_ACCOUNT_SID", sid),
                ("TWILIO_AUTH_TOKEN", token),
                ("TWILIO_WHATSAPP_FROM", from_number),
            ]
            if not val
        ]
        print(f"[sender] Missing Twilio credentials in .env: {missing}")
        return None, None, None

    return sid, token, from_number


def _with_whatsapp_prefix(number: str) -> str:
    """
    Ensures a phone number carries the 'whatsapp:' prefix Twilio
    requires for WhatsApp messages. Idempotent — safe to call on a
    number that already has the prefix.
    """
    if number.startswith("whatsapp:"):
        return number
    return f"whatsapp:{number}"


def send_whatsapp_message(to: str, body: str) -> str | None:
    """
    Sends a WhatsApp message to a single recipient via Twilio.

    Parameters:
        to   : Recipient's phone number. Accepts plain E.164 format
               (+2348012345678) or with the 'whatsapp:' prefix already
               present — both are handled correctly.
        body : The message text. Twilio's WhatsApp limit is 1600 chars;
               messages longer than 1500 are truncated here to leave
               a safe buffer.

    Returns:
        The Twilio message SID string on success (used for logging and
        audit trails), or None if the send failed. Never raises —
        failures are logged and None is returned so the background task
        that calls this can handle failure gracefully.
    """
    sid, token, from_number = _get_credentials()
    if sid is None:
        return None

    to_formatted = _with_whatsapp_prefix(to)
    from_formatted = _with_whatsapp_prefix(from_number)

    # Truncate to stay safely below Twilio's 1600-char hard limit.
    if len(body) > 1500:
        body = body[:1497] + "..."

    url = TWILIO_MESSAGES_URL.format(sid=sid)

    try:
        response = httpx.post(
            url,
            data={
                "From": from_formatted,
                "To": to_formatted,
                "Body": body,
            },
            auth=(sid, token),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        message_sid = response.json().get("sid", "unknown")
        print(f"[sender] Message sent to {to}. SID: {message_sid}")
        return message_sid

    except httpx.HTTPStatusError as e:
        # Twilio returned 4xx/5xx — log the body which contains a
        # Twilio error code and human-readable message.
        print(
            f"[sender] Twilio API error sending to {to}: "
            f"{e.response.status_code} — {e.response.text}"
        )
        return None

    except httpx.RequestError as e:
        # Network-level failure: DNS, timeout, connection refused, etc.
        print(f"[sender] Network error sending to {to}: {e}")
        return None


# ---------------------------------------------------------------------------
# Quick manual test — runs only when this file is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os as _os

    print("=== TEST 1: Missing credentials path ===")
    # Temporarily remove credentials to verify the failure path logs
    # correctly and returns None without crashing.
    saved = {
        k: _os.environ.pop(k, None)
        for k in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"]
    }
    result = send_whatsapp_message("+2348012345678", "Test message")
    print(f"Result (expected None): {result}")
    for k, v in saved.items():
        if v is not None:
            _os.environ[k] = v

    print("\n=== TEST 2: whatsapp: prefix helper ===")
    print(_with_whatsapp_prefix("+2348012345678"))         # should add prefix
    print(_with_whatsapp_prefix("whatsapp:+14155238886"))  # should not double-prefix

    print("\n=== TEST 3: Credential check ===")
    sid, _, _ = _get_credentials()
    if sid:
        print("Twilio credentials found in .env — ready to send.")
    else:
        print(
            "No Twilio credentials found. Add these to .env to enable sending:\n"
            "  TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "  TWILIO_AUTH_TOKEN=your_auth_token\n"
            "  TWILIO_WHATSAPP_FROM=+14155238886"
        )
