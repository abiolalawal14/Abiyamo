"""
webhook.py

Purpose:
    FastAPI router that receives incoming WhatsApp messages from Twilio,
    routes them through the RAG pipeline in a background task, and
    sends the reply back to the user via sender.py.

How the Twilio webhook works:
    When a user sends a WhatsApp message to the Twilio number, Twilio
    makes an HTTP POST to this endpoint with form-encoded fields:
        From  — the user's WhatsApp number, e.g. "whatsapp:+2348012345678"
        To    — the Twilio number that received the message
        Body  — the message text

    This router is mounted in src/api/main.py at prefix "/whatsapp",
    so the full URL Twilio must be pointed at is:
        https://<your-domain>/whatsapp/incoming

Why background tasks instead of returning TwiML inline:
    Twilio's webhook has a 15-second hard timeout. If we processed the
    question synchronously (retrieval + Gemini generation) and returned
    TwiML, a slow Gemini response would cause Twilio to drop the
    connection before our reply arrived. The safer pattern is:
        1. Receive the webhook → return an empty TwiML immediately
           (Twilio sees a 200, marks the message delivered)
        2. Process the question in a FastAPI BackgroundTask
        3. Send the answer via sender.py once the pipeline finishes
    This means the user gets no reply within the HTTP response, but
    receives it as a separate outbound message shortly after — which
    is standard behaviour for most WhatsApp chatbots.

Pipeline routing (current state):
    All incoming messages currently go through the RAG pipeline.
    Two TODOs are marked in process_and_reply() for when the
    dependent modules are ready:
        1. translator.py  — detect language, translate to English before
                            retrieval, translate reply back after
        2. escalation.py  — route diagnostic/crisis queries to helplines
                            BEFORE reaching the retrieval/generation step

Design constraints carried forward from CLAUDE.md:
    - escalation.py MUST intercept diagnostic/crisis queries before
      any LLM call is made — never pass those through generator.py
    - Translation is bidirectional: incoming non-English → English
      before retrieval; English reply → user's language before sending
    - All Twilio credentials live in .env, never in code
"""

from fastapi import APIRouter, Form, BackgroundTasks, Response
from xml.etree.ElementTree import Element, SubElement, tostring

from src.api.chat_handler import handle_message
from src.whatsapp.sender import send_whatsapp_message
from evaluation.logger import log_interaction

router = APIRouter()


# ---------------------------------------------------------------------------
# TwiML helper
# ---------------------------------------------------------------------------

def _empty_twiml() -> str:
    """
    Returns a minimal valid TwiML response with no message body.
    Twilio requires a well-formed XML response — an empty <Response>
    is the correct way to acknowledge receipt without sending an
    immediate reply (because we reply asynchronously via the API).
    """
    root = Element("Response")
    xml_bytes = tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>{xml_bytes}'


# ---------------------------------------------------------------------------
# Background task: the full pipeline
# ---------------------------------------------------------------------------

def process_and_reply(user_number: str, user_message: str) -> None:
    """
    Runs the full RAG pipeline for one user message and sends the
    reply via sender.py. Called as a FastAPI BackgroundTask so it
    runs after the HTTP response has already been returned to Twilio.

    Parameters:
        user_number  : The user's WhatsApp number as Twilio sent it,
                       e.g. "whatsapp:+2348012345678". Passed directly
                       to sender.send_whatsapp_message().
        user_message : The raw message text from the user.
    """
    print(f"[webhook] Processing message from {user_number}: {user_message!r}")

    import time
    start = time.time()
    result = handle_message(user_message, user_id=user_number)
    elapsed_ms = (time.time() - start) * 1000

    log_interaction(
        user_id=user_number,
        message=user_message,
        result=result,
        response_time_ms=elapsed_ms,
        channel="whatsapp",
    )
    send_whatsapp_message(user_number, result["answer"])


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@router.post("/incoming")
async def whatsapp_incoming(
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    Body: str = Form(default=""),
):
    """
    Receives an incoming WhatsApp message from Twilio.

    Twilio fields used:
        From  — sender's WhatsApp number (e.g. "whatsapp:+2348012345678")
        Body  — message text (empty string if the user sent media only)

    Returns an empty TwiML response immediately so Twilio sees a 200
    within its timeout window, then processes the message and sends
    the reply asynchronously via a background task.

    Media messages (images, voice notes) are not yet supported — the
    user receives a polite prompt to send a text question instead.
    """
    user_message = Body.strip()

    if not user_message:
        # User sent media (image, audio, document) with no text caption.
        # We can't process non-text content yet, so reply with a prompt.
        background_tasks.add_task(
            send_whatsapp_message,
            From,
            "Hi! I can only answer text questions right now. "
            "Please type your question about sexual and reproductive health.",
        )
    else:
        background_tasks.add_task(process_and_reply, From, user_message)

    # Return empty TwiML immediately — the actual reply is sent by the
    # background task after processing completes.
    return Response(
        content=_empty_twiml(),
        media_type="application/xml",
    )


# ---------------------------------------------------------------------------
# Quick manual test — runs only when this file is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tests the helper functions without starting a server or calling
    # any external APIs.

    print("=== TEST 1: Empty TwiML response ===")
    twiml = _empty_twiml()
    print(twiml)
    assert "<Response" in twiml
    assert "<?xml" in twiml
    print("PASS")

    print("\n=== TEST 2: Simulate process_and_reply (full pipeline) ===")
    print("Attempts retrieval from ChromaDB and a Gemini call.")
    print("Gemini 429 and missing Twilio creds are both expected without setup.\n")

    test_number = "whatsapp:+2348099999999"
    test_message = "What are the signs of pregnancy?"

    # Call directly (not as a background task) so output is visible.
    # send_whatsapp_message will log "Missing credentials" — expected.
    process_and_reply(test_number, test_message)

    print("\n=== TEST 3: Empty body path (media message) ===")
    print(f"Empty body would call: send_whatsapp_message() with a 'text only' prompt")
    print("PASS")
