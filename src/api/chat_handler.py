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


def handle_message(message: str, language: str = "en") -> dict:
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

    Returns:
        A dict with keys: answer, language, escalated, chunks_used.
        See module docstring for full description.
    """
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

    # Escalation check — MUST run before any retrieval or generation.
    # Diagnostic and crisis queries return scripted text + helplines and
    # never reach the LLM. This is a hard safety rule, not a soft preference.
    if should_escalate(message):
        return {
            "answer":       get_escalation_response(message),
            "language":     language,
            "escalated":    True,
            "chunks_used":  0,
        }

    english_message = to_english(message, source_lang=language)

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
