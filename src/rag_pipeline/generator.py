"""
generator.py

Purpose:
    Takes a fully-assembled prompt (built by prompt_builder.py) and
    sends it to the Gemini API to generate a response. Returns the
    response text as a plain string.

Where this sits in the pipeline:
    translator.py (incoming) → retriever.py → prompt_builder.py
    → [THIS FILE] → translator.py (outgoing)

    This module is the final step before the answer is translated back
    into the user's language (if they wrote in Hausa, Yoruba, or Igbo).

Design notes:
    - Uses the new google-genai SDK (v2.x), NOT the old
      google-generativeai package. The client interface changed
      significantly between versions — this file uses:
          from google import genai
          client = genai.Client(api_key=...)
          client.models.generate_content(model=..., contents=..., config=...)
    - API key is loaded from .env — NEVER hardcoded.
    - A module-level _client_cache avoids re-creating the API client on
      every call, consistent with the caching pattern used across this
      project (embedder.py, retriever.py).
    - Temperature is set low (0.3) on purpose — this is a health
      information tool, not a creative writing tool. Accuracy matters
      more than variety.
    - On any API error, generate_answer() returns a safe fallback
      message rather than raising an exception. The caller (chat_handler
      or whatsapp layer) should never crash because of a generation
      failure — it should gracefully tell the user something went wrong.
    - system_instruction is passed via GenerateContentConfig, which is
      the correct place for it in the v2 SDK. It is kept separate from
      the main prompt contents so the LLM treats it as a persistent
      persona instruction, not part of the user's message.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.rag_pipeline.prompt_builder import build_prompt, get_system_prompt

# Load .env from project root so GEMINI_API_KEY is available.
# override=False means if the key was already set in the shell
# environment, that value is respected rather than overwritten.
load_dotenv(override=False)

# Default model. gemini-2.0-flash is chosen for this project because:
# - It has a generous free-tier rate limit (suitable for a pilot)
# - Low latency — important for a WhatsApp chatbot where users expect
#   a reply within seconds, not minutes
# - Quality is sufficient for grounded RAG responses where the model
#   is primarily reformatting retrieved text, not reasoning from scratch
DEFAULT_MODEL = "gemini-3.5-flash"

# Temperature for generation. Kept low because this is a health
# information tool — reproducible, accurate answers are more important
# than creative or varied phrasing. 0.3 still allows natural-sounding
# language without wandering from the retrieved facts.
GENERATION_TEMPERATURE = 0.3

# Max tokens for the response. 1024 is enough for a thorough but
# concise chatbot answer; very long responses tend to lose the user on
# WhatsApp anyway.
MAX_OUTPUT_TOKENS = 1024

# Fallback message returned when the Gemini API call fails for any
# reason (network error, quota exhausted, invalid key, etc.).
# Kept generic so it works regardless of what broke.
FALLBACK_RESPONSE = (
    "I'm sorry, I wasn't able to generate a response right now. "
    "Please try again in a moment. If this keeps happening, speak "
    "to a health worker or visit a nearby health facility."
)

# Module-level client cache — same pattern used in embedder.py and
# retriever.py. The genai.Client object is safe to reuse across calls;
# creating it on every request would add unnecessary overhead.
_client_cache = None


# ---------------------------------------------------------------------------
# Step 1: Build (and cache) the Gemini API client
# ---------------------------------------------------------------------------

def get_client() -> genai.Client:
    """
    Loads the GEMINI_API_KEY from the environment and returns a cached
    genai.Client instance. Raises a clear error if the key is missing
    so the problem is obvious at startup rather than silently failing
    at the first user message.
    """
    global _client_cache
    if _client_cache is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[generator] GEMINI_API_KEY not found. "
                "Add it to your .env file as:\n"
                "    GEMINI_API_KEY=your_key_here"
            )
        _client_cache = genai.Client(api_key=api_key)
        print("[generator] Gemini client initialised.")
    return _client_cache


# ---------------------------------------------------------------------------
# Step 2: Main entry point — generate an answer from a prompt
# ---------------------------------------------------------------------------

def generate_answer(
    prompt: str,
    system_prompt: str = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Sends the prompt to the Gemini API and returns the response text.

    Parameters:
        prompt        : The full user-turn prompt, built by
                        prompt_builder.build_prompt(). It already
                        contains the retrieved context and the user's
                        question.
        system_prompt : Optional. The persona/instruction text to pass
                        as Gemini's system_instruction. Defaults to
                        prompt_builder.get_system_prompt() if not
                        supplied — this keeps the default consistent
                        across the whole pipeline without requiring the
                        caller to import prompt_builder separately.
        model         : Gemini model name. Defaults to DEFAULT_MODEL.
                        Can be overridden for testing or future upgrades
                        without changing this module's logic.

    Returns:
        The generated response as a plain string, or FALLBACK_RESPONSE
        if the API call fails for any reason.
    """
    if system_prompt is None:
        # Use the same system prompt defined in prompt_builder so
        # both inline prompting and system-instruction-based prompting
        # are always in sync.
        system_prompt = get_system_prompt()

    try:
        client = get_client()
    except EnvironmentError as e:
        print(e)
        return FALLBACK_RESPONSE

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=GENERATION_TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text.strip()

    except Exception as e:
        # Catching broadly here is deliberate — any API failure
        # (quota, network, content-filter block, etc.) should return
        # the fallback message rather than propagating an exception
        # to the WhatsApp or API layer, which has no sensible way to
        # handle a generation error other than sending this message.
        print(f"[generator] Gemini API call failed: {e}")
        return FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# Convenience wrapper: full pipeline from question + chunks to answer
# ---------------------------------------------------------------------------

def answer_from_chunks(question: str, chunks: list) -> str:
    """
    Convenience function that combines prompt_builder and generator into
    one call. Used by chat_handler.py to keep its own code simple.

    Parameters:
        question : The user's question in English.
        chunks   : Retrieved chunks from retriever.retrieve_relevant_chunks().

    Returns:
        The generated answer string, or FALLBACK_RESPONSE on failure.
    """
    prompt = build_prompt(question, chunks)
    return generate_answer(prompt)


# ---------------------------------------------------------------------------
# Quick manual test — runs only when this file is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # This test makes a real Gemini API call using the key in .env.
    # Make sure .env exists and contains GEMINI_API_KEY before running.
    mock_chunks = [
        {
            "text": (
                "Family planning is the practice of controlling the number "
                "and spacing of children a person or couple has, using "
                "various methods. These include natural methods such as the "
                "safe period (calendar method) and Standard Days Method, as "
                "well as modern methods such as condoms, pills, injectables, "
                "implants, and intrauterine devices (IUDs)."
            ),
            "source": "National-Training-Manual-Adolescent-Health-Nigeria-2011.pdf",
            "chunk_index": 42,
            "distance": 0.18,
        },
        {
            "text": (
                "The safe period method relies on tracking the menstrual "
                "cycle to identify days when pregnancy is less likely. "
                "A woman with a regular 28-day cycle is typically fertile "
                "around days 10-17. Abstaining or using a barrier method "
                "during this window is the core of the calendar method."
            ),
            "source": "12-questions-srh.pdf",
            "chunk_index": 7,
            "distance": 0.23,
        },
    ]

    test_question = "What is the safe period method of family planning?"

    print("[generator] Running end-to-end test with mock chunks...\n")
    answer = answer_from_chunks(test_question, mock_chunks)

    print(f"Question : {test_question}")
    print(f"Answer   : {answer}")

    print("\n[generator] Testing empty-chunks fallback (no context)...\n")
    answer_no_context = answer_from_chunks(test_question, [])
    print(f"Answer (no context): {answer_no_context}")
