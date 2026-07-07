"""
prompt_builder.py

Purpose:
    Takes the user's question (in English, already translated upstream
    if the original was in Hausa/Yoruba/Igbo) and the list of retrieved
    chunks from the knowledge base, and assembles a complete prompt
    string ready to send to the Gemini LLM.

Where this sits in the pipeline:
    translator.py (incoming) → retriever.py → [THIS FILE] → generator.py
    → translator.py (outgoing)

    This module is called AFTER retrieval and BEFORE generation.
    It does not embed, retrieve, translate, or call any API — it
    only formats text. That isolation makes it easy to test and tune
    the prompt wording without touching any model code.

Design notes:
    - This module never handles diagnostic or crisis queries. Those are
      routed to escalation.py before the RAG pipeline is ever entered.
      prompt_builder.py can safely assume every question it receives is
      an educational SRH query.
    - If retrieval returns no chunks (empty knowledge base, low-similarity
      query, etc.), the prompt still works — it explicitly tells the LLM
      there is no relevant context, which reliably produces honest
      "I don't have information on that" responses rather than
      hallucination.
    - get_system_prompt() is exposed separately so generator.py can
      pass it to Gemini's dedicated system instruction field, rather
      than mixing it inline with the user turn. Some Gemini API versions
      handle these differently, so keeping them separable is safer.
"""

# Maximum characters to include per retrieved chunk. Trimming here keeps
# the total prompt length predictable and avoids hitting Gemini's context
# limit when multiple long chunks are returned.
MAX_CHUNK_CHARS = 600

# The system prompt defines Abiyamo's identity, tone, and hard constraints.
# Defined as a module-level constant so it is easy to find, read, and
# update without touching any logic code.
SYSTEM_PROMPT = (
    "You are Abiyamo — a trusted, knowledgeable friend for Nigerian young "
    "people aged 16 to 24. You are warm, non-judgmental, and speak in a "
    "friendly but accurate tone. You never make users feel embarrassed for "
    "asking any question about sexual and reproductive health. You speak "
    "like someone who genuinely cares, not like a textbook or a formal "
    "health worker.\n\n"
    "Rules:\n"
    "- Answer only from the retrieved context provided below\n"
    "- Keep responses under 200 words and suitable for WhatsApp\n"
    "- Do not start with a greeting like 'Hello' or 'Hi'\n"
    "- Do not end with 'Would you like me to continue or explain any part "
    "further?' — remove this entirely\n"
    "- If the retrieved context does not directly address the question, "
    "say exactly: 'I am here to help 🌿 Here are some topics I can assist "
    "with:\n"
    "- Contraception and family planning\n"
    "- STIs and how to protect yourself\n"
    "- Pregnancy and menstruation\n"
    "- Puberty and body changes\n"
    "- Consent and relationships\n\n"
    "Which of these would you like to know more about, or feel free to "
    "ask your question in your own words.'\n"
    "- Never guess or fill gaps with information not in the retrieved "
    "context"
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_prompt(
    question: str, chunks: list, include_sources: bool = False, last_messages: list | None = None
) -> str:
    """
    Assembles the full prompt string to send to the Gemini LLM.

    Parameters:
        question        : The user's question in English. Translation
                          (if the original was in Hausa/Yoruba/Igbo)
                          must happen before this is called — that is
                          translator.py's responsibility, not this
                          module's.
        chunks          : List of dicts returned by
                          retriever.retrieve_relevant_chunks(). Each
                          dict has keys:
                              "text"        — chunk content
                              "source"      — source filename
                              "chunk_index" — position in document
                              "distance"    — cosine distance (lower = closer)
                          May be an empty list — handled gracefully.
        include_sources : If True, appends the source filename and chunk
                          index next to each context block. Useful during
                          development and evaluation; not needed for
                          end-user responses.
        last_messages   : Optional list of the last 1-2 exchanges from
                          chat_handler.py's session store, each a dict
                          {"role": "user"|"assistant", "content": str}.
                          When provided and non-empty, prepended as a
                          "Previous conversation" block so Gemini has
                          enough context to resolve a short follow-up
                          reply ("yes", "tell me more") on its own,
                          instead of the fixed keyword-based redirect
                          this replaced. None/empty (the common case —
                          most questions are standalone) leaves the
                          prompt exactly as before.

    Returns:
        A single prompt string structured as:
            [system instructions]
            ---
            [Previous conversation block, if last_messages given]
            CONTEXT FROM KNOWLEDGE BASE:
            [numbered chunks, or a "no context" notice]
            ---
            USER QUESTION:
            [question]
            ---
            ANSWER:
    """
    if not question or not question.strip():
        # Returning a minimal valid prompt rather than raising an exception
        # keeps generator.py's error surface simpler — it only needs to
        # handle LLM failures, not upstream validation failures.
        question = "(no question provided)"

    context_block = _build_context_block(chunks, include_sources)
    history_block = _build_history_block(last_messages, question)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "---\n\n"
        f"{history_block}"
        "CONTEXT FROM KNOWLEDGE BASE:\n"
        f"{context_block}\n\n"
        "---\n\n"
        "USER QUESTION:\n"
        f"{question.strip()}\n\n"
        "---\n\n"
        "ANSWER:"
    )

    return prompt


def get_system_prompt() -> str:
    """
    Returns the system prompt string on its own.

    Exposed as a separate function so generator.py can pass it to
    Gemini's system_instruction field if the API version being used
    supports a distinct system turn (rather than embedding it inline
    in the user message). Keeping this separable costs nothing here
    and avoids having to refactor prompt_builder.py later when the
    API wiring is added.
    """
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_history_block(last_messages: list | None, question: str) -> str:
    """
    Formats the last 1-2 exchanges into a "Previous conversation" block
    prepended before the retrieved context. Returns "" when there's no
    history -- the common case, since most questions are standalone,
    not follow-ups -- so build_prompt()'s output is byte-for-byte
    unchanged for every existing caller that doesn't pass last_messages.

    Rendered here, not left to the caller, because this is a prompt
    formatting concern (same reasoning as _build_context_block()).
    """
    if not last_messages:
        return ""

    lines = [
        f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
        for msg in last_messages
    ]

    return (
        "Previous conversation:\n"
        + "\n".join(lines)
        + f"\n\nNow answer this follow-up: {question.strip()}\n\n---\n\n"
    )


def _build_context_block(chunks: list, include_sources: bool) -> str:
    """
    Formats retrieved chunks into a numbered context block.

    Each chunk is trimmed to MAX_CHUNK_CHARS. Trimming is done here
    (not in retriever.py) because the right limit depends on how many
    chunks are shown and how long the surrounding prompt is — that is
    a prompt concern, not a retrieval concern.

    If chunks is empty, returns an explicit notice string rather than
    an empty string. An explicit message reliably produces "I don't
    know" responses from the LLM; an empty context block risks the
    model filling the silence with hallucinated content.
    """
    if not chunks:
        return (
            "No relevant information was found in the knowledge base "
            "for this question."
        )

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        trimmed = text[:MAX_CHUNK_CHARS]
        if len(text) > MAX_CHUNK_CHARS:
            trimmed += "..."

        if include_sources:
            source = chunk.get("source", "unknown")
            chunk_index = chunk.get("chunk_index", "?")
            header = f"[{i}] Source: {source}, chunk {chunk_index}"
        else:
            header = f"[{i}]"

        parts.append(f"{header}\n{trimmed}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Quick manual test — runs only when this file is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Mock chunks matching the structure returned by retriever.py —
    # no actual ChromaDB or model calls needed to test this module.
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
                "around days 10–17. Abstaining or using a barrier method "
                "during this window is the core of the calendar method."
            ),
            "source": "12-questions-srh.pdf",
            "chunk_index": 7,
            "distance": 0.23,
        },
    ]

    def _safe(text: str) -> str:
        # Windows consoles are often cp1252, which can't print the 🌿
        # emoji now embedded directly in SYSTEM_PROMPT -- encode
        # defensively so this test never crashes on display alone.
        # Terminal limitation, not a pipeline bug.
        return text.encode("ascii", errors="backslashreplace").decode("ascii")

    test_question = "What is the safe period method of family planning?"

    print("=" * 60)
    print("TEST 1: Normal prompt (no sources)")
    print("=" * 60)
    print(_safe(build_prompt(test_question, mock_chunks)))

    print("\n" + "=" * 60)
    print("TEST 2: Normal prompt (with sources)")
    print("=" * 60)
    print(_safe(build_prompt(test_question, mock_chunks, include_sources=True)))

    print("\n" + "=" * 60)
    print("TEST 3: Empty chunks (no relevant context found)")
    print("=" * 60)
    print(_safe(build_prompt(test_question, [])))

    print("\n" + "=" * 60)
    print("TEST 4: System prompt only (for generator.py system instruction field)")
    print("=" * 60)
    print(_safe(get_system_prompt()))

    print("\n" + "=" * 60)
    print("TEST 5: With conversation history (follow-up)")
    print("=" * 60)
    mock_history = [
        {"role": "user", "content": "What is an STI?"},
        {"role": "assistant", "content": "An STI is an infection passed between people during sexual contact."},
    ]
    follow_up_prompt = build_prompt("Tell me more", mock_chunks, last_messages=mock_history)
    print(_safe(follow_up_prompt))
    print(f"\nContains 'Previous conversation:': {'Previous conversation:' in follow_up_prompt}")
    print(f"Contains prior user turn: {'What is an STI?' in follow_up_prompt}")
    print(f"Contains prior assistant turn: {'infection passed between people' in follow_up_prompt}")

    print("\n" + "=" * 60)
    print("TEST 6: No history -> output identical to TEST 1 (backward compatible)")
    print("=" * 60)
    no_history_prompt = build_prompt(test_question, mock_chunks)
    empty_list_prompt = build_prompt(test_question, mock_chunks, last_messages=[])
    print(f"last_messages=None matches no-param call : {no_history_prompt == build_prompt(test_question, mock_chunks, last_messages=None)}")
    print(f"last_messages=[] also produces no history : {'Previous conversation:' not in empty_list_prompt}")
