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
    "You are Abiyamo, a friendly and knowledgeable health assistant for "
    "young people in Nigeria. Your role is to provide accurate, clear, and "
    "non-judgmental information about sexual and reproductive health (SRH).\n\n"
    "Guidelines:\n"
    "- Your audience is young people aged 16–24. Use clear, simple English. "
    "Avoid overly clinical language unless explaining a term is necessary.\n"
    "- Answer only from the provided context. If the context does not contain "
    "enough information to answer the question, say so honestly — do not "
    "make up information that is not in the context.\n"
    "- Never diagnose a medical condition or recommend specific medication. "
    "If a user's question suggests they may need direct medical attention, "
    "encourage them to visit a health facility or speak to a health worker.\n"
    "- Be warm, respectful, and non-judgmental. These topics can feel "
    "sensitive — the user should feel safe asking anything.\n"
    "- Keep answers concise and practical where possible. Avoid unnecessary "
    "filler or repetition."
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_prompt(question: str, chunks: list, include_sources: bool = False) -> str:
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

    Returns:
        A single prompt string structured as:
            [system instructions]
            ---
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

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "---\n\n"
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

    test_question = "What is the safe period method of family planning?"

    print("=" * 60)
    print("TEST 1: Normal prompt (no sources)")
    print("=" * 60)
    print(build_prompt(test_question, mock_chunks))

    print("\n" + "=" * 60)
    print("TEST 2: Normal prompt (with sources)")
    print("=" * 60)
    print(build_prompt(test_question, mock_chunks, include_sources=True))

    print("\n" + "=" * 60)
    print("TEST 3: Empty chunks (no relevant context found)")
    print("=" * 60)
    print(build_prompt(test_question, []))

    print("\n" + "=" * 60)
    print("TEST 4: System prompt only (for generator.py system instruction field)")
    print("=" * 60)
    print(get_system_prompt())
