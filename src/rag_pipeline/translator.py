"""
translator.py

Purpose:
    Bidirectional translation between English and the three target
    languages — Hausa, Yoruba, and Igbo — using the NLLB-200 distilled
    600M model running entirely locally (no API calls, no rate limits).

Where this sits in the pipeline:
    [THIS FILE — incoming] → retriever.py → prompt_builder.py
    → generator.py → [THIS FILE — outgoing]

    to_english() runs BEFORE retrieval so ChromaDB can compare the
    query against English-language knowledge base chunks.
    from_english() runs AFTER generation, translating the answer
    back into the user's original language.

Design notes:
    - Uses facebook/nllb-200-distilled-600M, NOT the full 3.3B-parameter
      NLLB-200 model. Reason: the dev machine has no GPU; the distilled
      model fits on CPU without making latency unacceptable for a pilot.
    - Model and tokenizer are loaded once and cached at module level.
      First call to any translation function takes ~30-60s on CPU while
      the model loads; subsequent calls are fast.
    - On any translation error, the original text is returned unchanged
      and a warning is printed. The pipeline degrades to English-only
      rather than crashing, which is safe for a pilot.
    - English-to-English calls short-circuit immediately — no model is
      loaded and no overhead is incurred. to_english("...", "en") is
      always a no-op.
    - NLLB uses flores-200 language codes internally (e.g. "hau_Latn"),
      not BCP-47 codes (e.g. "ha"). The LANGUAGE_CODES dict maps between
      the two so callers only ever see BCP-47 codes.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Maps BCP-47 codes (used throughout this project) to the flores-200
# codes NLLB-200 expects internally. Only the four project languages
# are listed — any other code is treated as unsupported.
LANGUAGE_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "ha": "hau_Latn",
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
}

# Plain-name → NLLB code mapping. Exposed so webhook.py and other
# entry points can validate a user's stated language preference
# without duplicating this list.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "english": "eng_Latn",
    "hausa":   "hau_Latn",
    "yoruba":  "yor_Latn",
    "igbo":    "ibo_Latn",
}

# NLLB-200 distilled 600M. The full model (3.3B) would give marginally
# better quality but cannot run on CPU at acceptable latency for a pilot.
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Maximum tokens the model may generate per translation call. 512 is
# generous for typical question and answer lengths; very long text is
# not expected in this chatbot context.
MAX_NEW_TOKENS = 512

# Module-level cache — same _cache pattern used in retriever.py and
# generator.py. Loading the model is expensive; it must only happen once.
_model_cache: AutoModelForSeq2SeqLM | None = None
_tokenizer_cache: AutoTokenizer | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model() -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Loads the NLLB model and tokenizer from HuggingFace (or local cache
    if already downloaded) and stores them in module-level cache variables.
    Subsequent calls return the cached objects immediately.
    """
    global _model_cache, _tokenizer_cache
    if _model_cache is None or _tokenizer_cache is None:
        print(
            f"[translator] Loading '{MODEL_NAME}' "
            "(first call only — may take 30-60s on CPU)..."
        )
        _tokenizer_cache = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model_cache = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print("[translator] Model loaded.")
    return _model_cache, _tokenizer_cache


def _translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Core translation function. Both language codes must be flores-200
    codes (e.g. "hau_Latn"), not BCP-47 — callers are responsible for
    converting via LANGUAGE_CODES before calling this.

    Returns the translated string, or the original text unchanged if
    any step in the translation pipeline raises an exception.
    """
    if not text or not text.strip():
        return text

    try:
        model, tokenizer = _load_model()

        # Setting src_lang on the tokenizer before encoding tells it
        # which language-ID token to prepend to the input sequence.
        # This must be set per call because the same cached tokenizer
        # object is reused across calls with different source languages.
        tokenizer.src_lang = source_lang

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # forced_bos_token_id is how NLLB steers the decoder toward a
        # specific output language — it forces the first generated token
        # to be the target language's special ID token, which conditions
        # the rest of the generation on that language.
        #
        # convert_tokens_to_ids works for both NllbTokenizer and
        # NllbTokenizerFast (faster Rust-based variant) without needing
        # the .lang_code_to_id dict attribute, which is only on the
        # slow tokenizer.
        target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_new_tokens=MAX_NEW_TOKENS,
                max_length=None,  # suppress conflict with model's default max_length=200
            )

        translated = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return translated

    except Exception as e:
        print(f"[translator] Translation failed ({source_lang} -> {target_lang}): {e}")
        return text


# ---------------------------------------------------------------------------
# Public interface — called by chat_handler.py
# ---------------------------------------------------------------------------

def to_english(text: str, source_lang: str = "en") -> str:
    """
    Translates text into English from the given source language.

    Parameters:
        text        : The user's message as received.
        source_lang : BCP-47 code of the source language
                      ("en", "ha", "yo", or "ig"). Defaults to "en".
                      If "en", returns the text unchanged immediately —
                      no model is loaded.

    Returns:
        The English translation, or text unchanged if source_lang is
        "en", if the language code is unrecognised, or if translation
        fails.
    """
    if source_lang == "en":
        return text

    nllb_source = LANGUAGE_CODES.get(source_lang)
    if nllb_source is None:
        print(
            f"[translator] Unsupported language code '{source_lang}' — "
            "returning text unchanged."
        )
        return text

    return _translate(text, source_lang=nllb_source, target_lang=LANGUAGE_CODES["en"])


def from_english(text: str, target_lang: str = "en") -> str:
    """
    Translates text from English into the given target language.

    Parameters:
        text        : The English-language answer produced by the generator.
        target_lang : BCP-47 code of the target language
                      ("en", "ha", "yo", or "ig"). Defaults to "en".
                      If "en", returns the text unchanged immediately —
                      no model is loaded.

    Returns:
        The translated answer, or text unchanged if target_lang is "en",
        if the language code is unrecognised, or if translation fails.
    """
    if target_lang == "en":
        return text

    nllb_target = LANGUAGE_CODES.get(target_lang)
    if nllb_target is None:
        print(
            f"[translator] Unsupported language code '{target_lang}' — "
            "returning text unchanged."
        )
        return text

    return _translate(text, source_lang=LANGUAGE_CODES["en"], target_lang=nllb_target)


def get_supported_languages() -> list[str]:
    """Returns the plain-name list of languages this module can translate."""
    return list(SUPPORTED_LANGUAGES.keys())


# ---------------------------------------------------------------------------
# Quick manual test — runs only when this file is executed directly.
# First call loads the model (~30-60s on CPU). Requires internet access
# on first run to download the model (~2.4 GB), cached locally after that.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== TEST 1: Hausa -> English ===")
    hausa_q = "Kisa ake nufi da hana haihuwa?"
    result = to_english(hausa_q, source_lang="ha")
    print(f"Input  (ha): {hausa_q}")
    print(f"Output (en): {result}")

    print("\n=== TEST 2: English -> Hausa ===")
    english_a = "Family planning means choosing when and how many children to have."
    result = from_english(english_a, target_lang="ha")
    print(f"Input  (en): {english_a}")
    print(f"Output (ha): {result}")

    print("\n=== TEST 3: English passthrough (no translation, no model load) ===")
    original = "What is a condom?"
    result = to_english(original, source_lang="en")
    print(f"Input  (en): {original}")
    print(f"Output (en): {result}")
    print(f"Unchanged  : {original == result}")

    print("\n=== TEST 4: Yoruba -> English ===")
    yoruba_q = "Kini itumo idena oyun?"
    result = to_english(yoruba_q, source_lang="yo")
    print(f"Input  (yo): {yoruba_q}")
    print(f"Output (en): {result}")

    print("\n=== TEST 5: English -> Igbo ===")
    english_a = "You can prevent HIV by using condoms consistently."
    result = from_english(english_a, target_lang="ig")
    print(f"Input  (en): {english_a}")
    print(f"Output (ig): {result}")

    print("\n=== TEST 6: Unsupported language code ===")
    result = to_english("Bonjour", source_lang="fr")
    print(f"Input  (fr): Bonjour")
    print(f"Output     : {result}  (expected: unchanged)")
