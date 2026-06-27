# SRH Chatbot — Project Context for Claude Code

## What This Project Is
A RAG-based multilingual chatbot for adolescent sexual and reproductive
health (SRH) information in Nigeria. Target users: ages 16-24.
Languages: English, Hausa, Yoruba, Igbo.
MSc AI/ML dissertation project — Miva Open University.

---

## Tech Stack
- Python 3.10+, FastAPI 0.138.0
- ChromaDB ("Abiyamo" collection, cosine distance) for vector storage
- sentence-transformers (all-MiniLM-L6-v2) for embeddings — LOCAL, no API
- Gemini API via **google-genai v2.8.0** (new SDK, NOT google-generativeai)
  - Client pattern: `genai.Client(api_key=...)` then `client.models.generate_content(model=..., contents=..., config=...)`
  - `system_instruction` goes in `GenerateContentConfig`, NOT inline
  - DEFAULT_MODEL: gemini-3.5-flash, GENERATION_TEMPERATURE: 0.3
  (gemini-2.0-flash shut down June 1 2026; gemini-2.5-flash-lite shuts down
  July 22 2026; gemini-3.5-flash has no announced shutdown date as of June 2026)
- NLLB-200 distilled (facebook/nllb-200-distilled-600M) for translation — NOT YET BUILT
- WhatsApp via Twilio REST API using httpx directly (NO Twilio SDK)
- LangChain for document loading and chunking

---

## Current Build Status

### Phase 1 — COMPLETE

#### Track A (Knowledge Base) — src/knowledge_base/
- pdf_loader.py ✅ — loads PDFs, OCR fallback (pytesseract + pdf2image)
- text_splitter.py ✅ — chunks text (500 chars, 50 overlap, RecursiveCharacterTextSplitter)
- embedder.py ✅ — local all-MiniLM-L6-v2, NO API calls, no rate limits
- build_knowledge_base.py ✅ — full ingestion pipeline, writes to ChromaDB
- manage_documents.py ✅ — list/remove/check documents in collection
- watch_and_ingest.py ✅ — scheduled auto-ingestion with ingestion_log.json

ChromaDB collection: "Abiyamo"
- 1848 chunks (FMOH Manual: 1711, 12 Questions SRH doc: 137)
- MUST use cosine distance: metadata={"hnsw:space": "cosine"}
- Was rebuilt from L2 to cosine after testing revealed L2 was ChromaDB's
  default and gave weaker retrieval for this embedding model
- Every get_collection() call across ALL files must include this metadata

#### Track B (Annotation) — src/annotation/
- merge_annotations.py ✅ — merges 2 annotators' Label Studio CSV exports,
  includes JMP long-format export for Cohen's Kappa (3 separate scores,
  one per category: educational/diagnostic/crisis)
- calculate_kappa.py — EXISTS but DEFERRED (Kappa calculated in JMP instead)
- STATUS: BLOCKED — waiting on annotators to finish labeling in Label Studio
- Multi-label design: a query CAN be educational AND diagnostic AND/OR crisis simultaneously

---

### Phase 2 — NOT STARTED
- Blocked on Track B annotation completing first
- Plan: fine-tune DistilBERT for 3-class multi-label intent classification
- Will be run on Google Colab (GPU), not local machine
- When ready: replace ONLY escalation._detect_type() — public interface unchanged

---

### Phase 3 — LARGELY COMPLETE

#### src/safety/ ✅
- escalation.py ✅ BUILT
  - Keyword-based crisis/diagnostic detection (13/13 test cases pass)
  - BYPASSES LLM entirely — returns scripted text + helpline numbers only
  - _detect_type() is the ONLY function to replace when Phase 2 DistilBERT
    classifier is ready; public interface (should_escalate,
    get_escalation_response) stays UNCHANGED
- data/helplines.json ✅ BUILT
  - Placeholder structure: national crisis, sexual_abuse, general_health categories
  - All numbers marked PLACEHOLDER_NOT_VERIFIED except 112 (Nigeria emergency)
  - State-level structure ready for additions
  - DO NOT deploy to real users until every number is personally verified active

#### src/rag_pipeline/
- retriever.py ✅ BUILT — queries ChromaDB, returns top 3 chunks
  (text/source/chunk_index/distance)
- translator.py ✅ BUILT
  - NLLB-200 distilled 600M, bidirectional (English <-> Hausa/Yoruba/Igbo)
  - Language codes: eng_Latn, hau_Latn, yor_Latn, ibo_Latn
  - Public interface: to_english(text, source_lang) / from_english(text, target_lang)
  - Model cached at module level (_model_cache / _tokenizer_cache pattern)
  - Falls back to original text on error — pipeline never crashes due to
    a translation failure
  - Wired into chat_handler.py: to_english() before retrieval,
    from_english() after generation
- prompt_builder.py ✅ BUILT
  - build_prompt(question, chunks) assembles Gemini prompt
  - get_system_prompt() exposed separately for Gemini's system_instruction field
  - MAX_CHUNK_CHARS=600
- generator.py ✅ BUILT
  - Uses google-genai v2.8.0 (new SDK)
  - DEFAULT_MODEL=gemini-2.0-flash, GENERATION_TEMPERATURE=0.3
  - _client_cache pattern
  - Returns FALLBACK_RESPONSE string on any API error — never raises to caller
  - answer_from_chunks() convenience wrapper combines prompt_builder + generator

#### src/whatsapp/
- sender.py ✅ BUILT
  - send_whatsapp_message(to, body) POSTs to Twilio REST API via httpx
  - Returns message SID or None on failure
  - Requires in .env: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
- webhook.py ✅ BUILT
  - FastAPI APIRouter, POST /whatsapp/incoming
  - Receives Twilio form-POST, returns empty TwiML IMMEDIATELY
  - Processes in BackgroundTask via chat_handler.handle_message()
  - Sends reply via sender.py AFTER pipeline completes
  - Mounted in main.py at prefix /whatsapp
  - BackgroundTasks used specifically to avoid Twilio's 15-second timeout

#### src/api/
- chat_handler.py ✅ BUILT
  - handle_message(message, language) is the SINGLE pipeline entry point
  - Called by both main.py and webhook.py
  - Routing order: escalation check → (TODO: translation) → retrieval → generation
  - Returns {answer, language, escalated, chunks_used}
  - Translation TODOs already marked — translator.py will drop in here
- main.py ✅ BUILT
  - FastAPI app with lifespan startup
  - Startup pre-loads embedding model, logs ChromaDB chunk count
  - Endpoints: GET /health, POST /chat
  - Mounts WhatsApp router
  - Logs every /chat call via evaluation/logger.py
  - Run: uvicorn src.api.main:app --reload --port 8000

#### evaluation/
- logger.py ✅ BUILT
  - log_interaction(user_id, message, result, response_time_ms, channel)
  - Appends to data/pilot_logs.jsonl
  - Phone numbers SHA-256 hashed (16 hex chars) before writing
  - read_logs() and count_logs() used by pilot_report.py and /health
- pilot_report.py ✅ BUILT
  - 7-section terminal report using rich + pandas
  - Sections: overview, language distribution, escalation rate, chunk stats,
    response time percentiles, 14-day daily volume bar chart, last 10 interactions
  - Generates synthetic preview if no log exists
  - Run: python -m evaluation.pilot_report

---

### Phase 4 — NOT STARTED

---

## What To Build Next (in order)

1. **Retry logic in generator.py** — Add exponential backoff for Gemini
   API rate limit errors (429). Currently returns FALLBACK_RESPONSE
   immediately on any error. Scalability concern for pilot.

3. **Phase 2 intent classifier** (Google Colab, GPU) — when Track B
   annotation is ready. Replace escalation._detect_type() ONLY.

4. **Phase 4 pilot evaluation** — after translator.py is built and
   credentials are configured.

---

## Credentials Needed Before Live Deployment

- GEMINI_API_KEY — current key is quota-exhausted (free tier: 0).
  Generate new key at Google AI Studio.
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM —
  not yet set in .env. Required for WhatsApp sending.
- data/helplines.json — ALL numbers are PLACEHOLDER_NOT_VERIFIED.
  Must be personally verified before any real user receives them.

---

## Critical Design Decisions (DO NOT CHANGE WITHOUT UNDERSTANDING WHY)

1. **Embedding model**: local all-MiniLM-L6-v2, NOT Gemini API.
   Reason: Gemini free tier has 1000/day hard cap; project needs 1848+
   embeddings for knowledge base alone.

2. **ChromaDB distance**: MUST be cosine (metadata={"hnsw:space": "cosine"}).
   Reason: L2 is ChromaDB's default but gives weaker retrieval for
   sentence-transformer models. All get_collection() calls must include this.
   This setting only applies at collection CREATION — cannot be changed
   retroactively on existing collections.

3. **Escalation bypasses LLM**: diagnostic/crisis queries NEVER touch Gemini.
   Returns scripted text + verified helplines only. Safety-critical.

4. **Translation is two-directional**: question → English BEFORE retrieval
   (so ChromaDB can match), answer → user language AFTER generation.

5. **NLLB-200 distilled 600M**: not full model. Reason: no GPU on dev machine.
   Same language coverage, modestly reduced quality on complex text.

6. **Gemini SDK is google-genai v2.8.0**: NOT google-generativeai (old SDK).
   These are different packages with different APIs. Do not confuse them.

7. **WhatsApp uses httpx directly**: no Twilio Python SDK. httpx was already
   a project dependency.

8. **Twilio webhook uses BackgroundTasks**: empty TwiML returned immediately,
   reply sent after pipeline. Avoids Twilio's 15-second response timeout.

9. **helplines.json placeholder safety**: is_placeholder() check in
   escalation.py prevents deploying unverified numbers. DO NOT remove this check.

---

## Folder Structure

```
srh-chatbot/
├── data/
│   ├── raw_pdfs/              (source PDFs — FMOH manual, 12 Questions SRH doc)
│   ├── chroma_db/             (ChromaDB persistent storage — DO NOT DELETE
│   │                           unless intentionally rebuilding)
│   ├── raw_queries/           (annotator Label Studio exports go here)
│   ├── helplines.json         (PLACEHOLDER — verify all numbers before use)
│   ├── ingestion_log.json     (tracks processed PDFs for watch_and_ingest)
│   └── pilot_logs.jsonl       (anonymised interaction logs for evaluation)
├── src/
│   ├── knowledge_base/        (Phase 1 Track A — all built ✅)
│   ├── annotation/            (Phase 1 Track B — built, blocked on data 🟡)
│   ├── safety/                (Phase 3 — escalation.py built ✅)
│   ├── rag_pipeline/          (Phase 3 — all built ✅: retriever, prompt_builder,
│   │                           generator, translator)
│   ├── whatsapp/              (Phase 3 — sender.py, webhook.py ✅)
│   └── api/                   (Phase 3 — main.py, chat_handler.py ✅)
├── evaluation/
│   ├── logger.py              ✅
│   └── pilot_report.py        ✅
└── CLAUDE.md
```

---

## Style and Conventions

- Every module handles ONE file/task at a time — no folder looping inside modules
- Module-level model caching (_model_cache pattern) used consistently
- get_collection() ALWAYS includes metadata={"hnsw:space": "cosine"}
- All sensitive config (API keys) in .env, never hardcoded
- Test block at bottom of every file (if __name__ == "__main__":)
- Comments explain WHY decisions were made, not just what the code does
- Gemini SDK: google-genai v2.8.0 — use genai.Client(), not genai.configure()