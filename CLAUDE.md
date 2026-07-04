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
- preprocess_queries.py ✅ — cleans Google Form CSV export, splits multi-question
  cells, deduplicates, outputs three CSVs for Label Studio and Chapter 4 analysis
  - Splitting order: numbered → newline → slash (Pass 2b) → comma (Pass 2c) →
    question mark → long sentence
  - Pass 2b guard: slash fragments must each be >15 chars (filters HIV/AIDS, yes/no)
  - Pass 2c guard: comma fragments must be >8 chars AND ≥1 must start with a
    question word or end with "?" — prevents destroying normal sentences
  - ingest_synthetic_queries() appends approved synthetic queries from
    data/raw_queries/synthetic_queries.csv into the Label Studio pipeline;
    synthetic rows tagged participant_id="SYNTHETIC" in metadata
  - SYNTHETIC_QUERIES_PATH = "data/raw_queries/synthetic_queries.csv"
  - Terminal summary shows split method breakdown and synthetic ingestion stats
  - Safe to rerun — appends only new participants/queries, never duplicates
- data/raw_queries/synthetic_queries.csv ✅ — 54 approved synthetic queries
  (20 diagnostic, 20 crisis, 14 educational) written by researcher to address
  class imbalance in real participant data
- merge_annotations.py ✅ — merges 2 annotators' Label Studio CSV exports,
  includes JMP long-format export for Cohen's Kappa (3 separate scores,
  one per category: educational/diagnostic/crisis)
- calculate_kappa.py — EXISTS but DEFERRED (Kappa calculated in JMP instead)
- STATUS: COMPLETE — both annotators finished in Label Studio; merged and
  resolved into training_dataset.csv (see Phase 2)
- Multi-label design: a query CAN be educational AND diagnostic AND/OR crisis simultaneously

**Current dataset state (as of last run):**
- 81 real participants, 138 real queries
- 54 synthetic queries (participant_id="SYNTHETIC")
- **192 total queries** in label_studio_import.csv
- Output files: data/raw_queries/label_studio_import.csv,
  queries_with_metadata.csv, participant_summary.csv
- data/raw_queries/annotator_exports.csv — Label Studio Cloud export,
  2 annotators, 197 tasks (194 double-annotated, 3 single-annotated)
- merge_annotations.py output: 54.6% full agreement between annotators;
  disagreements resolved via UNION (conservative — a category counts if
  EITHER annotator flagged it, since missing crisis/diagnostic is worse
  than over-including it)
- data/raw_queries/training_dataset.csv — final training data: 251 rows
  (197 real + 54 synthetic), columns query_id/text/educational/diagnostic/
  crisis/source/gold_label. Used by BOTH notebooks in Phase 2.

---

### Phase 2 — COMPLETE
- notebooks/phase2_baseline_models.ipynb ✅ — TF-IDF + Logistic Regression /
  SVM / Random Forest baselines, run on Colab (CPU is fine, <5 min).
  Establishes that a classical baseline is or isn't sufficient before
  justifying DistilBERT's added complexity in Chapter 4.
- notebooks/phase2_intent_classifier.ipynb ✅ — fine-tunes
  distilbert-base-uncased for 3-label multi-label classification
  (educational/diagnostic/crisis), run on Colab GPU. Saves trained model +
  tokenizer + classifier_config.json to models/intent_classifier/.
  - classifier_config.json: crisis_threshold=0.35 (lower than the 0.5 used
    for the other two labels) — missing a crisis query is worse than a
    false positive, so crisis recall is favoured over precision.
  - Both notebooks load data/raw_queries/training_dataset.csv directly
    (already-binary label columns) — no annotator-column parsing needed.
  - Colab gotcha (hit and fixed twice): the clone-repo cell force-syncs
    to origin (`git fetch` + `reset --hard`) instead of `git pull`,
    because the artifact-saving cell commits inside the Colab clone and
    its unauthenticated `git push` fails silently, causing "divergent
    branches" on the next run otherwise.
  - Colab gotcha: cell 4 installs transformers/accelerate with `-U`
    (not pinned) — pinning transformers to an old version alone breaks
    Trainer with `ImportError: cannot import name 'EncoderDecoderCache'`
    because Colab's preinstalled accelerate expects a newer transformers API.
- src/safety/escalation.py ✅ — _detect_type() now calls the trained
  classifier, with automatic fallback to keyword matching (see Phase 3
  below for details).
- **IMPORTANT — local vs. deployed state**: models/intent_classifier/
  exists in this repo checkout but is EMPTY (the model was trained on
  Colab and its weights have not been copied down here yet). Until the
  trained model files (config.json, model weights, tokenizer files,
  classifier_config.json) are placed in that directory, escalation.py
  silently runs on the Phase 3 keyword fallback, not the classifier.

---

### Phase 3 — LARGELY COMPLETE

#### src/safety/ ✅
- escalation.py ✅ BUILT + PATCHED + Phase 2 classifier integrated
  - _detect_type() runs the trained DistilBERT classifier
    (models/intent_classifier/) first via _load_classifier(); crisis
    threshold 0.35, other labels 0.5, from classifier_config.json.
    Multi-label internally, but still returns str | None (crisis >
    diagnostic > None priority) so should_escalate() and
    get_escalation_response() did not need to change.
  - Falls back to the original phrase-based detection
    (_detect_type_keywords(), 13/13 test cases pass) whenever the
    classifier directory/config is missing or fails to load — see the
    "local vs. deployed state" note under Phase 2 above; this fallback
    is ACTIVE on this machine right now, not just a theoretical path.
  - BYPASSES LLM entirely either way — returns scripted text + helpline
    numbers only
  - CRISIS_PHRASES patched (session 2): added 20 new phrases covering
    assault disclosures ("i was assaulted", "assaulted by", "sexually assaulted"),
    vague disclosures ("something happened to me"), non-consensual touch
    ("he touched me", "touching me inappropriately"), coercion ("i said no but",
    "forcing himself", "without my consent"), physical abuse ("he beats me"),
    self-harm extension ("hurting myself"), drink spiking ("something in my drink")
  - All 5 previously failing crisis patterns now correctly escalate (LLM bypassed)
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
  - SUPPORTED_LANGUAGES dict (plain name → NLLB code) + get_supported_languages()
    exposed so webhook/entry points can validate language without duplicating the map
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

2. **Wire translation fully into chat_handler.py** — translator.py is built
   and wired but the TODO stubs need completing: detect language → to_english()
   before retrieval → from_english() after generation. translator.py is ready.

3. **Copy the trained DistilBERT classifier down from Colab** —
   models/intent_classifier/ is empty on this machine; escalation.py is
   silently running on the keyword fallback until config.json, model
   weights, tokenizer files, and classifier_config.json are placed there.

4. **Phase 4 pilot evaluation** — after translator.py is wired and
   credentials are configured.

---

## Credentials Needed Before Live Deployment

- GEMINI_API_KEY — key in .env is ACTIVE and tested (session 2). /chat
  endpoint responding. Monitor quota usage on free tier.
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM —
  values are in .env but WhatsApp webhook NOT yet tested end-to-end.
  Required for WhatsApp sending.
- data/helplines.json — ALL numbers are PLACEHOLDER_NOT_VERIFIED.
  Must be personally verified before any real user receives them.

## Environment / Dependencies

- Python 3.12.3 (Anaconda base environment)
- NumPy 2.2.6 — scipy upgraded to 1.18.0 and scikit-learn to 1.9.0
  to fix NumPy 2.x incompatibility (server would not start otherwise)
- pyarrow, numexpr, bottleneck still emit NumPy 1.x warnings on import
  but these are non-fatal — pandas and the pipeline work correctly
- requirements.txt is now populated with pinned versions
- Server start command: python -m uvicorn src.api.main:app --port 8000
  (model load takes ~30-60s on CPU; check uvicorn.log for startup status)

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
│   ├── annotation/            (Phase 1 Track B — COMPLETE ✅)
│   ├── safety/                (Phase 3 — escalation.py built ✅, Phase 2
│   │                           classifier integrated with fallback)
│   ├── rag_pipeline/          (Phase 3 — all built ✅: retriever, prompt_builder,
│   │                           generator, translator)
│   ├── whatsapp/              (Phase 3 — sender.py, webhook.py ✅)
│   └── api/                   (Phase 3 — main.py, chat_handler.py ✅)
├── models/
│   ├── baselines/             (Phase 2 — TF-IDF baseline models + vectorizer)
│   └── intent_classifier/     (Phase 2 — EMPTY on this machine; trained on
│                               Colab, not yet copied down — see Phase 2 note)
├── notebooks/
│   ├── phase2_baseline_models.ipynb     ✅
│   └── phase2_intent_classifier.ipynb   ✅
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