# SRH Chatbot — Project Context for Claude Code

## What This Project Is
A RAG-based multilingual chatbot for adolescent sexual and reproductive
health (SRH) information in Nigeria. Target users: ages 16-24.
Languages: English, Hausa, Yoruba, Igbo.
MSc AI/ML dissertation project — Miva Open University.

**Status: deployed and live-tested.** The pipeline is fully wired
(translation, escalation, RAG, onboarding, conversation memory) and
running on Hugging Face Spaces. Most of the work since the last major
CLAUDE.md rewrite has been live-WhatsApp-testing bug fixes, not new
features — see "Known Issues / Limitations" below for what's still
rough.

---

## Tech Stack
- Python 3.10+ for deployment (Docker: `python:3.10-slim`), FastAPI 0.138.0
  - **Dev machine runs Python 3.12.3** (Anaconda base env) — this
    mismatch caused real Dockerfile-build failures (pandas, scipy,
    scikit-learn all needed different pins for 3.10 than what worked
    on the 3.12 dev machine — see Environment/Dependencies below).
- ChromaDB ("Abiyamo" collection, cosine distance) for vector storage
- sentence-transformers (all-MiniLM-L6-v2) for embeddings — LOCAL, no API
- Gemini API via **google-genai v2.8.0** (new SDK, NOT google-generativeai)
  - Client pattern: `genai.Client(api_key=...)` then `client.models.generate_content(model=..., contents=..., config=...)`
  - `system_instruction` goes in `GenerateContentConfig`, NOT inline
  - DEFAULT_MODEL: gemini-3.5-flash, GENERATION_TEMPERATURE: 0.3
  (gemini-2.0-flash shut down June 1 2026; gemini-2.5-flash-lite shuts down
  July 22 2026; gemini-3.5-flash has no announced shutdown date as of June 2026)
  - **Free tier quota (20 requests/day) was repeatedly exhausted during
    dev testing** — expect this to bite during the real pilot too
    unless the project is moved to a paid tier before launch.
- NLLB-200 distilled (facebook/nllb-200-distilled-600M) for translation
  — BUILT and wired into chat_handler.py (bidirectional, before
  escalation/classification and after generation)
- DistilBERT intent classifier (models/intent_classifier/) — TRAINED
  and copied down from Colab, ACTIVE (not the empty-directory fallback
  state described in older versions of this doc)
- thefuzz (rapidfuzz backend) — fuzzy LGA-name matching for facility lookup
- WhatsApp via Twilio REST API using httpx directly (NO Twilio SDK)
- LangChain for document loading and chunking
- huggingface_hub — pilot log persistence to a private HF Dataset repo
  (Hugging Face Spaces' filesystem is ephemeral; see Deployment below)
- **Deployment**: Docker container on Hugging Face Spaces
  (`abiola114/abiyamo`), source mirrored on GitHub
  (`abiolalawal14/Abiyamo`) — see Deployment section below for the
  two-repo workflow.

---

## Current Build Status

### Phase 1 — COMPLETE

#### Track A (Knowledge Base) — src/knowledge_base/
- pdf_loader.py ✅ — loads PDFs, OCR fallback (pytesseract + pdf2image)
- text_splitter.py ✅ — chunks text (500 chars, 50 overlap, RecursiveCharacterTextSplitter)
- embedder.py ✅ — local all-MiniLM-L6-v2, NO API calls, no rate limits
- build_knowledge_base.py ✅ — full ingestion pipeline, writes to ChromaDB
  - Imports were bare (`from pdf_loader import ...`) which only worked
    when run as a standalone script from inside src/knowledge_base/ —
    broke when main.py imports it as `src.knowledge_base.build_knowledge_base`
    for the rebuild-at-startup path (see Phase 3/main.py below). Fixed
    to relative imports (`from .pdf_loader import ...`).
- manage_documents.py ✅ — list/remove/check documents in collection
- watch_and_ingest.py ✅ — scheduled auto-ingestion with ingestion_log.json

ChromaDB collection: "Abiyamo"
- 1848 chunks (FMOH Manual: 1711, 12 Questions SRH doc: 137) — confirmed
  present and correct via live query (`collection.count()`)
- MUST use cosine distance: metadata={"hnsw:space": "cosine"}
- Was rebuilt from L2 to cosine after testing revealed L2 was ChromaDB's
  default and gave weaker retrieval for this embedding model
- Every get_collection() call across ALL files must include this metadata
- **data/chroma_db/ is NOT committed to git** — chroma.sqlite3 exceeds
  Hugging Face's 10MB non-LFS file limit. Instead, main.py's lifespan
  rebuilds the collection from data/raw_pdfs/ on startup if it's empty
  (see main.py below). data/raw_pdfs/ (the two source PDFs, ~8MB) IS
  committed so this rebuild has something to work from.

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

### Phase 2 — COMPLETE, model copied down and ACTIVE
- notebooks/phase2_baseline_models.ipynb ✅ — TF-IDF + Logistic Regression /
  SVM / Random Forest baselines, run on Colab (CPU is fine, <5 min).
- notebooks/phase2_intent_classifier.ipynb ✅ — fine-tunes
  distilbert-base-uncased for 3-label multi-label classification
  (educational/diagnostic/crisis), run on Colab GPU. Saves trained model +
  tokenizer + classifier_config.json to models/intent_classifier/.
  - Colab gotcha (hit and fixed twice): the clone-repo cell force-syncs
    to origin (`git fetch` + `reset --hard`) instead of `git pull`,
    because the artifact-saving cell commits inside the Colab clone and
    its unauthenticated `git push` fails silently, causing "divergent
    branches" on the next run otherwise.
  - Colab gotcha: cell 4 installs transformers/accelerate with `-U`
    (not pinned) — pinning transformers to an old version alone breaks
    Trainer with `ImportError: cannot import name 'EncoderDecoderCache'`
    because Colab's preinstalled accelerate expects a newer transformers API.
- **models/intent_classifier/ IS NOW POPULATED** (config.json, model
  weights ~267MB, tokenizer files, classifier_config.json — trained
  2026-07-04, dataset_size=251, crisis_recall≈0.82). This directory is
  too large for GitHub's 100MB non-LFS limit, so it's **only pushed to
  the Hugging Face Space** (which has LFS configured), not to the
  GitHub mirror — GitHub's srh-chatbot repo leaves it untracked.
  - **classifier_config.json's actual trained `crisis_threshold` is
    ≈0.4999997 (effectively 0.5), NOT the 0.35 originally planned.**
    Whatever the training run actually produced is what's loaded at
    runtime — the code reads this value from the file, it's not
    hardcoded, so this is just a fact to be aware of, not a bug to fix.
  - **The classifier alone is not fully trustworthy near its decision
    boundary.** Live testing repeatedly found short/ambiguous text
    scoring right at ~0.5 regardless of actual meaning (abortion
    topic-vs-personal framing, "Kini Atosi" mistranslated to "What is
    the Root", "What are the symptoms of STIs?" scoring
    diagnostic=0.528). escalation.py now treats the classifier as a
    last-resort fallback behind several deterministic layers — see
    Phase 3 below.

---

### Phase 3 — COMPLETE, extensively hardened by live WhatsApp testing

#### src/safety/escalation.py ✅
Detection order (each step only runs if the previous one found nothing):
1. **CRISIS_PHRASES / DIAGNOSTIC_PHRASES keyword lists** — checked
   FIRST, unconditionally, for every message regardless of length. Not
   a classifier-unavailable fallback; the classifier's boundary
   unreliability (above) made deterministic phrases the primary
   detection layer, with the classifier as backup.
   - CRISIS_PHRASES covers: self-harm, rape/sexual assault, coercion/
     non-consent, physical abuse, trafficking/exploitation, vague
     first-disclosures ("something happened to me").
   - DIAGNOSTIC_PHRASES covers: pregnancy concerns, personal STI/genital
     symptoms, HIV/STI personal concern, post-exposure/emergency
     contraception, abortion **in personal-distress framing only**
     ("i need an abortion" / "i had an abortion" — general topic
     questions like "what is abortion?" must NOT match these and stay
     educational), facility-seeking ("which phc", "nearest clinic",
     etc. — routes to escalation so the user gets real facility names
     instead of a RAG answer with no facility data), personal medical
     follow-up ("following up with my doctor about my discharge" —
     distinct from generic "follow up on <topic>", see below), and a
     handful of newly-added specific phrases ("i need help urgently",
     "i need medical help", "it hurts when i urinate", "it hurts inside").
2. **SAFE_EDUCATIONAL_TERMS** — checked next, unconditionally. Terms
   that force educational (None) UNLESS a crisis/diagnostic phrase
   already matched in step 1. Currently: "masturbation" (classifier
   misclassified a mistranslated Yoruba query about it), "follow up" /
   "follow-up" (generic "give me more info", not a personal concern —
   genuinely personal follow-ups are caught by the specific phrases in
   step 1 first).
3. **Classifier fallback** — only reached if steps 1-2 found nothing.
   - For messages **≥10 words OR containing an explicit danger signal**
     ("rape", "assault", "forced", "abuse", "pregnant and scared",
     "help me urgently"): classifier used normally (crisis at the
     configured threshold, diagnostic at 0.5).
   - For **short messages (<10 words) with no danger signal**: the
     classifier is trusted ONLY for a crisis call, and only above a
     RAISED bar (0.6, not the normal threshold) — diagnostic-via-
     classifier is not considered at all for such messages. This fixed
     a real, previously-failing case ("What are the symptoms of STIs?"
     scoring diagnostic=0.528) with zero regressions, because it never
     touches the keyword-phrase step above.
   - **This gate is deliberately scoped to the classifier fallback
     ONLY, never to keyword-phrase matching.** An earlier draft would
     have also skipped keyword phrases for short messages, which would
     have broken deterministic detection of things like "I want to
     kill myself" (5 words, contains none of the 6 danger-signal
     words) — DO NOT extend this gate to cover keyword matching.
- BYPASSES LLM entirely regardless of detection path — returns
  scripted text + helpline numbers only.
- `get_helplines_for_state(state, lga=None)` — **fuzzy LGA matching**
  via `thefuzz.fuzz.partial_ratio` (threshold 70), ranking facilities
  by closeness to the user's onboarding-captured LGA rather than
  requiring an exact string match (users rarely type the facility
  dataset's exact spelling — "Ibadan South West" vs "Ibadan S/W").
  - **Known limitation**: `partial_ratio` can score unrelated LGAs
    highly if they share characters/prefixes — e.g. "Atiba" scores 75
    against "Ibadan South West" (above the 70 threshold) purely from
    coincidental character overlap. `fuzz.ratio`/`token_sort_ratio`
    don't fully fix this either (Ibadan-variant LGAs score 90+ against
    each other on those too). Accepted because it only affects which
    facility is listed FIRST, never which are hidden.
  - **Related data-coverage gap**: only 5 of each state's LGAs are
    represented in helplines.json (see scripts/import_facilities.py
    below) — a real user's LGA is very often simply not in the data at
    all, in which case fuzzy matching just falls back to ranking
    whatever unrelated LGAs ARE present.
- data/helplines.json ✅ BUILT
  - Placeholder structure: national crisis, sexual_abuse, general_health categories
  - All numbers marked PLACEHOLDER_NOT_VERIFIED except 112 (Nigeria emergency)
  - **DO NOT deploy to real users until every number is personally verified active**

#### scripts/import_facilities.py ✅
- Reads data/raw_queries/facilities.csv (~8,464 rows), writes
  data/helplines.json's "states" section (37 states, 5 facilities each).
- Selection algorithm: **one facility per LGA** (the highest-priority
  facility present in that LGA — MCH > PHCC > PHC > Other), LGAs
  visited alphabetically, capped at 5 per state.
  - Originally filled the 5-per-state quota tier-by-tier across the
    WHOLE state with no regard for LGA, which meant a state's raw data
    being sorted/grouped by LGA could put all 5 selected facilities in
    a single LGA (e.g. every Oyo facility was from Afijo LGA only).
    Fixed to the one-per-LGA algorithm above.
  - **Side effect**: since LGAs are visited alphabetically and capped
    at 5, states with LGAs that sort late alphabetically (e.g. Oyo's
    five "Ibadan..." LGAs) may have NO representation at all — this is
    the data-coverage gap referenced above, not yet resolved.
  - Re-run with: `python scripts/import_facilities.py`

#### src/rag_pipeline/
- retriever.py ✅ BUILT — queries ChromaDB, returns top 3 chunks
  (text/source/chunk_index/distance)
- translator.py ✅ BUILT and WIRED (no longer a TODO)
  - NLLB-200 distilled 600M, bidirectional (English <-> Hausa/Yoruba/Igbo)
  - Language codes: eng_Latn, hau_Latn, yor_Latn, ibo_Latn
  - Public interface: to_english(text, source_lang) / from_english(text, target_lang)
  - `source_lang`/`target_lang` MUST be BCP-47 codes ("yo"), NOT the
    plain names stored in user_sessions.json ("yoruba") — passing the
    plain name silently no-ops (LANGUAGE_CODES.get("yoruba") is None).
    chat_handler.py maps plain-name → BCP-47 via `_PLAIN_TO_BCP47`
    before ever calling these.
  - Model cached at module level (_model_cache / _tokenizer_cache pattern)
  - Falls back to original text on error — pipeline never crashes due to
    a translation failure
  - **_KNOWN_TRANSLATION_OVERRIDES**: a small, explicitly-flagged
    glossary of exact phrases NLLB mistranslates badly, intercepted
    BEFORE calling NLLB. Currently only covers "atosi"/"kini atosi"
    (Yoruba slang for masturbation, which NLLB mistranslated to "What
    is the Root" — nonsense that then got misclassified downstream).
    Needs a native speaker to identify and add other gaps; not
    intended to be a general solution.
  - Wired into chat_handler.py: to_english() runs BEFORE the escalation
    check (not just before RAG retrieval) — a real safety-critical
    ordering fix, since a non-English crisis message would never be
    recognised as one otherwise.
- prompt_builder.py ✅ BUILT
  - `build_prompt(question, chunks, include_sources=False, last_messages=None)`
  - `last_messages`: optional list of `{"role": "user"|"assistant", "content": str}`
    dicts (see chat_handler.py's conversation memory below) — when
    non-empty, prepends a "Previous conversation" block before the
    retrieved context so Gemini can resolve short follow-up replies
    ("yes", "tell me more") using real context. `None`/empty leaves
    the prompt exactly as before (backward compatible).
  - get_system_prompt() exposed separately for Gemini's system_instruction field
  - MAX_CHUNK_CHARS=600
  - SYSTEM_PROMPT now also instructs: responses under 250 words in
    short WhatsApp-friendly paragraphs; explicit fallback text when
    retrieved context doesn't answer the question (prevents
    hallucination on vague queries); never open with "Hello"/"Hi".
- generator.py ✅ BUILT
  - Uses google-genai v2.8.0 (new SDK), DEFAULT_MODEL=gemini-3.5-flash
  - `answer_from_chunks(question, chunks, last_messages=None)` threads
    conversation history through to build_prompt()
  - _client_cache pattern
  - Returns FALLBACK_RESPONSE string on any API error — never raises to
    caller. chat_handler.py checks for this exact string to avoid
    storing a failed generation as if it were real conversation memory.
  - **Retry/backoff for 429 rate-limit errors is still NOT implemented**
    — see What To Build Next.

#### src/whatsapp/
- sender.py ✅ BUILT
  - send_whatsapp_message(to, body) POSTs to Twilio REST API via httpx
  - Truncation is now **sentence-aware**: `_truncate_at_sentence()` cuts
    at the last complete `.`/`!`/`?` before MAX_WHATSAPP_CHARS (1500),
    appending "(Reply 'more' if you would like me to continue)" —
    replaced a crude `body[:1497] + "..."` that could cut mid-word.
  - Requires in .env: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
- webhook.py ✅ BUILT
  - FastAPI APIRouter, POST /whatsapp/incoming
  - Receives Twilio form-POST, returns empty TwiML IMMEDIATELY
  - Processes in BackgroundTask via chat_handler.handle_message(user_message, user_id=user_number)
  - Sends reply via sender.py AFTER pipeline completes
  - Mounted in main.py at prefix /whatsapp
  - BackgroundTasks used specifically to avoid Twilio's 15-second timeout
  - **Not confirmed tested end-to-end with a real Twilio number/phone**
    — all live testing this far has driven `handle_message()` directly
    in Python, not through an actual WhatsApp message round-trip.

#### src/api/chat_handler.py ✅ — the single pipeline entry point, substantially rebuilt
`handle_message(message, language="en", user_id=None)` — routing order:

1. **Session reset** — trigger phrases ("reset", "start over", "/start",
   etc., case-insensitive, punctuation-stripped). Deletes the session
   entry and immediately re-primes a pending onboarding entry (same
   shape as a brand-new user) so the next message resumes onboarding
   correctly instead of producing a second welcome message.
2. **Language switching** — a menu trigger ("language", "switch
   language"), a direct phrase ("speak yoruba", "change to hausa"), or
   (only once the menu was shown, or for a fully-onboarded user) a bare
   1-4 digit reply. Updates ONLY the language field — never resets
   saved state/LGA. Confirmation messages are in the NEW language
   (Hausa/Igbo wording not yet native-speaker-verified — flagged in
   code).
3. **Onboarding** — now THREE steps: language (1-4) → state (free
   text) → LGA (free text, or "skip"). Only after LGA is captured does
   `onboarded` become true. Session schema:
   ```json
   {
     "language": "english", "state": "Oyo", "lga": "Ibadan North",
     "last_messages": [...], "onboarded": true
   }
   ```
4. **Translate to English** (BEFORE classification/retrieval — see
   translator.py note above).
5. **Escalation check** — should_escalate() / get_escalation_response(),
   now passed the session's `state` AND `lga` for facility prioritisation.
6. **RAG pipeline** — retrieve_relevant_chunks() → answer_from_chunks()
   (with `last_messages` from the session) → translate back if needed.
   **Conversation memory update** happens here: appends the exchange,
   keeps only the last 2 exchanges (4 messages), and is skipped
   entirely for escalation responses and failed generations
   (FALLBACK_RESPONSE) — only genuine educational exchanges are stored.

**Removed**: the old `_is_follow_up_reply()` keyword-based redirect
("yes"/"continue"/"more" → a fixed "please be more specific" message).
Replaced entirely by conversation memory (step 6) — Gemini now
resolves these naturally using real prior context instead of a
scripted deflection.

- main.py ✅ BUILT
  - FastAPI app with lifespan startup
  - Startup pre-loads embedding model, logs ChromaDB chunk count
  - **Rebuild-at-startup**: if the ChromaDB collection is empty (fresh
    ephemeral filesystem, see Deployment below), calls
    `add_documents_from_folder("data/raw_pdfs")` to rebuild it from the
    committed source PDFs before serving requests. Uses OCR (slow,
    several minutes) unless `pypdf` is installed, in which case normal
    text extraction is used instead (`pypdf` is now in requirements.txt).
  - Endpoints: GET /health, POST /chat
  - Mounts WhatsApp router
  - Logs every /chat call via evaluation/logger.py
  - Run: uvicorn src.api.main:app --reload --port 8000

#### evaluation/
- logger.py ✅ BUILT + **now persists across Hugging Face Space restarts**
  - log_interaction(user_id, message, result, response_time_ms, channel)
  - Appends to data/pilot_logs.jsonl (format UNCHANGED)
  - Phone numbers SHA-256 hashed (16 hex chars) before writing
  - **New**: after every local append, also pushes the full log file to
    a private Hugging Face Dataset repo (`HF_LOGS_DATASET_REPO` env var,
    reuses the `HF_TOKEN` secret already needed elsewhere — no new
    credential). On a fresh process (local file missing), restores from
    the dataset repo first so history survives a Space restart. Silently
    no-ops if `HF_LOGS_DATASET_REPO`/`HF_TOKEN` aren't set — local-only
    logging is unaffected.
  - **Chosen over Google Sheets (needs a new service-account credential)
    or periodic email (needs a new email credential, and batching
    reintroduces a loss window if the process restarts mid-batch)** —
    see logger.py's own docstring for the full tradeoff writeup.
  - Live-tested end-to-end against the real `abiola114/abiyamo-pilot-logs`
    dataset repo (push + restore-after-simulated-restart both verified).
  - **`HF_LOGS_DATASET_REPO` must be added as a Hugging Face Space
    secret** (alongside the existing `HF_TOKEN`) for this to activate in
    production — not yet confirmed done.
  - read_logs() and count_logs() also trigger the restore-from-remote
    check, so they reflect prior history even right after a restart.
- pilot_report.py ✅ BUILT
  - 7-section terminal report using rich + pandas
  - Sections: overview, language distribution, escalation rate, chunk stats,
    response time percentiles, 14-day daily volume bar chart, last 10 interactions
  - Generates synthetic preview if no log exists
  - Run: python -m evaluation.pilot_report

---

### Phase 4 — NOT STARTED
Pilot evaluation itself hasn't begun. Infrastructure is otherwise
ready (translation wired, classifier active, logs persist across
restarts) — see What To Build Next for what's actually blocking it.

---

## Deployment

Two git remotes, kept in sync manually (no CI):
- **GitHub** (`abiolalawal14/Abiyamo`, branch `master`) — source of
  record, `srh-chatbot` local checkout.
- **Hugging Face Space** (`abiola114/abiyamo`, SDK: docker, branch
  `main`) — separate local checkout at `../abiyamo`, actually serves
  the live app. Workflow after making changes in `srh-chatbot`: copy
  the touched files into the `abiyamo` checkout, commit and push both
  remotes separately (they have unrelated/diverged histories in
  places — GitHub also received unrelated Colab-notebook commits
  directly).
- Dockerfile (root of the HF Space repo): `python:3.10-slim`, installs
  `tesseract-ocr`/`poppler-utils` (OCR fallback), `pip install -r
  requirements.txt`, runs `uvicorn src.api.main:app --host 0.0.0.0 --port 7860`.
- **models/intent_classifier/ (255MB) is pushed to the HF Space only**
  (which has Git LFS) — GitHub's 100MB non-LFS limit means it's left
  untracked in the GitHub mirror.
- **Required Hugging Face Space secrets**: `GEMINI_API_KEY`,
  `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM`,
  `HF_TOKEN` (already present, also used for the log-persistence
  feature), and `HF_LOGS_DATASET_REPO` (NOT yet confirmed added — see
  evaluation/logger.py above).

---

## What To Build Next (in order)

1. **Confirm `HF_LOGS_DATASET_REPO` is set as a Space secret** — the
   log-persistence code is built and live-tested, but won't activate
   in production until this secret exists on the Space.
2. **Retry/backoff logic in generator.py** — still returns
   FALLBACK_RESPONSE immediately on any Gemini error, including 429
   rate limits. The free tier's 20/day cap was hit repeatedly during
   dev testing; a real pilot will hit it too.
3. **Test the Twilio WhatsApp webhook end-to-end** with a real phone
   number — every fix this far has been verified by calling
   `handle_message()` directly, not through an actual inbound WhatsApp
   message via Twilio's webhook.
4. **Personally verify every helplines.json phone number/facility**
   before real users see them (placeholder safety check already
   prevents deploying unverified numbers, but the data itself still
   needs manual verification).
5. **Improve LGA coverage in helplines.json** — currently only 5 of
   each state's (up to 33+) LGAs are represented; consider raising
   `MAX_FACILITIES_PER_STATE` or accepting more per state to reduce how
   often a user's real LGA has zero facilities to fuzzy-match against.
6. **Native-speaker review** of the Hausa/Igbo language-switch
   confirmation messages, and expansion of translator.py's
   `_KNOWN_TRANSLATION_OVERRIDES` glossary for other slang terms NLLB
   mistranslates.
7. **Phase 4 pilot evaluation** — once the above are addressed.

---

## Credentials Needed Before Live Deployment

- GEMINI_API_KEY — key in .env is ACTIVE. Free tier (20 req/day) was
  repeatedly exhausted during dev testing this session — budget for
  this in the pilot or upgrade the tier.
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM — values
  are in .env; webhook code is built but **not confirmed tested
  end-to-end** with a real number (see What To Build Next).
- HF_TOKEN — already used for Space deployment; ALSO now needed for
  pilot-log persistence (evaluation/logger.py).
- HF_LOGS_DATASET_REPO — new, NOT yet confirmed set as a Space secret.
  Without it, logs work locally but are lost on every Space restart.
- data/helplines.json — ALL numbers are PLACEHOLDER_NOT_VERIFIED.
  Must be personally verified before any real user receives them.

## Environment / Dependencies

- **Two different Python versions matter here**: dev machine runs
  Python 3.12.3 (Anaconda base environment); the deployed Docker image
  runs Python 3.10 (`python:3.10-slim`). Several dependency pins in
  requirements.txt were chosen specifically for 3.10 compatibility and
  are OLDER than what works fine on the 3.12 dev machine — don't
  "upgrade" them without re-checking 3.10 compatibility:
  - numpy==2.2.6, pandas==2.2.3, scipy==1.15.3 (latest that supports
    3.10 — 1.16+ requires 3.11), scikit-learn==1.7.2 (latest that
    supports 3.10 — 1.8+ requires 3.11)
- pyarrow, numexpr, bottleneck still emit NumPy 1.x warnings on import
  but these are non-fatal — pandas and the pipeline work correctly
- requirements.txt additions this round: pypdf (fast PDF text
  extraction, avoids the slow OCR path), python-multipart (Twilio
  form-POST parsing), huggingface_hub (log persistence), thefuzz
  (fuzzy LGA matching)
- Server start command: python -m uvicorn src.api.main:app --port 8000
  (model load takes ~30-60s on CPU; first-ever startup with an empty
  ChromaDB collection also triggers the PDF rebuild, which can take
  several minutes if OCR is used instead of pypdf)

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

4. **Translation is two-directional, and happens BEFORE escalation**:
   incoming message → English (before classification AND retrieval) →
   ... → answer → user's language. Doing escalation detection on
   untranslated text means a real crisis message in Hausa/Yoruba/Igbo
   would never be recognised as one — this was an actual bug, now fixed.

5. **NLLB-200 distilled 600M**: not full model. Reason: no GPU on dev machine.
   Same language coverage, modestly reduced quality on complex text —
   known to mistranslate some slang (see _KNOWN_TRANSLATION_OVERRIDES).

6. **Gemini SDK is google-genai v2.8.0**: NOT google-generativeai (old SDK).
   These are different packages with different APIs. Do not confuse them.

7. **WhatsApp uses httpx directly**: no Twilio Python SDK. httpx was already
   a project dependency.

8. **Twilio webhook uses BackgroundTasks**: empty TwiML returned immediately,
   reply sent after pipeline. Avoids Twilio's 15-second response timeout.

9. **helplines.json placeholder safety**: is_placeholder() check in
   escalation.py prevents deploying unverified numbers. DO NOT remove this check.

10. **Escalation keyword phrases run unconditionally, BEFORE the
    classifier, for every message** — not just as a classifier-
    unavailable fallback. The classifier's confidence sits right at
    ~0.5 for a recurring set of ambiguous topics (abortion, STI
    symptoms, translated slang). Deterministic phrases are the primary
    safety layer; the classifier is backup, not the other way around.

11. **The short-message classifier gate (escalation.py) applies ONLY to
    the classifier fallback step, NEVER to keyword-phrase matching.**
    Do not extend it to skip CRISIS_PHRASES/DIAGNOSTIC_PHRASES for
    short messages — that would silently downgrade genuine short crisis
    disclosures ("I want to kill myself" is 5 words) to needing a
    raised classifier bar instead of deterministic detection. This was
    caught and deliberately avoided; see escalation.py's own comments.

12. **Conversation memory (chat_handler.py) only stores genuine
    educational exchanges** — never escalation responses (scripted
    safety text, not real conversation) and never a failed generation
    (FALLBACK_RESPONSE would otherwise get replayed to Gemini next turn
    as if it were useful context).

13. **data/chroma_db/ and models/intent_classifier/ are not both
    committed to the same remote** — chroma_db is rebuilt at startup
    instead of committed at all (10MB HF limit); the classifier is
    LFS-pushed to the HF Space only, not GitHub (100MB GitHub non-LFS
    limit). Don't assume either is present just because the other is.

---

## Folder Structure

```
srh-chatbot/
├── Dockerfile                  (python:3.10-slim, HF Spaces deployment)
├── data/
│   ├── raw_pdfs/               (source PDFs, COMMITTED — needed for
│   │                           startup rebuild since chroma_db isn't)
│   ├── raw_queries/
│   │   ├── facilities.csv      (~8,464 Nigerian health facility rows)
│   │   └── ...                 (annotator Label Studio exports)
│   ├── chroma_db/              (ChromaDB persistent storage — NOT
│   │                           committed, rebuilt at startup if empty)
│   ├── helplines.json          (PLACEHOLDER — verify all numbers before use;
│   │                           regenerated by scripts/import_facilities.py)
│   ├── user_sessions.json      (onboarding/language/state/LGA/conversation
│   │                           memory per user, hashed phone number keys —
│   │                           NOT committed, runtime state)
│   ├── ingestion_log.json      (tracks processed PDFs for watch_and_ingest)
│   └── pilot_logs.jsonl        (anonymised interaction logs — mirrored to
│                                a private HF Dataset repo, see logger.py)
├── scripts/
│   └── import_facilities.py    (rebuilds helplines.json from facilities.csv)
├── src/
│   ├── knowledge_base/         (Phase 1 Track A — all built ✅)
│   ├── annotation/             (Phase 1 Track B — COMPLETE ✅)
│   ├── safety/                 (escalation.py — keyword-first detection,
│   │                           fuzzy LGA matching, short-message gate)
│   ├── rag_pipeline/           (retriever, prompt_builder, generator,
│   │                           translator — all wired, incl. conversation
│   │                           memory + translation overrides)
│   ├── whatsapp/               (sender.py, webhook.py — webhook not yet
│   │                           end-to-end tested with a real number)
│   └── api/                    (main.py — startup rebuild logic;
│                                chat_handler.py — reset/language-switch/
│                                onboarding/conversation-memory routing)
├── models/
│   ├── baselines/               (Phase 2 — TF-IDF baseline models + vectorizer)
│   └── intent_classifier/       (POPULATED and ACTIVE — trained 2026-07-04;
│                                pushed to the HF Space only, not GitHub)
├── notebooks/
│   ├── phase2_baseline_models.ipynb     ✅
│   └── phase2_intent_classifier.ipynb   ✅
├── evaluation/
│   ├── logger.py               (now persists to a private HF Dataset repo)
│   └── pilot_report.py         ✅
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
- Windows console (cp1252) can't print some Yoruba/Hausa/Igbo characters
  or emoji — test blocks that print non-ASCII output encode defensively
  (`.encode("ascii", errors="backslashreplace").decode("ascii")`) rather
  than crashing; this is a terminal display limitation, not a pipeline bug
