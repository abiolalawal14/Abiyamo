# SRH Chatbot — Project Context for Claude Code

## What This Project Is
A RAG-based multilingual chatbot for adolescent sexual and reproductive
health (SRH) information in Nigeria. Target users: ages 16-24.
Languages: English, Hausa, Yoruba, Igbo.
MSc AI/ML dissertation project — Miva Open University.

**Status: deployed and live-tested, ethical clearance obtained,
actively prepping for the real pilot.** The pipeline is fully wired
(translation, escalation, RAG, onboarding, conversation memory) and
running on Hugging Face Spaces. A real end-to-end WhatsApp test
(2026-08-13, via Twilio sandbox) confirmed onboarding, RAG answers,
and crisis escalation all work through an actual phone — but also
surfaced two real issues, both still OPEN, see "Live WhatsApp Pilot
Test Findings (2026-08-13)" under Phase 3 below:
1. A knowledge-base content gap — no chunk directly defines "what is
   pregnancy" as a basic question.
2. A safety gap — once should_escalate() fires, the very next message
   in that same conversation is evaluated with zero awareness a crisis
   was just disclosed, and can fall straight through to the ordinary
   RAG/generation pipeline. Needs a design decision (time-boxed vs.
   turn-count vs. sticky crisis mode) before it's fixed — not yet
   decided.

See "Pilot Readiness Checklist" below for the full punch list being
worked through before real participants are messaged.

---

## Pilot Readiness Checklist (as of 2026-08-13)

Six things identified as needed before real participants are messaged,
tracked here so progress survives across sessions:

1. **Facility/helpline data** — ✅ MOSTLY DONE. Full LGA coverage fixed
   (see Phase 3 below); candidate numbers for the four crisis_fallback
   hotlines (SURPIN, MANI, NAPTIP, WARIF) found via web research and
   wired in behind a real is_placeholder() gate. **Still open: none of
   the four numbers have been phone-verified** — call each one,
   confirm it's live, then flip `"verified": true` in
   data/helplines.json (or ask Claude to do the flip once you've
   confirmed which numbers are correct).
2. **Twilio number** — ✅ DECIDED: sandbox, not production (Meta
   business verification isn't worth it for a short academic pilot).
   Webhook already configured by the user pointing at the deployed
   Space's `/whatsapp/incoming`.
3. **Gemini quota** — 🔲 OPEN. Free tier (20 req/day) not yet moved to
   a paid tier. Will not survive real pilot traffic.
4. **Real end-to-end WhatsApp test** — ✅ DONE 2026-08-13 via Twilio
   sandbox. Onboarding, RAG answers, and crisis escalation all
   confirmed working through an actual phone. Surfaced the two new
   findings above (see Phase 3 → "Live WhatsApp Pilot Test Findings").
5. **HF_LOGS_DATASET_REPO secret** — 🔲 OPEN, not yet confirmed set.
   Check: Space Settings → Variables and secrets for the key, or
   trigger one real interaction and check the Space's Logs tab for a
   `[logger]`-prefixed line (its absence means the secret isn't set
   and logs won't survive a restart).
6. **Consent/monitoring plan** — ✅ Consent form and participant
   invitation already exist from the ethics application (outside this
   codebase, not something this project needs to build). **Still
   open**: whether the consent/data-use statement needs to also appear
   inside the WhatsApp chat itself (onboarding currently doesn't
   mention that messages are logged and reviewed — see
   evaluation/logger.py's own docstring), and who is actually watching
   for live crisis disclosures during the pilot window and how
   promptly — not yet confirmed by the user.

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
     first-disclosures ("something happened to me"). **Extended
     2026-08-15** with "forced himself/herself on me" and "i didn't
     consent"/"didn't consent to what happened" — found via an honest
     unseen-phrasing test probe (not asserted against invented
     expectations, see escalation.py TEST 8) that showed these common,
     real disclosure constructions matched no existing phrase (only
     "forcing himself" present-continuous and "didn't agree to" did).
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
   - **Extended 2026-08-15** with definitive (non-hedged) pregnancy
     disclosure phrases ("i am pregnant", "i'm pregnant", "just found
     out i'm pregnant", "i tested positive for pregnancy", etc.) — live
     testing found "I am pregnant, what can I do?" matched NONE of the
     existing phrases (all hedged: "i think i'm pregnant", "i might be
     pregnant"), fell through to the classifier, got excluded by the
     short-message gate (below), and reached RAG untouched, where a
     known KB content gap ("what is pregnancy" has no defining chunk —
     see "What To Build Next" below) produced the generic "Here are
     some topics" fallback baked into prompt_builder.py's SYSTEM_PROMPT.
     **Accepted tradeoff, documented in-code**: the bare "i am
     pregnant"/"i'm pregnant" phrases also match inside hypothetical
     framing ("What should I do IF I'm pregnant?") — the same
     false-positive class already excluded for abortion above, but with
     no exclusion mechanism available here. Kept per this file's own
     "false positive is far less harmful than false negative" principle
     (Critical Design Decision #10) — see escalation.py's
     DIAGNOSTIC_PHRASES comment and the explicit "if I'm pregnant" test
     case documenting this as a visible, intentional choice.
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
   - **CONSIDERED AND REJECTED 2026-08-15, with real evidence, not just
     left alone**: raising this gate to also permit diagnostic-via-
     classifier for short messages (at a stricter bar), to generalize
     detection beyond DIAGNOSTIC_PHRASES for unseen short phrasings.
     Probed the live trained model directly before deciding: diagnostic
     scores for a mix of genuine and false-positive short messages all
     clustered in a ~0.46-0.55 band with no separating threshold —
     e.g. "I have vaginal discharge" (genuine) scored 0.541, "My
     breasts feel sore and tender" (genuine) scored 0.548, but the
     known false positive "What are the symptoms of STIs?" scored 0.528
     — BETWEEN the two genuine cases. No threshold excludes the false
     positive without also excluding real diagnostic disclosures. This
     confirms the classifier genuinely does not discriminate in the
     short-message regime (matches the ~0.5-boundary unreliability
     documented in Phase 2 above), not that 0.65 (the originally
     proposed number) was simply untuned. See the comment block above
     `_SHORT_MESSAGE_CRISIS_PROBABILITY_THRESHOLD` in escalation.py for
     the full probe data. Real generalization for short diagnostic
     messages, if ever pursued, needs a different approach entirely
     (e.g. an LLM-based intent check, or retraining on more short-text
     examples) — not a threshold tweak.
   - **This gate is deliberately scoped to the classifier fallback
     ONLY, never to keyword-phrase matching.** An earlier draft would
     have also skipped keyword phrases for short messages, which would
     have broken deterministic detection of things like "I want to
     kill myself" (5 words, contains none of the 6 danger-signal
     words) — DO NOT extend this gate to cover keyword matching.
- BYPASSES LLM entirely regardless of detection path — returns
  scripted text + helpline numbers only.
- `get_helplines_for_state(state, lga=None)` — thin wrapper around
  `_resolve_lga_scope()` (below), kept for backward compatibility with
  scripts/import_facilities.py and this module's own tests, which only
  need the bare facility list.
- `_resolve_lga_scope(state, lga)` ✅ BUILT 2026-08-15 — returns
  `(facilities, matched_lga)`. **HARD LGA FILTER, replacing the old
  fuzzy-rank-then-pad behaviour**, fixed after live user testing
  reported a Bwari/FCT crisis disclosure returning Bwari + Abaji +
  Kwali facilities together.
  - **Root cause**: the old `get_helplines_for_state()` returned
    `matching + other` — ALL of a state's facilities, just re-sorted so
    LGA matches came first. `_crisis_response()`/`_diagnostic_response()`
    then sliced `facilities[:3]`. Since scripts/import_facilities.py
    deliberately selects exactly ONE facility per LGA, every LGA has
    exactly 1 facility — so ANY time an LGA matched, the `[:3]` slice
    was **guaranteed** to pad with 2 facilities from unrelated LGAs.
    100% reproducible on every escalation response with a resolved LGA,
    not a fuzzy-matching edge case.
  - **Fix**: exact match (case-insensitive, stripped) short-circuits
    fuzzy scoring entirely — this isn't just an optimisation.
    `fuzz.partial_ratio` scores a pure substring match as 100, so a
    query of "Ibadan North" scores 100 against BOTH the exact LGA
    "Ibadan North" AND the two real, distinct LGAs "Ibadan North East"/
    "Ibadan North West" (Oyo has all three) — without checking for an
    exact match first, which one wins depends on incidental list order,
    not correctness. When no exact match exists, fuzzy-score every
    facility and take ONLY the facilities sharing the single
    best-scoring LGA's exact name (never a second, different LGA that
    also happened to clear `_FUZZY_LGA_MATCH_THRESHOLD`, e.g. "Ibadan
    South West" (83) blending in alongside a genuine top match). Falls
    back to the full state list (`matched_lga=None`) only when nothing
    clears the threshold at all.
  - `_crisis_response()`/`_diagnostic_response()` now vary their intro
    sentence based on `matched_lga`: "Please visit this health facility
    in {LGA} LGA, {state}..." (singular, since the hard filter almost
    always yields exactly 1 facility) vs. the old "one of these health
    facilities in {state}..." wording for the genuine state-wide
    fallback case. This also fixes a related complaint (sexual-violence
    responses "overwhelming" the user with multiple facilities) as a
    side effect — no separate crisis-specific logic needed.
  - Confirmed via 12 new LGA-isolation tests (escalation.py `__main__`
    TEST 7): 5 states' exact matches, capitalization/suffix/misspelling
    variants, the Ibadan-collision case specifically, a near-miss
    misspelling correctly falling back rather than guessing wrong, an
    LGA absent from the dataset, and the no-lga-provided case (protects
    chat_handler.py TEST 17's "Oyo spans >= 3 distinct LGAs" invariant,
    confirmed still passing — that path is untouched by this fix).
  - **Data-coverage gap RESOLVED 2026-08-13**: was "only 5 of each
    state's LGAs represented"; MAX_FACILITIES_PER_STATE raised from 5
    to 50 in scripts/import_facilities.py, all 37 states now have every
    real LGA covered (783 facilities total, up from 185).
- `is_placeholder(entry)` ✅ BUILT 2026-08-13 — the gate this doc has
  described since early Phase 3 but which did NOT actually exist in
  code until now. Returns True if an entry has no number, the number
  is still the literal "PLACEHOLDER_NOT_VERIFIED" sentinel, or
  `verified` is not `True`. `_get_verified_crisis_hotlines()` filters
  `crisis_fallback` entries through it and `_crisis_response()` appends
  any that pass as extra lines after the always-present 112 line.
  Currently returns nothing extra for any real user, since nothing in
  crisis_fallback is verified yet (see helplines.json bullet below) —
  confirmed via a real crisis response before/after check, and via an
  in-memory simulated "verified: true" flip that correctly caused a
  hotline to appear.
- `_format_officer_contact(facility)` ✅ BUILT 2026-08-14 — renders a
  facility's `officer_contact` block (see scripts/
  enrich_facility_contacts.py below) as an extra indented line under
  that facility's name/LGA in `_crisis_response()` and
  `_diagnostic_response()`. Deliberately NOT gated behind an
  `is_placeholder()`-style `verified: true` check the way
  `crisis_fallback` numbers are — see Critical Design Decision #14
  below for why that's a considered choice, not an oversight.
- data/helplines.json ✅ BUILT, coverage fixed 2026-08-13, per-facility
  officer contacts added 2026-08-14 (see below)
  - Placeholder structure: national crisis, sexual_abuse, general_health categories
  - `crisis_fallback` (SURPIN, MANI, NAPTIP, WARIF) now holds real
    candidate numbers found via web research (previously literal
    "PLACEHOLDER_NOT_VERIFIED" text) — but all four are still
    `"verified": false`. **This directory of numbers was ALSO
    previously dead code** — crisis_fallback was written into the JSON
    by import_facilities.py but nothing in src/ ever read it, so even
    a fully verified number would never have reached a user. Both the
    data AND the missing wiring were fixed together 2026-08-13 (see
    is_placeholder() bullet above).
  - 112 (Nigeria emergency) remains the only genuinely verified number.
  - **DO NOT flip any of the four hotlines' `verified` field to `true`
    until it has been personally called and confirmed active** — see
    Pilot Readiness Checklist item 1 above.

#### Live WhatsApp Pilot Test Findings (2026-08-13)
A real end-to-end WhatsApp conversation via Twilio sandbox (onboarding
→ RAG questions → crisis disclosure → follow-ups) surfaced two issues,
both still OPEN:

1. **Knowledge-base content gap — "what is pregnancy" has no good
   answer.** Retrieval for "what is pregnancy" returns chunks about
   abortion and teen-pregnancy statistics (distance 0.43–0.51) — no
   chunk in the source PDFs actually defines pregnancy in basic terms.
   Gemini correctly (per its own SYSTEM_PROMPT anti-hallucination
   instruction in prompt_builder.py) refuses to answer and returns the
   scripted "here are some topics" fallback instead. Confirmed by
   contrast: "Pregnancy and menstruation" (the menu-option wording)
   retrieves genuinely on-topic chunks (distance 0.38–0.43) and answers
   correctly. **Fix is content, not code** — add a basic pregnancy
   definitional passage to the knowledge base source PDFs.
2. **Safety gap — escalation does not persist across a conversation.**
   After "i was raped" correctly escalates, the very next message
   ("who can i call?") does NOT escalate — should_escalate() evaluates
   every message in complete isolation, with no session-level memory
   that this user is mid-crisis. It fell through to the ordinary RAG
   pipeline and Gemini generated an answer from a weakly-matched chunk
   (distance 0.56), with no crisis framing. A second follow-up ("what
   about Naptif") also missed escalation and got the generic "topics"
   menu — the exact wrong response to send someone right after a rape
   disclosure. Root cause is structural: escalation responses are
   deliberately excluded from conversation memory (Critical Design
   Decision #12 below), so there's no signal anywhere that a crisis is
   in progress once the first scripted reply goes out.
   **NOT YET FIXED — needs a design decision first**: should
   heightened-caution mode after an escalation be time-boxed (next N
   minutes), turn-count-boxed (next N messages), or sticky until the
   user clearly asks something unrelated/educational? Discussed with
   the user 2026-08-13, not yet decided.

#### scripts/import_facilities.py ✅
- Reads data/raw_queries/facilities.csv (~8,464 rows), writes
  data/helplines.json's "states" section (37 states, up to
  MAX_FACILITIES_PER_STATE facilities each).
- Selection algorithm: **one facility per LGA** (the highest-priority
  facility present in that LGA — MCH > PHCC > PHC > Other), LGAs
  visited alphabetically.
  - Originally filled the 5-per-state quota tier-by-tier across the
    WHOLE state with no regard for LGA, which meant a state's raw data
    being sorted/grouped by LGA could put all 5 selected facilities in
    a single LGA (e.g. every Oyo facility was from Afijo LGA only).
    Fixed to the one-per-LGA algorithm above.
  - **MAX_FACILITIES_PER_STATE raised from 5 to 50 (2026-08-13)** — the
    old cap of 5 meant most states only got their first 5 LGAs
    alphabetically (every state in the raw dataset has MORE than 5
    distinct LGAs — smallest is FCT with 6, largest is Kano with 44),
    so a real user's LGA was very often simply absent, forcing the
    fuzzy-match fallback to rank unrelated LGAs. 50 is comfortably
    above Kano's 44, so this is now effectively "one facility per LGA,
    every LGA" for all 37 states — 783 facilities total (was 185).
    Confirmed via re-run: all 17 of the module's own escalation tests
    still pass, and the previously-failing Oyo/"Ibadan South West"
    case now returns a genuine Ibadan South West facility.
  - `crisis_fallback`'s four hotline entries (SURPIN, MANI, NAPTIP,
    WARIF) are also defined in this script's `build_helplines()` —
    edit them HERE, not just in the generated JSON, or the next re-run
    will silently overwrite any manual edit back to placeholder text.
  - Re-run with: `.venv/Scripts/python.exe scripts/import_facilities.py`
    (must use the project venv — the system/base Python interpreter
    does not have thefuzz installed and the script's own test suite
    will fail on import).

#### scripts/enrich_facility_contacts.py ✅ BUILT 2026-08-14
- Adds an `officer_contact` block (officer_name, phone, email, as_of,
  source, verified, notes) to facilities already in `data/
  helplines.json`, pulled from `api.nphcda.gov.ng` — an **undocumented**
  API discovered by reading phc.nphcda.gov.ng's own JS bundle's network
  calls, not an official published API. Could change or start requiring
  auth without notice.
- **The phone number is the individual facility officer-in-charge's
  personal mobile from an NPHCDA facility survey, NOT an official
  facility hotline.** Self-reported, not independently phone-verified,
  and can go stale as staff transfer — `as_of` (the survey's own
  `updated_at` date) is carried through specifically so the response
  text can show the user how old the contact is.
- Resolution path per facility: state → LGA → ward → facility record
  (state/LGA/ward IDs resolved via `boundary/states|lgas|wards/`,
  fuzzy-matched with `thefuzz` since the API's admin-name spelling
  doesn't always match ours) → `boundary/facility/<id>/` for the detail
  record. ~37% of facilities in helplines.json have no ward name (blank
  in the source CSV for several states) — for those, all of the LGA's
  wards are scanned (closest-named first, stopping early on a strong
  match) rather than skipped, so coverage isn't limited to states with
  clean ward data.
- Facility-name matching strips generic descriptor words (PHC, MCH,
  Primary, Health, Centre, etc.) from both sides before comparing —
  our dataset abbreviates ("Umunne-Ato Phc"), the API spells out
  ("Umunne-Ato Primary Health Centre"), so comparing raw strings scores
  low even for the same facility. Smoke-tested on Abia (14/15 matched)
  and Akwa Ibom's missing-ward fallback (5/5 matched) before the full
  run — spot-checked matches by hand, not just by score.
- Additive only — never modifies name/lga/ward/type/verified/notes on
  existing facility entries, and `get_helplines_for_state()`/
  `escalation.py`'s existing keys (`f['name']`, `f['lga']`) are
  untouched, so this cannot break anything that was already working.
- Resumable: progress cached to `data/.facility_contact_cache.json`
  (flushed every 10 facilities), keyed by state|lga|name — a re-run
  after an interruption skips facilities already resolved (match or
  confirmed no-match) instead of re-querying the API. An unmatched/
  no-phone report is written to `data/
  facility_contact_enrichment_report.json` for follow-up.
- Rate-limited (`--delay`, default 0.2s between requests) out of
  courtesy to a government server not built for bulk access — a full
  789-facility run is a few thousand requests and takes roughly
  15-35 minutes. `--states`/`--limit` flags exist for testing on a
  subset before a full run.
- Run: `.venv/Scripts/python.exe scripts/enrich_facility_contacts.py`
  (same venv requirement as import_facilities.py — httpx/thefuzz).

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
  - **Confirmed tested end-to-end 2026-08-13** via Twilio sandbox with a
    real phone — onboarding, RAG answers, and crisis escalation all
    worked through an actual WhatsApp round-trip. See "Live WhatsApp
    Pilot Test Findings" above for the two issues that transcript
    surfaced.

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
- **Last deploy: 2026-08-14** — data/helplines.json, src/safety/
  escalation.py, scripts/enrich_facility_contacts.py (new — per-facility
  officer_contact from api.nphcda.gov.ng, 722/783 matched). Pushed to
  GitHub master (0b6164c) and the HF Space main (0a4beb8). Space will
  rebuild ChromaDB from data/raw_pdfs/ on restart per the ephemeral-
  filesystem behaviour described above — check `/health` after a
  deploy before assuming the knowledge base is ready.
  (Previous deploy: 2026-08-13 — data/helplines.json,
  scripts/import_facilities.py, src/safety/escalation.py (LGA coverage
  fix + is_placeholder()/crisis_fallback wiring). GitHub master
  (d5b8f1d), HF Space main (d40318d).)

---

## What To Build Next (in order)

1. **Decide and implement the crisis-follow-up fix** (see "Live
   WhatsApp Pilot Test Findings" above) — this is the highest-priority
   open item, found via real pilot testing, not theoretical. Needs the
   user to decide: time-boxed, turn-count-boxed, or sticky
   heightened-caution window after should_escalate() fires.
   **Closely related, same open decision, found 2026-08-15**:
   mid-conversation LOCATION capture is equally missing — "I'm in
   Bwari" said in one message, then "I was raped" in the next, does NOT
   currently use Bwari for facility referral, since state/lga only ever
   come from the explicit onboarding prompts, never inferred from a
   later free-text message. Deliberately NOT bolted onto the
   safety-critical escalation path without the same deliberate design
   decision as the crisis-persistence question above.
2. **Add a basic "what is pregnancy" (and similarly basic) definitional
   passage to the knowledge base** — content fix, not code; also found
   via real pilot testing.
3. **Phone-verify the four crisis_fallback hotlines** (SURPIN, MANI,
   NAPTIP, WARIF) — candidate numbers are in data/helplines.json and
   scripts/import_facilities.py, all still `verified: false`. Call
   each, confirm, then flip to `true`.
4. **Confirm `HF_LOGS_DATASET_REPO` is set as a Space secret** — the
   log-persistence code is built and live-tested, but won't activate
   in production until this secret exists on the Space. Check via
   Space Settings, or trigger a real interaction and check the Logs
   tab for a `[logger]`-prefixed line.
5. **Retry/backoff logic in generator.py** — still returns
   FALLBACK_RESPONSE immediately on any Gemini error, including 429
   rate limits. The free tier's 20/day cap was hit repeatedly during
   dev testing; a real pilot will hit it too. Move to a paid tier
   before real pilot traffic regardless.
6. **Native-speaker review** of the Hausa/Igbo language-switch
   confirmation messages, and expansion of translator.py's
   `_KNOWN_TRANSLATION_OVERRIDES` glossary for other slang terms NLLB
   mistranslates.
7. **Decide whether the in-chat onboarding needs its own consent/data-
   use disclosure**, and confirm who is actually monitoring for live
   crisis disclosures during the pilot window (see Pilot Readiness
   Checklist item 6) — process questions, not code, but need answers
   before real participants start.
8. **Phase 4 pilot evaluation** — once the above are addressed.

### DONE this session (2026-08-13)
- ~~Test the Twilio WhatsApp webhook end-to-end with a real phone
  number~~ — done via Twilio sandbox; see "Live WhatsApp Pilot Test
  Findings" above for what it found.
- ~~Improve LGA coverage in helplines.json~~ — MAX_FACILITIES_PER_STATE
  raised 5→50, full coverage across all 37 states.
- Built the is_placeholder() gate and wired crisis_fallback into
  _crisis_response() — previously documented as existing but wasn't.
- Committed and pushed to both remotes (GitHub `master` and the HF
  Space `main`) — commit covers data/helplines.json,
  scripts/import_facilities.py, src/safety/escalation.py.

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
- data/helplines.json — the four crisis_fallback hotlines (SURPIN,
  MANI, NAPTIP, WARIF) hold researched candidate numbers but are all
  `verified: false`; 112 is the only genuinely verified number. Must
  be personally called and confirmed before any real user sees them —
  is_placeholder() structurally blocks this already, so nothing reaches
  a user until the flip to `verified: true` happens.

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
   escalation.py prevents deploying unverified numbers. DO NOT remove
   this check. (Built for real 2026-08-13 — earlier versions of this
   doc described it as already existing, but it did not; crisis_fallback
   was written into helplines.json but never read anywhere in src/
   until this was implemented.)

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

14. **Facility `officer_contact` numbers are labelled, not gated —
    deliberately different from crisis_fallback's `is_placeholder()`
    gate (#9).** `is_placeholder()` exists because a crisis_fallback
    entry is presented AS an organisation's official hotline — showing
    one that turns out to be dead or wrong reads as an official line
    failing, so it's hidden entirely until personally phone-verified.
    `officer_contact` is a different kind of number: an individual
    facility survey respondent's personal mobile, always rendered with
    an explicit "self-reported... not verified" clause in the response
    text itself (`_format_officer_contact()` in escalation.py). The
    honesty is in the label, not in withholding the number — phone-
    verifying ~789 individuals (vs. 4 institutional hotlines) isn't
    practical for this project, and gating it the same way as #9 would
    make it permanently invisible, defeating the reason it was built
    (the user explicitly wanted these numbers reachable for better
    navigation, with NAPTIP-style institutional numbers shown alongside
    once THEY clear #9's gate). Do not "fix consistency" by applying
    `is_placeholder()` here — that would silently undo a considered
    tradeoff, not correct a bug.

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
│   │                           regenerated by scripts/import_facilities.py;
│   │                           per-facility officer_contact added by
│   │                           scripts/enrich_facility_contacts.py)
│   ├── .facility_contact_cache.json   (resumability cache for
│   │                           enrich_facility_contacts.py, NOT committed)
│   ├── facility_contact_enrichment_report.json  (unmatched/no-phone
│   │                           facilities from the last enrichment run)
│   ├── user_sessions.json      (onboarding/language/state/LGA/conversation
│   │                           memory per user, hashed phone number keys —
│   │                           NOT committed, runtime state)
│   ├── ingestion_log.json      (tracks processed PDFs for watch_and_ingest)
│   └── pilot_logs.jsonl        (anonymised interaction logs — mirrored to
│                                a private HF Dataset repo, see logger.py)
├── scripts/
│   ├── import_facilities.py    (rebuilds helplines.json from facilities.csv)
│   └── enrich_facility_contacts.py  (adds officer_contact per facility
│                                from api.nphcda.gov.ng, resumable)
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
