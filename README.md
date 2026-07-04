---
title: Abiyamo SRH Chatbot
emoji: 🤝
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Abiyamo — SRH Chatbot

A RAG-based multilingual chatbot providing sexual and reproductive health (SRH)
information for young people in Nigeria. Built as an MSc AI/ML dissertation
project at Miva Open University.

**Target users:** Ages 16–24  
**Languages:** English, Hausa, Yoruba, Igbo  
**Channels:** REST API, WhatsApp (via Twilio)

---

## How it works

```
User message (any language)
        │
        ▼
 Escalation check ──── crisis/diagnostic? ──► Scripted response + helplines
        │
        │ educational
        ▼
  to_english()          ← NLLB-200 distilled 600M (local, no API)
        │
        ▼
  ChromaDB retrieval    ← all-MiniLM-L6-v2 embeddings (local, no API)
        │
        ▼
  Gemini generation     ← gemini-3.5-flash
        │
        ▼
  from_english()        ← NLLB-200 (if user language ≠ English)
        │
        ▼
  Answer logged + returned
```

---

## Tech stack

| Component | Technology |
|---|---|
| API framework | FastAPI 0.138.0 |
| Vector store | ChromaDB ("Abiyamo" collection, cosine distance) |
| Embedding model | sentence-transformers all-MiniLM-L6-v2 (local) |
| Translation model | facebook/nllb-200-distilled-600M (local) |
| LLM | Gemini via google-genai v2.8.0 |
| WhatsApp | Twilio REST API via httpx |
| Document loading | LangChain |

---

## Project structure

```
srh-chatbot/
├── data/
│   ├── raw_pdfs/           source PDFs (FMOH manual, 12 Questions SRH doc)
│   ├── chroma_db/          ChromaDB persistent storage — do not delete
│   ├── raw_queries/        annotator Label Studio exports
│   ├── helplines.json      PLACEHOLDER — verify all numbers before use
│   ├── ingestion_log.json  tracks processed PDFs
│   └── pilot_logs.jsonl    anonymised interaction logs
├── src/
│   ├── knowledge_base/     PDF loading, chunking, embedding, ingestion pipeline
│   ├── annotation/         Label Studio CSV merge + Cohen's Kappa
│   ├── safety/             escalation.py — crisis/diagnostic routing
│   ├── rag_pipeline/       retriever, prompt_builder, generator, translator
│   ├── whatsapp/           Twilio webhook + sender
│   └── api/                main.py (FastAPI app) + chat_handler.py (pipeline)
├── evaluation/
│   ├── logger.py           interaction logger (SHA-256 hashed phone numbers)
│   └── pilot_report.py     7-section terminal report (rich + pandas)
├── CLAUDE.md               full project context for Claude Code
└── README.md
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
TWILIO_ACCOUNT_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_token_here
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```

Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com).  
The free tier (60 RPM) is sufficient for a pilot.

### 4. Build the knowledge base (first time only)

```bash
python -m src.knowledge_base.build_knowledge_base
```

This loads the PDFs from `data/raw_pdfs/`, chunks them, embeds them locally
with all-MiniLM-L6-v2, and writes 1,848 chunks to ChromaDB.

---

## Running the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check — reports chunk count and log count |
| POST | `/chat` | Send a message, get a response |
| POST | `/whatsapp/incoming` | Twilio webhook for WhatsApp messages |

### Example chat request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the safe period method?", "language": "en"}'
```

```bash
# Hausa
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Kisa ake nufi da hana haihuwa?", "language": "ha"}'
```

Response shape:
```json
{
  "answer": "...",
  "language": "en",
  "escalated": false,
  "chunks_used": 3
}
```

---

## Supported languages

| BCP-47 | Language | NLLB-200 code |
|---|---|---|
| `en` | English | `eng_Latn` |
| `ha` | Hausa | `hau_Latn` |
| `yo` | Yoruba | `yor_Latn` |
| `ig` | Igbo | `ibo_Latn` |

Translation happens automatically. Pass the user's BCP-47 language code in the
`language` field — the pipeline translates the question to English before
retrieval and translates the answer back before responding.

---

## Pilot evaluation report

```bash
python -m evaluation.pilot_report
```

Generates a 7-section terminal report: overview, language distribution,
escalation rate, chunk stats, response time percentiles, 14-day volume
chart, and last 10 interactions. Generates a synthetic preview if no
real logs exist yet.

---

## Safety design

- **Escalation bypasses the LLM entirely.** Crisis and diagnostic queries return
  scripted text and verified helpline numbers — Gemini is never called.
- **Helplines are placeholder by default.** `data/helplines.json` contains the
  structure but all numbers (except Nigeria emergency `112`) are marked
  `PLACEHOLDER_NOT_VERIFIED`. Do not deploy to real users until every number
  is personally verified as active.
- **Phone numbers are hashed before logging.** `evaluation/logger.py`
  SHA-256 hashes all phone numbers (16 hex chars) before writing to
  `pilot_logs.jsonl`.

---

## Build status

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Knowledge base | Complete | 1,848 chunks in ChromaDB |
| Phase 1 — Annotation pipeline | Built, blocked | Waiting on annotators to finish Label Studio labelling |
| Phase 2 — DistilBERT classifier | Not started | Blocked on Phase 1 annotation; will run on Google Colab |
| Phase 3 — RAG + API + WhatsApp | Complete | All components built and wired |
| Phase 4 — Pilot evaluation | Not started | Requires credentials + live deployment |

---

## Key design decisions

- **Local embedding model** — all-MiniLM-L6-v2 runs locally. Gemini free tier
  has a hard 1,000/day cap which would be exhausted by the knowledge base
  alone (1,848 chunks).
- **ChromaDB cosine distance** — L2 (ChromaDB's default) gives weaker retrieval
  for sentence-transformer embeddings. Every `get_collection()` call includes
  `metadata={"hnsw:space": "cosine"}`.
- **NLLB-200 distilled 600M** — not the full 3.3B model. The dev machine has
  no GPU; the distilled model runs on CPU at acceptable latency for a pilot.
- **Twilio BackgroundTasks** — the WhatsApp webhook returns empty TwiML
  immediately and processes the pipeline in a background task, avoiding
  Twilio's 15-second response timeout.
