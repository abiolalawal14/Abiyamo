"""
main.py

Purpose:
    FastAPI application entry point. Creates the app, configures
    middleware, mounts the WhatsApp webhook router, and exposes two
    HTTP endpoints:
        GET  /health  — liveness check + knowledge base stats
        POST /chat    — direct text chat (for testing, dashboard, etc.)

    The WhatsApp interface lives in src/whatsapp/webhook.py and is
    mounted here at /whatsapp. All pipeline logic lives in
    src/api/chat_handler.py — main.py stays thin.

How to run (development):
    cd to the project root, then:
        .venv/Scripts/uvicorn src.api.main:app --reload --port 8000

    For Twilio to reach the webhook, expose the local server with ngrok:
        ngrok http 8000
    Then set the Twilio webhook URL to:
        https://<ngrok-subdomain>.ngrok.io/whatsapp/incoming

Startup behaviour:
    On startup the app pre-loads the sentence-transformers embedding
    model into memory. This adds ~2 seconds to startup time but means
    the first user message is answered at full speed rather than
    waiting for model load. Without this, the first request after a
    cold start would take noticeably longer.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.whatsapp.webhook import router as whatsapp_router
from src.api.chat_handler import handle_message
from evaluation.logger import log_interaction


# ---------------------------------------------------------------------------
# Startup: pre-load the embedding model so the first request is fast
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before serving requests) and once at
    shutdown. Used here to warm up the embedding model and log
    knowledge base stats so problems are visible immediately rather
    than on the first user request.
    """
    print("[main] Starting Abiyamo SRH Chatbot API...")

    # Pre-load the embedding model — this is the main cold-start cost.
    # Importing get_embedding_model() here (rather than at module level)
    # keeps the import from running during test collection.
    try:
        from src.rag_pipeline.retriever import get_embedding_model, get_collection
        get_embedding_model()

        collection = get_collection()
        chunk_count = collection.count()
        print(f"[main] Knowledge base ready — {chunk_count} chunks in 'Abiyamo' collection.")
    except Exception as e:
        # Log but don't abort startup — the app can still serve
        # requests that hit the fallback path even if ChromaDB fails.
        print(f"[main] Warning: could not verify knowledge base at startup: {e}")

    from src.knowledge_base.build_knowledge_base import (
        add_documents_from_folder, get_collection
    )

    # Check if knowledge base is empty
    collection = get_collection()
    if collection.count() == 0:
        print("[main] Knowledge base empty — rebuilding from raw_pdfs...")
        add_documents_from_folder("data/raw_pdfs")
        print(f"[main] Knowledge base ready — {collection.count()} chunks")
    else:
        print(f"[main] Knowledge base ready — {collection.count()} chunks")

    print("[main] API ready.")
    yield
    # Shutdown — nothing to clean up for now.
    print("[main] Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Abiyamo SRH Chatbot",
    description=(
        "RAG-based sexual and reproductive health chatbot for young "
        "people in Nigeria. Supports English, Hausa, Yoruba, and Igbo "
        "(translation support coming in Phase 3)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — open during pilot so the admin dashboard and any test clients
# can connect from any origin. Tighten to specific domains before
# public launch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount the WhatsApp webhook router. All routes defined in webhook.py
# are available under /whatsapp (e.g. POST /whatsapp/incoming).
app.include_router(whatsapp_router, prefix="/whatsapp", tags=["WhatsApp"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="The user's question in any supported language.",
        examples=["What is the safe period method of family planning?"],
    )
    language: str = Field(
        default="en",
        description=(
            "BCP-47 language code of the message. "
            "Supported: 'en', 'ha', 'yo', 'ig'. "
            "Translation is not yet active — all messages are currently "
            "processed as English regardless of this value."
        ),
        examples=["en", "ha", "yo", "ig"],
    )


class ChatResponse(BaseModel):
    answer: str = Field(description="The chatbot's response.")
    language: str = Field(description="Language code of the response.")
    escalated: bool = Field(
        description=(
            "True if the query was routed to the safety escalation path "
            "(helplines) rather than the RAG pipeline. When True, the "
            "answer contains scripted safety text, not generated content."
        )
    )
    chunks_used: int = Field(
        description=(
            "Number of knowledge base chunks retrieved for this response. "
            "0 if escalated or if the knowledge base had no relevant content."
        )
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """
    Liveness check. Returns the knowledge base chunk count so it is
    easy to confirm the vector store is populated after deployment.
    Returns 'degraded' status if ChromaDB is unreachable rather than
    returning an error — the API can still run in fallback mode.
    """
    try:
        from src.rag_pipeline.retriever import get_collection
        collection = get_collection()
        chunk_count = collection.count()
        return {
            "status": "ok",
            "knowledge_base": {
                "collection": "Abiyamo",
                "chunks": chunk_count,
            },
        }
    except Exception as e:
        return {
            "status": "degraded",
            "knowledge_base": None,
            "error": str(e),
        }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Direct text chat endpoint. Accepts a question and returns an answer
    from the RAG pipeline.

    Primarily used for:
    - Testing the pipeline without going through WhatsApp
    - An admin/analyst dashboard
    - Integration testing during development

    For production WhatsApp users, all traffic goes through
    POST /whatsapp/incoming instead.
    """
    import time
    start = time.time()
    result = handle_message(request.message, request.language)
    elapsed_ms = (time.time() - start) * 1000

    log_interaction(
        user_id="api",
        message=request.message,
        result=result,
        response_time_ms=elapsed_ms,
        channel="api",
    )
    return ChatResponse(**result)


# ---------------------------------------------------------------------------
# Dev entry point — run with: python -m src.api.main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
