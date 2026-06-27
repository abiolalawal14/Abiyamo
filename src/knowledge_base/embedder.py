"""
embedder.py

Purpose:
    Take text chunks (output of text_splitter.py) and turn each one into
    an embedding vector using a LOCAL sentence-transformers model
    (all-MiniLM-L6-v2), running entirely on your own machine.

Why local instead of Gemini's API:
    Testing showed Gemini's free tier embedding API has a hard daily cap
    of 1000 requests per project (separate from, and in addition to, its
    per-minute limit). This project's knowledge base alone produces over
    1,800 chunks from just two documents, which exceeds that daily limit
    before accounting for retries or future documents. A local model
    avoids this entirely: no rate limits, no daily caps, no internet
    dependency once the model is downloaded, and no billing risk.

    The tradeoff: embeddings now come from a different, smaller model
    than Gemini's. This changes the vector dimension from 3072 (Gemini)
    to 384 (all-MiniLM-L6-v2). This is not a problem on its own, but it
    means any previously stored Gemini embeddings are incompatible with
    these new ones — the knowledge base must be rebuilt from scratch
    after switching, which is expected and already accounted for.

Design notes:
    - Same reusable pattern as before: this module wraps the embedding
      logic behind one function, so the rest of the pipeline doesn't
      need to know HOW embeddings are generated, only that calling
      embed_chunks() returns chunks with an "embedding" key added.
    - The model is loaded once and reused across calls, since loading it
      repeatedly would be slow. The first run will download the model
      (a few hundred MB) — this requires an internet connection ONCE;
      after that, it runs fully offline.

Requirements (install via pip):
    pip install sentence-transformers

No API key, no .env entry, and no internet connection needed at runtime
after the first download.
"""

from sentence_transformers import SentenceTransformer

# Centralized model name — change here only, nowhere else, if you ever
# want to try a different sentence-transformers model.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level cache so the model is only loaded once per program run,
# not once per function call — loading it repeatedly would be slow.
_model_cache = None


# ---------------------------------------------------------------------------
# Step 1: Load the model once, reused across all calls
# ---------------------------------------------------------------------------

def get_embedding_model() -> SentenceTransformer:
    """
    Loads and returns the sentence-transformers model, caching it so
    repeated calls don't reload it from disk every time.

    The first time this runs on your machine, it will download the model
    automatically (requires internet, one-time only). After that, it
    loads from a local cache and works fully offline.
    """
    global _model_cache
    if _model_cache is None:
        print(f"[embedder] Loading model '{EMBEDDING_MODEL_NAME}' (first run may download it)...")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[embedder] Model loaded.")
    return _model_cache


# ---------------------------------------------------------------------------
# Step 2: Embed a single piece of text
# ---------------------------------------------------------------------------

def embed_text(text: str, model: SentenceTransformer = None) -> list:
    """
    Generates an embedding vector for a single piece of text.

    Parameters:
        text: the text to embed.
        model: optionally pass an already-loaded model to avoid reloading
               it on every call — matters when embedding many chunks.

    Returns:
        A list of floats representing the embedding vector, or an empty
        list if embedding failed for this piece of text.
    """
    if model is None:
        model = get_embedding_model()

    try:
        vector = model.encode(text)
        return vector.tolist()  # convert from numpy array to a plain list
    except Exception as e:
        print(f"[embedder] Failed to embed text chunk: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 3: Embed a list of chunks (the typical use case from text_splitter.py)
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list) -> list:
    """
    Takes the list of chunk dictionaries returned by text_splitter.py's
    split_document(), and adds an "embedding" key to each one.

    Parameters:
        chunks: list of dicts, each with at least a "text" key
                (as produced by text_splitter.split_document()).

    Returns:
        The same list of dicts, each now including an "embedding" key.
        Chunks that failed to embed are still included, with
        "embedding": [], so calling code can detect and skip them.

    Performance note:
        Since this runs locally with no rate limits, chunks are embedded
        in a single batch call rather than one at a time — this is
        significantly faster than looping, and sentence-transformers is
        built to handle batches efficiently.
    """
    if not chunks:
        return []

    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    print(f"[embedder] Embedding {len(texts)} chunks locally (this runs in batches, no rate limits)...")
    vectors = model.encode(texts, show_progress_bar=True)

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        chunk_with_embedding = dict(chunk)  # avoid mutating the original dict
        chunk_with_embedding["embedding"] = vector.tolist()
        embedded_chunks.append(chunk_with_embedding)

    successful = sum(1 for c in embedded_chunks if c["embedding"])
    print(f"[embedder] Embedded {successful}/{len(chunks)} chunks successfully.")

    return embedded_chunks


# ---------------------------------------------------------------------------
# Quick manual test — only runs if you execute this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulates what text_splitter.py would hand off, so you can test this
    # file on its own before wiring it into the full pipeline.
    sample_chunks = [
        {"source": "sample.pdf", "chunk_index": 0,
         "text": "Sexual and reproductive health information is important for adolescents."},
        {"source": "sample.pdf", "chunk_index": 1,
         "text": "Access to accurate information helps young people make informed decisions."},
    ]

    result = embed_chunks(sample_chunks)

    for chunk in result:
        vector_preview = chunk["embedding"][:5] if chunk["embedding"] else []
        print(f"\nChunk {chunk['chunk_index']} (source: {chunk['source']})")
        print(f"Text: {chunk['text']}")
        print(f"Embedding length: {len(chunk['embedding'])}")
        print(f"First 5 values: {vector_preview}")