"""
retriever.py

Purpose:
    Takes a user's question, embeds it using the SAME local model used
    to build the knowledge base (all-MiniLM-L6-v2), and queries ChromaDB
    to retrieve the most relevant chunks for answering that question.

Why the same embedding model is required, not just preferred:
    ChromaDB compares vectors by mathematical distance. A query embedded
    with a different model would produce vectors in an entirely
    different vector space — comparing them to the knowledge base's
    vectors would be meaningless, not just lower quality. This is why
    embedder.py's model (all-MiniLM-L6-v2) is reused here directly,
    rather than treated as a separate choice.

Design notes:
    - This module ONLY retrieves — it does not build prompts or call any
      LLM. That is prompt_builder.py and generator.py's job
      (not yet built). Keeping retrieval isolated means it can be
      tested and verified on its own before being wired into the rest
      of the RAG pipeline.
    - Reuses the exact same ChromaDB connection settings as
      build_knowledge_base.py and manage_documents.py, so this always
      reads from the same "Abiyamo" collection — never a separate copy.
    - This module is for EDUCATIONAL queries only. Diagnostic and crisis
      queries should never reach this file — they are handled entirely
      by escalation.py instead, bypassing retrieval and generation
      altogether, as established earlier in this project's design.

Requirements:
    Same as embedder.py and build_knowledge_base.py — sentence-transformers
    and chromadb must already be installed.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# Must match build_knowledge_base.py and manage_documents.py exactly —
# all three files must point at the same collection.
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "Abiyamo"

# Must match embedder.py exactly — using a different model here would
# silently break retrieval (see module docstring above).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# How many chunks to retrieve per query by default. Adjustable per call,
# but centralized here as the default so it's easy to find and tune.
DEFAULT_TOP_K = 3

# Module-level cache, same pattern as embedder.py, so the model is only
# loaded once per program run, not once per question asked.
_model_cache = None


# ---------------------------------------------------------------------------
# Step 1: Load the embedding model (reused, not rebuilt, from embedder.py's
# pattern — kept as its own copy here so retriever.py can be tested and
# used independently without importing embedder.py's batch-processing
# logic, which it doesn't need)
# ---------------------------------------------------------------------------

def get_embedding_model() -> SentenceTransformer:
    """
    Loads and returns the same sentence-transformers model used to build
    the knowledge base, caching it so repeated calls don't reload it.
    """
    global _model_cache
    if _model_cache is None:
        print(f"[retriever] Loading model '{EMBEDDING_MODEL_NAME}'...")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[retriever] Model loaded.")
    return _model_cache


# ---------------------------------------------------------------------------
# Step 2: Connect to the same ChromaDB collection used elsewhere
# ---------------------------------------------------------------------------

def get_collection():
    """
    Connects to the persistent ChromaDB database and returns the
    "Abiyamo" collection. Identical connection logic to
    build_knowledge_base.py and manage_documents.py, kept in sync
    deliberately — all three files must always read/write the same data.

    IMPORTANT: explicitly configured to use cosine distance, matching
    the metric all-MiniLM-L6-v2 was trained/optimized for. ChromaDB
    defaults to L2 distance if not specified, which is a weaker match
    for sentence-transformer embeddings — this was confirmed during
    testing, where L2-based results showed only modest separation
    between strong and weak matches. This setting only takes effect
    when the collection is first CREATED; if "Abiyamo" already exists
    with the old default L2 setting, it must be rebuilt for this change
    to apply (see note in build_knowledge_base.py).
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Step 3: Embed the user's question
# ---------------------------------------------------------------------------

def embed_query(question: str) -> list:
    """
    Embeds a single user question using the same model and process used
    for the knowledge base chunks, so the resulting vector can be
    meaningfully compared against them.
    """
    model = get_embedding_model()
    vector = model.encode(question)
    return vector.tolist()


# ---------------------------------------------------------------------------
# Step 4: Main entry point — retrieve relevant chunks for a question
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> list:
    """
    Takes a user's question and returns the most relevant chunks from
    the knowledge base.

    Parameters:
        question: the user's educational SRH question (already confirmed
                   by the intent classifier to NOT be diagnostic/crisis —
                   this function does not check that itself, the caller
                   is responsible for only sending educational queries
                   here).
        top_k: how many chunks to retrieve. Defaults to DEFAULT_TOP_K.

    Returns:
        A list of dictionaries, one per retrieved chunk:
            {
                "text": the chunk's actual content,
                "source": which document it came from,
                "chunk_index": its position within that document,
                "distance": how close the match was (lower = more similar)
            }
        Returns an empty list if the knowledge base is empty or the
        query fails, rather than raising an error — a downstream
        component (prompt_builder.py) can decide how to handle "no
        relevant content found" gracefully.
    """
    if not question or not question.strip():
        print("[retriever] Empty question received — returning no results.")
        return []

    try:
        query_vector = embed_query(question)
    except Exception as e:
        print(f"[retriever] Failed to embed question: {e}")
        return []

    collection = get_collection()

    if collection.count() == 0:
        print("[retriever] Knowledge base is empty — no chunks to retrieve.")
        return []

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),  # avoid asking for more than exists
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[retriever] ChromaDB query failed: {e}")
        return []

    # ChromaDB returns results nested in lists (one outer list per query
    # embedding submitted) — since we only submit one question at a
    # time, we always take index [0].
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        retrieved_chunks.append({
            "text": text,
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", -1),
            "distance": distance,
        })

    return retrieved_chunks


# ---------------------------------------------------------------------------
# Quick manual test — only runs if you execute this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_question = "What is the safe period method of family planning?"

    print(f"[retriever] Test question: \"{test_question}\"\n")
    results = retrieve_relevant_chunks(test_question, top_k=3)

    if not results:
        print("No results returned — check that the knowledge base has been built "
              "(run build_knowledge_base.py first if you haven't).")
    else:
        for i, chunk in enumerate(results):
            print(f"--- Result {i + 1} (distance: {chunk['distance']:.4f}) ---")
            print(f"Source: {chunk['source']} (chunk {chunk['chunk_index']})")
            print(f"Text: {chunk['text'][:300]}...")
            print()