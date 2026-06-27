"""
manage_documents.py

Purpose:
    Provides tools to inspect and maintain the ChromaDB knowledge base
    after it has been built — listing what documents are currently
    stored, checking chunk counts, and removing a document's chunks
    (used before replacing a document with an updated version).

Design notes:
    - This file does NOT process PDFs or generate embeddings — that is
      build_knowledge_base.py's job. This file only manages what is
      already stored in ChromaDB.
    - Reuses the same get_collection() connection pattern as
      build_knowledge_base.py, so both files always talk to the same
      "Abiyamo" collection on disk.
    - This is the safety layer that makes incremental updates possible:
      before build_knowledge_base.py re-adds an updated document, this
      file's remove_document_by_source() can clear out the old version's
      chunks first, so outdated content doesn't linger in the knowledge
      base alongside the new version.

Requirements:
    Same as build_knowledge_base.py — chromadb must already be installed.
"""

import chromadb

# Must match build_knowledge_base.py exactly — both files connect to the
# same collection, so these settings are kept identical on purpose.
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "Abiyamo"


# ---------------------------------------------------------------------------
# Step 1: Connect to the same persistent collection used elsewhere
# ---------------------------------------------------------------------------

def get_collection():
    """
    Connects to the persistent ChromaDB database and returns the
    "Abiyamo" collection. Identical to the version in
    build_knowledge_base.py — both files must point at the same
    collection, so this connection logic is kept in sync deliberately.

    Explicitly configured for cosine distance, matching
    build_knowledge_base.py and retriever.py — see the detailed note in
    build_knowledge_base.py's get_collection() for why this matters.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Step 2: List every unique document currently in the knowledge base
# ---------------------------------------------------------------------------

def list_documents() -> dict:
    """
    Returns a summary of every unique source document currently stored
    in the collection, along with how many chunks each one has.

    Returns:
        A dictionary like:
            {
                "FMOH_Manual.pdf": 1711,
                "12_questions_SRH.pdf": 137
            }

    Useful for answering: "what's actually in the knowledge base right
    now?" before deciding what to add, remove, or replace.
    """
    collection = get_collection()
    all_data = collection.get(include=["metadatas"])

    document_counts = {}
    for metadata in all_data["metadatas"]:
        source = metadata.get("source", "unknown")
        document_counts[source] = document_counts.get(source, 0) + 1

    return document_counts


# ---------------------------------------------------------------------------
# Step 3: Print a readable summary (what you'll actually run/see)
# ---------------------------------------------------------------------------

def print_summary():
    """
    Prints a human-readable summary of the knowledge base: total chunk
    count, total document count, and a breakdown per document.
    """
    collection = get_collection()
    document_counts = list_documents()

    print(f"\n[manage_documents] Knowledge base summary for '{COLLECTION_NAME}':")
    print(f"  Total chunks: {collection.count()}")
    print(f"  Total documents: {len(document_counts)}")
    print("  Breakdown by document:")

    if not document_counts:
        print("    (knowledge base is currently empty)")
    else:
        for source, count in sorted(document_counts.items()):
            print(f"    - {source}: {count} chunks")


# ---------------------------------------------------------------------------
# Step 4: Remove all chunks belonging to a specific document
# ---------------------------------------------------------------------------

def remove_document_by_source(filename: str) -> int:
    """
    Deletes all chunks whose "source" metadata matches the given
    filename. Use this before re-adding an updated version of a
    document, so the old version's chunks don't remain in the knowledge
    base alongside the new ones.

    Parameters:
        filename: the exact source filename as stored in metadata
                   (this is the basename, e.g. "FMOH_Manual.pdf" — use
                   list_documents() first if you're unsure of the exact
                   stored name).

    Returns:
        The number of chunks that existed for this document before
        removal (0 if the document wasn't found, so calling code can
        tell the difference between "successfully removed" and
        "nothing to remove").
    """
    collection = get_collection()
    document_counts = list_documents()

    existing_count = document_counts.get(filename, 0)
    if existing_count == 0:
        print(f"[manage_documents] No chunks found for '{filename}' — nothing to remove.")
        return 0

    collection.delete(where={"source": filename})
    print(f"[manage_documents] Removed {existing_count} chunks for '{filename}'.")
    return existing_count


# ---------------------------------------------------------------------------
# Step 5: Check whether a document already exists in the knowledge base
# ---------------------------------------------------------------------------

def document_exists(filename: str) -> bool:
    """
    Quick check used by watch_and_ingest.py (built next) to decide
    whether a file in the raw_pdfs folder has already been processed,
    without needing to load the full document list every time.
    """
    document_counts = list_documents()
    return filename in document_counts


# ---------------------------------------------------------------------------
# Run this file directly to see a summary of your current knowledge base
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_summary()

    # Example of how you would remove a document if you needed to:
    # remove_document_by_source("12 questions and answers about sexual and reproductive health and rights.pdf")
    # print_summary()  # run again afterward to confirm it's gone