"""
build_knowledge_base.py

Purpose:
    Take a single PDF file all the way through the pipeline — load it,
    split it into chunks, embed each chunk — and store the result in a
    persistent ChromaDB collection.

    This file processes ONE document per call, by design. This is what
    makes incremental updates possible later: adding a new document does
    not require rebuilding the entire knowledge base, and updating an
    existing document only touches that document's own chunks.

Design notes:
    - Chains together pdf_loader.py -> text_splitter.py -> embedder.py.
    - Uses ChromaDB's PersistentClient so data survives between sessions
      (this is what avoids the Colab "data disappears" problem discussed
      earlier, as long as the chroma_db folder itself is preserved).
    - Each chunk is stored with metadata (source filename, chunk_index),
      which is what makes selective removal/replacement possible.

Requirements (install via pip):
    pip install chromadb

Folder expectation:
    data/chroma_db/   <- ChromaDB will create and manage this folder itself
"""

import os
import chromadb

from .pdf_loader import load_pdf
from .text_splitter import split_document
from .embedder import embed_chunks


# Centralized settings — change here only, nowhere else, if they ever change
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "Abiyamo"


# ---------------------------------------------------------------------------
# Step 1: Connect to the persistent ChromaDB collection
# ---------------------------------------------------------------------------

def get_collection():
    """
    Connects to the persistent ChromaDB database on disk and returns the
    "Abiyamo" collection, creating it if it doesn't exist yet.

    Using get_or_create_collection means this is safe to call every time —
    it will not wipe existing data if the collection already exists, which
    is essential for the incremental-update behavior this whole module is
    designed around.

    IMPORTANT: explicitly configured to use cosine distance, matching the
    metric all-MiniLM-L6-v2 (the embedding model used throughout this
    project) was trained/optimized for. ChromaDB defaults to L2 distance
    if not specified, which produced visibly weaker-differentiated
    retrieval results during testing. This setting only takes effect when
    the collection is first CREATED — if you previously built "Abiyamo"
    before this fix was added, it was created with the old L2 default and
    must be rebuilt from scratch (delete data/chroma_db/ and re-run this
    file) for cosine distance to actually apply.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Step 2: Remove an existing document's chunks (used before re-adding an
# updated version of a document, so old content doesn't linger alongside
# the new version)
# ---------------------------------------------------------------------------

def remove_document(filename: str):
    """
    Deletes all chunks belonging to a given source filename from the
    collection. Call this before re-adding a document if you are
    replacing it with an updated version — otherwise both the old and
    new chunks would exist side by side, and the chatbot could retrieve
    outdated information alongside the current version.
    """
    collection = get_collection()
    collection.delete(where={"source": filename})
    print(f"[build_knowledge_base] Removed existing chunks for '{filename}' (if any existed).")


# ---------------------------------------------------------------------------
# Step 3: Process and add a single document to the knowledge base
# ---------------------------------------------------------------------------

def add_document(file_path: str, replace_if_exists: bool = False):
    """
    Runs one PDF through the full pipeline and adds it to ChromaDB.

    Parameters:
        file_path: path to the PDF file to process.
        replace_if_exists: if True, removes any existing chunks for this
                            filename first. Set this to True when you are
                            deliberately updating a document with a newer
                            version. Leave False for first-time additions,
                            so you don't accidentally wipe something by
                            mistake.

    Returns:
        True if the document was added successfully, False if it failed
        or had no usable content (so calling code, e.g. watch_and_ingest.py,
        can log or report which documents succeeded and which didn't).
    """
    filename = os.path.basename(file_path)

    # Step A: Load the PDF (handles OCR fallback internally)
    document_result = load_pdf(file_path)
    if not document_result["success"]:
        print(f"[build_knowledge_base] Skipping '{filename}' — no usable text extracted.")
        return False

    # Step B: Split into chunks
    chunks = split_document(document_result)
    if not chunks:
        print(f"[build_knowledge_base] Skipping '{filename}' — produced no chunks.")
        return False

    # Step C: Generate embeddings for each chunk
    embedded_chunks = embed_chunks(chunks)
    embedded_chunks = [c for c in embedded_chunks if c["embedding"]]  # drop any that failed
    if not embedded_chunks:
        print(f"[build_knowledge_base] Skipping '{filename}' — no chunks were successfully embedded.")
        return False

    # Step D: If replacing, remove the old version first
    if replace_if_exists:
        remove_document(filename)

    # Step E: Store everything in ChromaDB
    collection = get_collection()

    ids = [f"{filename}_{c['chunk_index']}" for c in embedded_chunks]
    documents = [c["text"] for c in embedded_chunks]
    embeddings = [c["embedding"] for c in embedded_chunks]
    metadatas = [{"source": filename, "chunk_index": c["chunk_index"]} for c in embedded_chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[build_knowledge_base] Added {len(embedded_chunks)} chunks from '{filename}' to '{COLLECTION_NAME}'.")
    return True


# ---------------------------------------------------------------------------
# Step 4: Process every PDF in a folder (used for the initial batch build)
# ---------------------------------------------------------------------------

def add_documents_from_folder(folder_path: str):
    """
    Loops through every PDF in a folder and adds each one to the
    knowledge base. This is meant for the initial setup, when you are
    loading your starting set of verified documents all at once.

    For ongoing updates after initial setup, prefer calling add_document()
    directly on one new/updated file, or use watch_and_ingest.py once
    that is built.
    """
    if not os.path.isdir(folder_path):
        print(f"[build_knowledge_base] Folder not found: {folder_path}")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[build_knowledge_base] No PDF files found in {folder_path}")
        return

    print(f"[build_knowledge_base] Found {len(pdf_files)} PDF(s) to process.")

    results = {"succeeded": [], "failed": []}
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        success = add_document(full_path, replace_if_exists=False)
        if success:
            results["succeeded"].append(pdf_file)
        else:
            results["failed"].append(pdf_file)

    print(f"\n[build_knowledge_base] Done. Succeeded: {len(results['succeeded'])}, "
          f"Failed: {len(results['failed'])}")
    if results["failed"]:
        print(f"Failed files: {results['failed']}")

    return results


# ---------------------------------------------------------------------------
# Quick manual test — only runs if you execute this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # First-time setup: process every PDF currently in data/raw_pdfs/
    add_documents_from_folder("data/raw_pdfs")

    # Example of how you would later update a single document:
    # add_document("data/raw_pdfs/FMOH_SRH_Manual_2026.pdf", replace_if_exists=True)

    # Quick sanity check: confirm how many chunks are in the collection now
    collection = get_collection()
    print(f"\nTotal chunks currently in '{COLLECTION_NAME}': {collection.count()}")