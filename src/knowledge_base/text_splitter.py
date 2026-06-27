"""
text_splitter.py

Purpose:
    Take the raw extracted text from a single document (output of
    pdf_loader.py) and split it into smaller, overlapping chunks suitable
    for embedding.

Why chunking matters:
    Embedding an entire document as one vector loses precision — retrieval
    would return whole documents instead of the specific relevant passage.
    Splitting into smaller, overlapping chunks lets the RAG pipeline later
    retrieve just the part of a document that actually answers a user's
    question.

Design notes:
    - This module handles ONE document's text at a time, same pattern as
      pdf_loader.py. It does not loop over files itself.
    - Each chunk returned carries metadata (source filename, chunk index)
      so it can later be tagged in ChromaDB — this is what makes selective
      document removal/update possible, as discussed earlier.

Requirements (install via pip):
    pip install langchain-text-splitters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default chunking parameters — adjust here if needed, in one place only.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# Step 1: Build the splitter (kept as its own function so settings are
# easy to find and change without touching the rest of the logic)
# ---------------------------------------------------------------------------

def build_splitter(chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """
    Creates a LangChain text splitter configured with the given chunk size
    and overlap. RecursiveCharacterTextSplitter tries to split on natural
    boundaries first (paragraphs, then sentences, then words), only falling
    back to a hard character cut if nothing else works — this keeps chunks
    more semantically coherent than a naive fixed-length cut.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ---------------------------------------------------------------------------
# Step 2: Main entry point — this is the function other files will call
# ---------------------------------------------------------------------------

def split_document(document_result: dict,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    """
    Takes the dictionary returned by pdf_loader.load_pdf() and splits its
    text into chunks.

    Parameters:
        document_result: the dict returned by load_pdf(), expected to have
                          at least "filename" and "text" keys.
        chunk_size / chunk_overlap: optional overrides for this call only.

    Returns:
        A list of dictionaries, one per chunk, each containing:
            - source: the original filename (for later filtering/removal)
            - chunk_index: position of this chunk within the document
            - text: the chunk's actual text content

    If the document had no usable text (success=False from load_pdf),
    this returns an empty list rather than raising an error, so the
    pipeline can skip it gracefully and move to the next document.
    """
    if not document_result.get("success") or not document_result.get("text"):
        print(f"[{document_result.get('filename', 'unknown')}] No usable text to split — skipping.")
        return []

    splitter = build_splitter(chunk_size, chunk_overlap)
    raw_chunks = splitter.split_text(document_result["text"])

    filename = document_result["filename"]
    chunks = []
    for index, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "source": filename,
            "chunk_index": index,
            "text": chunk_text,
        })

    print(f"[{filename}] Split into {len(chunks)} chunks.")
    return chunks


# ---------------------------------------------------------------------------
# Quick manual test — only runs if you execute this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulates what pdf_loader.py would hand off, so you can test this
    # file on its own before wiring it to the real PDF loader.
    sample_document_result = {
        "filename": "data/raw_pdfs/12 questions and answers about sexual and reproductive health and rights.pdf",
        "text": (
            "Sexual and reproductive health information is important for "
            "adolescents. Access to accurate information helps young people "
            "make informed decisions about contraception, pregnancy, and "
            "sexually transmitted infections.\n\n"
            "Many young people face barriers to accessing this information, "
            "including stigma, cultural taboos, and limited availability of "
            "youth-friendly health services."
        ),
        "method_used": "normal",
        "page_count": 1,
        "success": True,
    }

    chunks = split_document(sample_document_result)

    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_index']} (source: {chunk['source']}):")
        print(chunk["text"])