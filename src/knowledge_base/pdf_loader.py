"""
pdf_loader.py

Purpose:
    Load a single PDF file and return clean, extracted text.
    Tries normal text extraction first. If the PDF turns out to be
    scanned/image-based (and normal extraction returns little or nothing),
    it automatically falls back to OCR.

Design notes:
    - This module handles ONE file at a time, on purpose.
    - Folder-looping (processing every PDF in a directory) belongs in
      build_knowledge_base.py or watch_and_ingest.py, NOT here.
    - This keeps the function reusable for both the initial batch load
      and the later scheduled auto-ingestion step.

Requirements (install via pip):
    pip install langchain-community pypdf pytesseract pdf2image Pillow

System-level requirements (must be installed separately, not via pip):
    - Tesseract OCR  (https://github.com/UB-Mannheim/tesseract/wiki for Windows)
    - Poppler        (needed by pdf2image to render PDF pages as images)
    Both must be added to your system PATH. Verify with:
        tesseract --version
        pdftoppm -v
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import pytesseract


# ---------------------------------------------------------------------------
# Step 1: Try normal text extraction (works for standard, text-based PDFs)
# ---------------------------------------------------------------------------

def extract_text_normal(file_path: str) -> str:
    """
    Attempts standard text extraction using LangChain's PyPDFLoader.
    Returns the combined text of all pages as a single string.
    Returns an empty string if extraction fails or produces nothing.
    """
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text = "\n".join(page.page_content for page in pages)
        return text.strip()
    except Exception as e:
        print(f"[normal extraction] Failed for {file_path}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Step 2: Decide whether the normal extraction actually worked
# ---------------------------------------------------------------------------

def is_extraction_sufficient(text: str, min_chars_per_page: int = 20, page_count: int = 1) -> bool:
    """
    Heuristic check: if the extracted text is too short relative to the
    number of pages, it's likely a scanned/image-based PDF rather than
    a genuinely short document.

    Adjust min_chars_per_page if you find this triggers incorrectly on
    your specific set of documents.
    """
    if not text:
        return False
    expected_minimum = min_chars_per_page * max(page_count, 1)
    return len(text) >= expected_minimum


# ---------------------------------------------------------------------------
# Step 3: OCR fallback for scanned/image-based PDFs
# ---------------------------------------------------------------------------

def extract_text_ocr(file_path: str) -> str:
    """
    Converts each PDF page to an image, then runs Tesseract OCR on each
    image to extract text. Used only when normal extraction fails.
    """
    try:
        images = convert_from_path(file_path)
    except Exception as e:
        print(f"[OCR] Failed to convert {file_path} to images: {e}")
        return ""

    extracted_pages = []
    for i, image in enumerate(images):
        try:
            page_text = pytesseract.image_to_string(image)
            extracted_pages.append(page_text)
        except Exception as e:
            print(f"[OCR] Failed on page {i + 1} of {file_path}: {e}")
            extracted_pages.append("")

    return "\n".join(extracted_pages).strip()


# ---------------------------------------------------------------------------
# Step 4: Get page count (needed for the sufficiency check above)
# ---------------------------------------------------------------------------

def get_page_count(file_path: str) -> int:
    """
    Returns the number of pages in the PDF using PyPDFLoader.
    Falls back to 1 if this fails, so the rest of the pipeline doesn't break.
    """
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return max(len(pages), 1)
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Step 5: Main entry point — this is the function other files will call
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> dict:
    """
    Main reusable function. Call this with a single PDF file path.

    Returns a dictionary with:
        - filename: the base name of the file
        - text: the extracted text (from normal extraction or OCR)
        - method_used: "normal" or "ocr", so you know which path was taken
        - page_count: number of pages in the document
        - success: True if any usable text was extracted, False otherwise

    This return format is deliberately consistent regardless of which
    extraction method succeeded, so build_knowledge_base.py and
    watch_and_ingest.py can call this without caring about the internals.
    """
    if not os.path.exists(file_path):
        return {
            "filename": os.path.basename(file_path),
            "text": "",
            "method_used": "none",
            "page_count": 0,
            "success": False,
        }

    filename = os.path.basename(file_path)
    page_count = get_page_count(file_path)

    # Try normal extraction first
    text = extract_text_normal(file_path)

    if is_extraction_sufficient(text, page_count=page_count):
        return {
            "filename": filename,
            "text": text,
            "method_used": "normal",
            "page_count": page_count,
            "success": True,
        }

    # Fall back to OCR if normal extraction wasn't sufficient
    print(f"[{filename}] Normal extraction insufficient — falling back to OCR.")
    ocr_text = extract_text_ocr(file_path)

    return {
        "filename": filename,
        "text": ocr_text,
        "method_used": "ocr",
        "page_count": page_count,
        "success": bool(ocr_text.strip()),
    }


# ---------------------------------------------------------------------------
# Quick manual test — only runs if you execute this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace this with a real path to one of your PDFs to test manually.
    test_path = "data/raw_pdfs/12 questions and answers about sexual and reproductive health and rights.pdf"
    result = load_pdf(test_path)

    print(f"Filename: {result['filename']}")
    print(f"Method used: {result['method_used']}")
    print(f"Page count: {result['page_count']}")
    print(f"Success: {result['success']}")
    print(f"First 300 characters of extracted text:\n{result['text'][:300]}")