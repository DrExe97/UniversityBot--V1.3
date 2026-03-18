"""
document_loader.py - Extract text from PDF, DOCX, and TXT files
Prepares document content for the knowledge base
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── SUPPORTED FORMATS ───────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}


def is_supported(filename: str) -> bool:
    """Check if a file format is supported."""
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


# ─── PDF EXTRACTION ──────────────────────────────────────────

def extract_from_pdf(filepath: str) -> str:
    """Extract text from a PDF file."""
    try:
        import PyPDF2

        text_parts = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Add page marker for context
                        text_parts.append(f"[Page {page_num + 1}/{total_pages}]\n{page_text.strip()}")
                except Exception as e:
                    logger.warning(f"Could not extract page {page_num + 1}: {e}")
                    continue

        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} chars from PDF ({total_pages} pages)")
        return full_text

    except ImportError:
        raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    except Exception as e:
        logger.error(f"PDF extraction failed for {filepath}: {e}")
        raise


# ─── DOCX EXTRACTION ─────────────────────────────────────────

def extract_from_docx(filepath: str) -> str:
    """Extract text from a Word document (.docx)."""
    try:
        from docx import Document

        doc = Document(filepath)
        text_parts = []

        # Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                # Preserve heading structure
                if para.style.name.startswith("Heading"):
                    text_parts.append(f"\n## {text}\n")
                else:
                    text_parts.append(text)

        # Extract tables
        for table in doc.tables:
            table_text = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    table_text.append(" | ".join(row_cells))
            if table_text:
                text_parts.append("\n[Table]\n" + "\n".join(table_text) + "\n")

        full_text = "\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} chars from DOCX")
        return full_text

    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    except Exception as e:
        logger.error(f"DOCX extraction failed for {filepath}: {e}")
        raise


# ─── TXT / MARKDOWN EXTRACTION ───────────────────────────────

def extract_from_txt(filepath: str) -> str:
    """Extract text from plain text or Markdown files."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                text = f.read()
            logger.info(f"Extracted {len(text)} chars from TXT (encoding: {encoding})")
            return text.strip()
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {filepath} with any supported encoding")


# ─── MAIN DISPATCHER ─────────────────────────────────────────

def extract_text(filepath: str) -> str:
    """
    Extract text from any supported file format.
    Dispatches to the correct extractor based on file extension.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()
    filename = path.name

    logger.info(f"Extracting text from: {filename} ({ext})")

    if ext == ".pdf":
        text = extract_from_pdf(filepath)
    elif ext in (".docx", ".doc"):
        text = extract_from_docx(filepath)
    elif ext in (".txt", ".md"):
        text = extract_from_txt(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Clean up the extracted text
    text = clean_text(text)

    if not text:
        raise ValueError(f"No text could be extracted from {filename}")

    return text


# ─── TEXT CLEANING ───────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted text for better chunking and retrieval.
    Removes excessive whitespace, fixes common OCR artifacts.
    """
    if not text:
        return ""

    import re

    # Replace Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove non-printable characters (except newlines and tabs)
    text = re.sub(r"[^\x20-\x7E\n\t\u00A0-\uFFFF]", "", text)

    # Fix common PDF artifacts: lone hyphens at end of lines (word wrapping)
    text = re.sub(r"-\n([a-z])", r"\1", text)

    return text.strip()


# ─── FILE METADATA ───────────────────────────────────────────

def get_file_metadata(filepath: str) -> dict:
    """Get file size and basic info."""
    path = Path(filepath)
    size_bytes = path.stat().st_size
    return {
        "filename":    path.name,
        "extension":   path.suffix.lower(),
        "size_bytes":  size_bytes,
        "size_kb":     round(size_bytes / 1024, 2),
    }
