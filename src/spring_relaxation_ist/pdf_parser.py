"""PDF text and table extraction utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def extract_text_by_page(pdf_path: Path) -> dict[int, str]:
    """
    Extract text from each page of a PDF using PyMuPDF (fitz).
    Returns a dict mapping 1-based page numbers to page text.
    """
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf") from exc

    result: dict[int, str] = {}
    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc, start=1):
        result[i] = page.get_text()
    doc.close()
    return result


def extract_tables_with_pdfplumber(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Extract tables from the PDF using pdfplumber.
    Returns a list of dicts: {page, table_index, data: list[list]}.
    """
    try:
        import pdfplumber  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pdfplumber is required: pip install pdfplumber") from exc

    results: list[dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for idx, table in enumerate(tables):
                results.append(
                    {
                        "page": page_num,
                        "table_index": idx,
                        "data": table,
                    }
                )
    return results


def detect_figure_references(text: str) -> list[str]:
    """
    Scan text for figure references like 'Figure 18' or 'Fig. 18'.
    """
    pattern = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+)\b", re.IGNORECASE)
    return [f"Figure {m.group(1)}" for m in pattern.finditer(text)]


def detect_table_references(text: str) -> list[str]:
    """
    Scan text for table references like 'Table I' or 'Table 2'.
    """
    pattern = re.compile(r"\bTable\s+([IVX]+|\d+)\b", re.IGNORECASE)
    return [f"Table {m.group(1)}" for m in pattern.finditer(text)]
