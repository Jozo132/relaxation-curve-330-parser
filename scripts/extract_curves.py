"""Standalone script to extract curves from the IST Report 330 PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spring_relaxation_ist.config import (
    CURVES_JSON_PATH,
    GENERATED_DIR,
    METADATA_JSON_PATH,
    PDF_PATH,
)
from spring_relaxation_ist.curve_extractor import extract_curves
from spring_relaxation_ist.downloader import sha256_of_file

if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"PDF not found at {PDF_PATH}. Run scripts/download_report.py first.")
        sys.exit(1)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    sha = sha256_of_file(PDF_PATH)
    dataset, metadata = extract_curves(PDF_PATH, pdf_sha256=sha)

    with open(CURVES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"Generated: {CURVES_JSON_PATH}")

    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"Generated: {METADATA_JSON_PATH}")
