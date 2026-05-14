"""
Main build script for the IST Report 330 relaxation curve parser.

Usage:
    python scripts/build.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spring_relaxation_ist.config import (
    CURVES_JSON_PATH,
    FIGURE_PREVIEWS_DIR,
    GENERATED_DIR,
    METADATA_JSON_PATH,
    PDF_PATH,
)
from spring_relaxation_ist.curve_extractor import extract_curves
from spring_relaxation_ist.downloader import download_pdf, sha256_of_file
from spring_relaxation_ist.figure_digitizer import generate_supported_figure_previews


def main() -> None:
    # Step 1 & 2: Download PDF if missing
    dl_info = download_pdf()
    print(f"\nDownloaded or found PDF:\n{dl_info['local_pdf']}")

    # Step 3–6: Parse and extract curves
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    sha = sha256_of_file(PDF_PATH)
    dataset, metadata = extract_curves(PDF_PATH, pdf_sha256=sha)

    with open(CURVES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)

    preview_paths, preview_warnings = generate_supported_figure_previews(
        PDF_PATH,
        FIGURE_PREVIEWS_DIR,
    )

    # Step 7: Print generated file paths
    print(f"\nGenerated relaxation curve JSON:\n{CURVES_JSON_PATH}")
    print(f"\nGenerated metadata JSON:\n{METADATA_JSON_PATH}")
    if preview_paths:
        print(f"\nGenerated figure overlay previews:\n{FIGURE_PREVIEWS_DIR}")
    for warning in preview_warnings:
        print(f"Preview warning: {warning}")


if __name__ == "__main__":
    main()
