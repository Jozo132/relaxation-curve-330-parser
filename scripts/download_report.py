"""Standalone script to download the IST Report 330 PDF."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spring_relaxation_ist.downloader import download_pdf

if __name__ == "__main__":
    result = download_pdf()
    print(f"PDF ready: {result['local_pdf']}")
    print(f"SHA256: {result['sha256']}")
