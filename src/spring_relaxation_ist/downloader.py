"""PDF downloader for IST Report 330."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import IST_REPORT_URL, PDF_PATH


def sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_pdf(path: Path) -> bool:
    """Return True if the file starts with the PDF magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except OSError:
        return False


def download_pdf(
    url: str = IST_REPORT_URL,
    dest: Path = PDF_PATH,
    timeout: int = 60,
) -> dict:
    """
    Download the IST Report 330 PDF if it does not already exist.

    Returns a dict with: url, local_pdf, sha256, file_size_bytes,
    download_timestamp, already_existed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and is_valid_pdf(dest):
        print(f"PDF already exists locally: {dest}")
        return {
            "url": url,
            "local_pdf": str(dest),
            "sha256": sha256_of_file(dest),
            "file_size_bytes": dest.stat().st_size,
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "already_existed": True,
        }

    print(f"Downloading PDF from:\n  {url}")
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download PDF: {exc}") from exc

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)

    if not is_valid_pdf(dest):
        os.remove(dest)
        raise RuntimeError("Downloaded file is not a valid PDF (missing %PDF header).")

    result = {
        "url": url,
        "local_pdf": str(dest),
        "sha256": sha256_of_file(dest),
        "file_size_bytes": dest.stat().st_size,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "already_existed": False,
    }
    print(f"Downloaded PDF ({result['file_size_bytes']:,} bytes) to:\n  {dest}")
    return result
