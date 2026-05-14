"""Validate generated JSON files against the schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spring_relaxation_ist.config import CURVES_JSON_PATH, METADATA_JSON_PATH
from spring_relaxation_ist.schema import MetadataOutput, RelaxationDataset

errors: list[str] = []

for path, schema in [
    (CURVES_JSON_PATH, RelaxationDataset),
    (METADATA_JSON_PATH, MetadataOutput),
]:
    if not path.exists():
        errors.append(f"Missing: {path}")
        continue
    with open(path) as f:
        data = json.load(f)
    try:
        schema.model_validate(data)  # type: ignore[attr-defined]
        print(f"OK: {path}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Invalid {path}: {exc}")

if errors:
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
