"""Prediction API for relaxation values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .interpolation import InterpolationError, interpolate_relaxation
from .schema import Material, RelaxationDataset


def _load_dataset(json_path: str | Path) -> RelaxationDataset:
    """Load and validate a relaxation dataset from a JSON file."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Relaxation curves JSON not found: {path}. "
            "Run 'npm run build' or 'python -m spring_relaxation_ist extract' first."
        )
    with open(path) as f:
        data = json.load(f)
    return RelaxationDataset.model_validate(data)


def _find_material(dataset: RelaxationDataset, material_name: str) -> Material:
    """Find a material by name or alias (case-insensitive)."""
    name_lower = material_name.lower()
    for mat in dataset.materials:
        if mat.material_name.lower() == name_lower:
            return mat
        if any(alias.lower() == name_lower for alias in mat.aliases):
            return mat
    available = [m.material_name for m in dataset.materials]
    raise ValueError(
        f"Unknown material: '{material_name}'. "
        f"Available materials: {available}"
    )


def predict_relaxation(
    json_path: str | Path,
    material: str,
    temperature_C: float,
    stress_MPa: float,
) -> dict[str, Any]:
    """
    Predict relaxation percent for a material at given temperature and stress.

    Parameters
    ----------
    json_path : path to the generated relaxation curves JSON
    material  : material name (or alias)
    temperature_C : temperature in °C
    stress_MPa    : stress in MPa

    Returns a dict with keys: material, temperature_C, stress_MPa,
    relaxation_percent, interpolation_method, source_curves, warnings.
    """
    dataset = _load_dataset(json_path)
    try:
        mat_obj = _find_material(dataset, material)
    except ValueError as exc:
        return {
            "material": material,
            "temperature_C": temperature_C,
            "stress_MPa": stress_MPa,
            "relaxation_percent": None,
            "interpolation_method": "failed",
            "source_curves": [],
            "warnings": [str(exc)],
        }

    try:
        relaxation, method, warnings = interpolate_relaxation(
            mat_obj, temperature_C, stress_MPa
        )
    except InterpolationError as exc:
        return {
            "material": material,
            "temperature_C": temperature_C,
            "stress_MPa": stress_MPa,
            "relaxation_percent": None,
            "interpolation_method": "failed",
            "source_curves": [],
            "warnings": [str(exc)],
        }

    source_curves = [
        c.curve_id
        for c in mat_obj.curves
        if c.extraction_method != "placeholder"
        and c.temperature_C == temperature_C
        and len(c.points) >= 2
    ]

    return {
        "material": mat_obj.material_name,
        "temperature_C": temperature_C,
        "stress_MPa": stress_MPa,
        "relaxation_percent": relaxation,
        "interpolation_method": method,
        "source_curves": source_curves,
        "warnings": warnings,
    }
