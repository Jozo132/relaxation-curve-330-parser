"""Tests for the interpolation module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spring_relaxation_ist.interpolation import InterpolationError, interpolate_relaxation
from spring_relaxation_ist.schema import Material, RelaxationDataset

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_test_material() -> Material:
    path = FIXTURES_DIR / "synthetic_curves.json"
    with open(path) as f:
        data = json.load(f)
    dataset = RelaxationDataset.model_validate(data)
    return dataset.materials[0]


def test_interpolate_at_known_stress() -> None:
    mat = _load_test_material()
    relaxation, method, warnings = interpolate_relaxation(mat, 20.0, 200.0)
    assert relaxation == pytest.approx(2.0)
    assert "linear_interpolation" in method
    assert warnings == []


def test_interpolate_midpoint() -> None:
    mat = _load_test_material()
    # Between 100 MPa (1.0%) and 200 MPa (2.0%) — midpoint should be ~1.5%
    relaxation, method, warnings = interpolate_relaxation(mat, 20.0, 150.0)
    assert relaxation == pytest.approx(1.5)


def test_no_extrapolation_below() -> None:
    mat = _load_test_material()
    with pytest.raises(InterpolationError, match="outside the available range"):
        interpolate_relaxation(mat, 20.0, 50.0)


def test_no_extrapolation_above() -> None:
    mat = _load_test_material()
    with pytest.raises(InterpolationError, match="outside the available range"):
        interpolate_relaxation(mat, 20.0, 500.0)


def test_unknown_temperature() -> None:
    mat = _load_test_material()
    with pytest.raises(InterpolationError, match="outside the available range"):
        interpolate_relaxation(mat, 999.0, 200.0)


def test_placeholder_curve_raises() -> None:
    """A material with only placeholder curves should raise InterpolationError."""
    mat = Material(
        material_id="placeholder-mat",
        material_name="Placeholder Material",
        family="steel",
        curves=[],
    )
    with pytest.raises(InterpolationError, match="No usable"):
        interpolate_relaxation(mat, 20.0, 200.0)
