"""Tests for the Pydantic schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from spring_relaxation_ist.schema import (
    Material,
    RelaxationCurve,
    RelaxationDataset,
    RelaxationPoint,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_relaxation_point_valid() -> None:
    pt = RelaxationPoint(stress_MPa=100.0, relaxation_percent=2.5)
    assert pt.stress_MPa == 100.0
    assert pt.relaxation_percent == 2.5


def test_relaxation_point_invalid() -> None:
    with pytest.raises(ValidationError):
        RelaxationPoint(stress_MPa="bad", relaxation_percent=2.5)  # type: ignore[arg-type]


def test_relaxation_curve_placeholder() -> None:
    curve = RelaxationCurve(
        curve_id="test_fig1_T20C",
        source_type="figure",
        source_ref="Figure 1",
        page=5,
        temperature_C=20.0,
        points=[],
        extraction_method="placeholder",
        extraction_confidence=0.0,
        warnings=["No data"],
    )
    assert curve.extraction_method == "placeholder"
    assert curve.points == []


def test_load_synthetic_fixture() -> None:
    path = FIXTURES_DIR / "synthetic_curves.json"
    with open(path) as f:
        data = json.load(f)
    dataset = RelaxationDataset.model_validate(data)
    assert dataset.dataset == "Synthetic test dataset"
    assert len(dataset.materials) == 1
    assert dataset.materials[0].material_name == "Test Steel"


def test_material_aliases() -> None:
    mat = Material(
        material_id="test",
        material_name="Test Steel",
        aliases=["TS", "test-alloy"],
        family="steel",
    )
    assert "TS" in mat.aliases


def test_dataset_round_trip() -> None:
    path = FIXTURES_DIR / "synthetic_curves.json"
    with open(path) as f:
        original = json.load(f)
    dataset = RelaxationDataset.model_validate(original)
    dumped = dataset.model_dump()
    assert dumped["dataset"] == original["dataset"]
    assert len(dumped["materials"]) == len(original["materials"])
