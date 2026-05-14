"""Tests for the prediction API."""

from __future__ import annotations

from pathlib import Path

import pytest

from spring_relaxation_ist.prediction import predict_relaxation

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SYNTHETIC_JSON = FIXTURES_DIR / "synthetic_curves.json"


def test_predict_valid() -> None:
    result = predict_relaxation(
        json_path=SYNTHETIC_JSON,
        material="Test Steel",
        temperature_C=20.0,
        stress_MPa=200.0,
    )
    assert result["material"] == "Test Steel"
    assert result["temperature_C"] == 20.0
    assert result["stress_MPa"] == 200.0
    assert result["relaxation_percent"] == pytest.approx(2.0)
    assert result["warnings"] == []


def test_predict_by_alias() -> None:
    result = predict_relaxation(
        json_path=SYNTHETIC_JSON,
        material="test-alloy",
        temperature_C=20.0,
        stress_MPa=300.0,
    )
    assert result["relaxation_percent"] == pytest.approx(3.5)


def test_predict_unknown_material() -> None:
    result = predict_relaxation(
        json_path=SYNTHETIC_JSON,
        material="Unknown Alloy",
        temperature_C=20.0,
        stress_MPa=200.0,
    )
    # Should return a result dict with error in warnings
    assert result["relaxation_percent"] is None
    assert len(result["warnings"]) > 0


def test_predict_out_of_range_stress() -> None:
    result = predict_relaxation(
        json_path=SYNTHETIC_JSON,
        material="Test Steel",
        temperature_C=20.0,
        stress_MPa=9999.0,
    )
    assert result["relaxation_percent"] is None
    assert any("outside the available range" in w for w in result["warnings"])


def test_predict_result_format() -> None:
    result = predict_relaxation(
        json_path=SYNTHETIC_JSON,
        material="Test Steel",
        temperature_C=20.0,
        stress_MPa=200.0,
    )
    required_keys = {
        "material",
        "temperature_C",
        "stress_MPa",
        "relaxation_percent",
        "interpolation_method",
        "source_curves",
        "warnings",
    }
    assert required_keys.issubset(result.keys())


def test_predict_missing_json() -> None:
    with pytest.raises(FileNotFoundError):
        predict_relaxation(
            json_path="/tmp/nonexistent_file_12345.json",
            material="Test Steel",
            temperature_C=20.0,
            stress_MPa=200.0,
        )


def test_predict_no_extrapolation() -> None:
    """Extrapolation outside temperature range should return warning, not crash."""
    result = predict_relaxation(
        json_path=SYNTHETIC_JSON,
        material="Test Steel",
        temperature_C=999.0,
        stress_MPa=200.0,
    )
    assert result["relaxation_percent"] is None
    assert len(result["warnings"]) > 0
