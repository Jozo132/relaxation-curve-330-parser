from __future__ import annotations

from spring_relaxation_ist.curve_calibration import (
    CurveControlPoints,
    load_curve_control_points,
    merge_curve_control_points,
    save_curve_control_points,
)


def test_curve_control_points_round_trip(tmp_path) -> None:
    calibration_path = tmp_path / "curve_controls.json"
    expected = {
        24: {
            "inconel-600": CurveControlPoints(start=(210.0, 4.8), end=(620.0, 2.9), line_style="dashed"),
        },
        25: {
            "elgiloy": CurveControlPoints(start=(190.0, 4.5), end=(620.0, 3.6), line_style="solid"),
            "18ni-co-mo-maraging-steel": CurveControlPoints(end=(980.0, 5.2)),
        },
    }

    save_curve_control_points(expected, calibration_path)

    assert load_curve_control_points(calibration_path) == expected


def test_merge_curve_control_points_prefers_primary_values() -> None:
    merged = merge_curve_control_points(
        CurveControlPoints(start=(200.0, 4.5), line_style="solid"),
        CurveControlPoints(start=(150.0, 5.1), end=(620.0, 3.6), line_style="dashed"),
    )

    assert merged == CurveControlPoints(start=(200.0, 4.5), end=(620.0, 3.6), line_style="solid")


def test_curve_control_points_persists_style_only_override(tmp_path) -> None:
    calibration_path = tmp_path / "curve_controls.json"
    expected = {
        26: {
            "tungsten-steel": CurveControlPoints(line_style="solid"),
        },
    }

    save_curve_control_points(expected, calibration_path)

    assert load_curve_control_points(calibration_path) == expected