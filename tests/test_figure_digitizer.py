"""Smoke tests for automatic figure digitisation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spring_relaxation_ist.curve_calibration import CurveControlPoints
from spring_relaxation_ist.curve_calibration import load_curve_control_points
from spring_relaxation_ist.config import PDF_PATH
from spring_relaxation_ist.figure_digitizer import (
    AxisCalibration,
    PREVIEW_ANCHOR_COLOR,
    SUPPORTED_FIGURE_SPECS,
    _best_endpoint_assignment,
    _column_centers,
    _estimate_track_line_style,
    _fit_tick_positions_from_peaks,
    _merge_track_fragments,
    _trace_curve_between_control_points,
    _track_to_points,
    _trace_monotonic_curves,
    _track_y_at,
    digitize_supported_figure_curves,
    generate_supported_figure_previews,
)


@pytest.mark.skipif(not PDF_PATH.exists(), reason="report PDF not available locally")
def test_digitize_supported_figure_curves_smoke() -> None:
    curves_by_material, warnings, failed_items = digitize_supported_figure_curves(PDF_PATH)

    assert warnings == []
    assert failed_items == []
    assert {curve.temperature_C for curve in curves_by_material["titanium-alloy"]} == {100.0, 150.0}
    assert {curve.temperature_C for curve in curves_by_material["beryllium-copper"]} == {100.0, 150.0}
    assert {curve.temperature_C for curve in curves_by_material["phosphor-bronze"]} == {100.0}
    assert {curve.temperature_C for curve in curves_by_material["patented-carbon-steel"]} == {100.0, 150.0, 200.0}
    assert {curve.temperature_C for curve in curves_by_material["oil-hardened-and-tempered-steel"]} == {100.0, 150.0, 200.0}
    assert {curve.temperature_C for curve in curves_by_material["silicon-chromium-steel"]} == {100.0, 150.0, 200.0, 250.0}
    assert {curve.temperature_C for curve in curves_by_material["stainless-steel-18cr8ni"]} == {150.0, 200.0, 250.0, 300.0, 350.0}
    assert {curve.temperature_C for curve in curves_by_material["inconel-600"]} == {200.0, 250.0, 300.0, 350.0, 400.0}
    assert {curve.temperature_C for curve in curves_by_material["tungsten-steel"]} == {250.0, 300.0, 350.0, 400.0}
    assert {curve.temperature_C for curve in curves_by_material["18ni-co-mo-maraging-steel"]} == {300.0, 350.0, 400.0}
    assert {curve.temperature_C for curve in curves_by_material["a286"]} == {400.0}
    assert {curve.temperature_C for curve in curves_by_material["inconel-x750"]} == {400.0}
    assert {curve.temperature_C for curve in curves_by_material["elgiloy"]} == {350.0}

    temperatures_by_figure = {
        spec.figure_num: spec.temperature_C
        for spec in SUPPORTED_FIGURE_SPECS
    }
    for figure_num, materials in load_curve_control_points().items():
        figure_temperature = temperatures_by_figure[figure_num]
        for material_id, controls in materials.items():
            curve = next(curve for curve in curves_by_material[material_id] if curve.temperature_C == figure_temperature)
            if controls.start is not None:
                assert curve.points[0].stress_MPa == round(float(controls.start[0]), 1)
                assert curve.points[0].relaxation_percent == round(float(controls.start[1]), 3)
            if controls.end is not None:
                assert curve.points[-1].stress_MPa == round(float(controls.end[0]), 1)
                assert curve.points[-1].relaxation_percent == round(float(controls.end[1]), 3)

    for material_id in [
        "titanium-alloy",
        "beryllium-copper",
        "phosphor-bronze",
        "patented-carbon-steel",
        "oil-hardened-and-tempered-steel",
        "silicon-chromium-steel",
        "stainless-steel-18cr8ni",
        "inconel-600",
        "tungsten-steel",
        "18ni-co-mo-maraging-steel",
        "a286",
        "inconel-x750",
        "elgiloy",
    ]:
        for curve in curves_by_material[material_id]:
            assert curve.extraction_method == "automatic"
            assert curve.page > 0
            assert len(curve.points) >= 8
            assert curve.points[0].stress_MPa < curve.points[-1].stress_MPa


@pytest.mark.skipif(not PDF_PATH.exists(), reason="report PDF not available locally")
def test_generate_supported_figure_previews_smoke(tmp_path) -> None:
    preview_paths, warnings = generate_supported_figure_previews(PDF_PATH, tmp_path)

    assert warnings == []
    assert len(preview_paths) == len(SUPPORTED_FIGURE_SPECS)

    sample_preview = next(path for path in preview_paths if path.name == "figure_22_overlay.png")
    pixels = np.asarray(Image.open(sample_preview).convert("RGB"))
    orange_pixels = np.count_nonzero(
        (pixels[:, :, 0] == 255)
        & (pixels[:, :, 1] == 140)
        & (pixels[:, :, 2] == 0)
    )

    anchor_preview = next(path for path in preview_paths if path.name == "figure_25_overlay.png")
    anchor_pixels = np.asarray(Image.open(anchor_preview).convert("RGB"))
    anchor_marker_pixels = np.count_nonzero(
        (anchor_pixels[:, :, 0] == PREVIEW_ANCHOR_COLOR[0])
        & (anchor_pixels[:, :, 1] == PREVIEW_ANCHOR_COLOR[1])
        & (anchor_pixels[:, :, 2] == PREVIEW_ANCHOR_COLOR[2])
    )

    assert orange_pixels > 100
    assert anchor_marker_pixels > 40


def test_merge_track_fragments_merges_compatible_segments() -> None:
    tracks = [
        [(0, 10.0), (10, 8.0), (20, 6.0)],
        [(18, 5.7), (30, 4.0), (40, 2.0)],
        [(0, 22.0), (10, 20.0), (20, 18.0)],
    ]

    merged = _merge_track_fragments(tracks, max_gap=12, max_y_deviation=3.0)

    assert len(merged) == 2
    merged_track = max(merged, key=len)
    assert merged_track[0][0] == 0
    assert merged_track[-1][0] == 40


def test_trace_monotonic_curves_traces_solid_and_dashed_lines_separately() -> None:
    height = 80
    width = 72
    plot_mask = np.zeros((height, width), dtype=bool)

    for x_coord in range(5, width - 5):
        solid_y = int(round(14 + (0.09 * x_coord)))
        dashed_y = int(round(34 + (0.11 * x_coord)))

        for y_coord in range(solid_y - 1, solid_y + 2):
            plot_mask[y_coord, x_coord] = True

        if (x_coord // 5) % 2 == 0:
            for y_coord in range(dashed_y - 1, dashed_y + 2):
                plot_mask[y_coord, x_coord] = True

    tracks = _trace_monotonic_curves(plot_mask, expected_count=2, reference_fraction=0.5)

    assert len(tracks) == 2
    assert tracks[0][0][0] <= 8
    assert tracks[0][-1][0] >= width - 8
    assert tracks[1][0][0] <= 12
    assert tracks[1][-1][0] >= width - 8

    reference_x = int(width * 0.5)
    assert _track_y_at(tracks[0], reference_x) < _track_y_at(tracks[1], reference_x)


def test_trace_monotonic_curves_keeps_crossing_solid_and_dashed_tracks_distinct() -> None:
    height = 96
    width = 120
    plot_mask = np.zeros((height, width), dtype=bool)

    for x_coord in range(6, width - 6):
        solid_y = int(round(24 + (0.20 * x_coord) + (0.0008 * ((x_coord - 60) ** 2))))
        dashed_y = int(round(62 - (0.18 * x_coord) + (0.0008 * ((x_coord - 60) ** 2))))

        for y_coord in range(solid_y - 1, solid_y + 2):
            plot_mask[y_coord, x_coord] = True

        if (x_coord // 4) % 2 == 0:
            for y_coord in range(dashed_y - 1, dashed_y + 2):
                plot_mask[y_coord, x_coord] = True

    tracks = _trace_monotonic_curves(
        plot_mask,
        expected_count=2,
        reference_fraction=0.25,
        expected_line_styles=("solid", "dashed"),
    )

    assert len(tracks) == 2
    assert tracks[0][0][0] <= 10
    assert tracks[0][-1][0] >= width - 10
    assert tracks[1][0][0] <= 10
    assert tracks[1][-1][0] >= width - 10
    assert _track_y_at(tracks[0], 20) < _track_y_at(tracks[1], 20)
    assert _track_y_at(tracks[0], width - 8) > _track_y_at(tracks[1], width - 8)
    assert {_estimate_track_line_style(track) for track in tracks} == {"solid", "dashed"}


def test_best_endpoint_assignment_prefers_line_style_override() -> None:
    solid_track = [(x_coord, 20.0 + (0.1 * x_coord)) for x_coord in range(0, 21)]
    dashed_track = [(0, 36.0), (4, 35.6), (8, 35.2), (12, 34.8), (16, 34.4), (20, 34.0)]
    calibration = AxisCalibration(
        x_ticks=((0, 0.0), (20, 20.0)),
        y_ticks=((0, 0.0), (5, 5.0), (10, 10.0), (15, 15.0)),
        x_slope=1.0,
        x_intercept=0.0,
        y_slope=1.0,
        y_intercept=0.0,
    )

    assigned = _best_endpoint_assignment(
        tracks=[dashed_track, solid_track],
        materials=("solid-material", "dashed-material"),
        bounds=(0, 0, 20, 40),
        calibration=calibration,
        figure_controls={
            "solid-material": CurveControlPoints(line_style="solid"),
            "dashed-material": CurveControlPoints(line_style="dashed"),
        },
    )

    assert _estimate_track_line_style(solid_track) == "solid"
    assert _estimate_track_line_style(dashed_track) == "dashed"
    assert assigned == [solid_track, dashed_track]


def test_trace_curve_between_control_points_prefers_controlled_branch() -> None:
    height = 96
    width = 120
    plot_mask = np.zeros((height, width), dtype=bool)

    for x_coord in range(6, width - 6):
        lower_curve_y = int(round(30 + (0.14 * x_coord) + (0.0007 * ((x_coord - 60) ** 2))))
        upper_curve_y = int(round(54 - (0.12 * x_coord) + (0.0007 * ((x_coord - 60) ** 2))))
        for y_coord in range(lower_curve_y - 1, lower_curve_y + 2):
            plot_mask[y_coord, x_coord] = True
        if (x_coord // 4) % 2 == 0:
            for y_coord in range(upper_curve_y - 1, upper_curve_y + 2):
                plot_mask[y_coord, x_coord] = True

    column_centers_by_x = [
        [float(group.mean()) for group in np.split(np.flatnonzero(plot_mask[:, column_index]), np.where(np.diff(np.flatnonzero(plot_mask[:, column_index])) > 1)[0] + 1) if len(group) >= 2]
        if np.count_nonzero(plot_mask[:, column_index])
        else []
        for column_index in range(width)
    ]
    calibration = AxisCalibration(
        x_ticks=((0, 0.0), (width - 1, float(width - 1))),
        y_ticks=((0, 0.0), (5, 5.0), (10, 10.0), (15, 15.0)),
        x_slope=1.0,
        x_intercept=0.0,
        y_slope=1.0,
        y_intercept=0.0,
    )

    controlled_track = _trace_curve_between_control_points(
        column_centers_by_x,
        plot_height=height,
        bounds=(0, 0, width - 1, height - 1),
        calibration=calibration,
        control_points=CurveControlPoints(start=(8.0, 53.0), end=(112.0, 42.0), line_style="dashed"),
    )

    assert controlled_track is not None
    assert controlled_track[0][0] <= 10
    assert controlled_track[-1][0] >= 110
    assert _track_y_at(controlled_track, 20) > 45.0
    assert _track_y_at(controlled_track, 100) > 40.0


def test_trace_curve_between_control_points_follows_solid_crossing_curve() -> None:
    height = 120
    width = 140
    plot_mask = np.zeros((height, width), dtype=bool)

    sample_x = np.arange(width, dtype=float)
    target_curve = np.interp(sample_x, [10, 40, 70, 100, 120], [70, 66, 55, 32, 24])
    other_curve = np.interp(sample_x, [10, 40, 70, 100, 120], [88, 80, 58, 24, 18])

    for x_coord in range(10, 121):
        target_y = int(round(float(target_curve[x_coord])))
        other_y = int(round(float(other_curve[x_coord])))
        for y_coord in range(target_y - 1, target_y + 2):
            plot_mask[y_coord, x_coord] = True
        for y_coord in range(other_y - 1, other_y + 2):
            plot_mask[y_coord, x_coord] = True

    column_centers_by_x = [
        _column_centers(plot_mask[:, column_index])
        for column_index in range(width)
    ]
    calibration = AxisCalibration(
        x_ticks=((0, 0.0), (width - 1, float(width - 1))),
        y_ticks=((0, 0.0), (5, 5.0), (10, 10.0), (15, 15.0)),
        x_slope=1.0,
        x_intercept=0.0,
        y_slope=1.0,
        y_intercept=0.0,
    )

    controlled_track = _trace_curve_between_control_points(
        column_centers_by_x,
        plot_height=height,
        bounds=(0, 0, width - 1, height - 1),
        calibration=calibration,
        control_points=CurveControlPoints(start=(10.0, 70.0), end=(120.0, 24.0), line_style="solid"),
    )

    assert controlled_track is not None
    assert abs(_track_y_at(controlled_track, 20) - float(target_curve[20])) <= 5.0
    assert abs(_track_y_at(controlled_track, 70) - float(target_curve[70])) <= 5.0
    assert abs(_track_y_at(controlled_track, 110) - float(target_curve[110])) <= 5.0


def test_fit_tick_positions_from_peaks_infers_missing_x_tick() -> None:
    fitted = _fit_tick_positions_from_peaks(
        ordered_tick_values=[100, 200, 300, 400, 500],
        detected_positions=[210.0, 610.0, 810.0, 1010.0],
        fixed_anchors=[(0, 10.0)],
        positive_slope=True,
    )

    assert fitted is not None
    assert fitted[100] == 210.0
    assert fitted[300] == 610.0
    assert fitted[400] == 810.0
    assert fitted[500] == 1010.0
    assert fitted[200] == pytest.approx(410.0)


def test_fit_tick_positions_from_peaks_infers_missing_y_tick() -> None:
    fitted = _fit_tick_positions_from_peaks(
        ordered_tick_values=[15, 10, 5],
        detected_positions=[40.0, 60.0],
        fixed_anchors=[(0, 100.0)],
        positive_slope=False,
    )

    assert fitted is not None
    assert fitted[15] == 40.0
    assert fitted[10] == 60.0
    assert fitted[5] == pytest.approx(80.0)


def test_track_to_points_resamples_within_manual_start_and_end_bounds() -> None:
    track = [
        (0, 1.0),
        (20, 2.0),
        (40, 3.0),
        (60, 4.0),
        (80, 5.0),
        (100, 6.0),
    ]
    calibration = AxisCalibration(
        x_ticks=((0, 0.0), (100, 100.0)),
        y_ticks=((0, 0.0), (5, 5.0), (10, 10.0), (15, 15.0)),
        x_slope=1.0,
        x_intercept=0.0,
        y_slope=1.0,
        y_intercept=0.0,
    )

    points = _track_to_points(
        track,
        bounds=(0, 0, 100, 15),
        calibration=calibration,
        control_points=CurveControlPoints(start=(40.0, 3.2), end=(60.0, 4.8)),
    )

    assert len(points) >= 8
    assert points[0].stress_MPa == 40.0
    assert points[0].relaxation_percent == 3.2
    assert points[-1].stress_MPa == 60.0
    assert points[-1].relaxation_percent == 4.8
    assert all(40.0 <= point.stress_MPa <= 60.0 for point in points)