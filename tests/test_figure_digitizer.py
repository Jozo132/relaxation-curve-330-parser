"""Smoke tests for automatic figure digitisation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spring_relaxation_ist.config import PDF_PATH
from spring_relaxation_ist.figure_digitizer import (
    SUPPORTED_FIGURE_SPECS,
    _merge_track_fragments,
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

    figure_26_a286 = next(curve for curve in curves_by_material["a286"] if curve.temperature_C == 400.0)
    figure_26_inconel_600 = next(
        curve for curve in curves_by_material["inconel-600"] if curve.temperature_C == 400.0
    )
    figure_26_tungsten = next(
        curve for curve in curves_by_material["tungsten-steel"] if curve.temperature_C == 400.0
    )
    assert figure_26_tungsten.points[-1].stress_MPa > figure_26_a286.points[-1].stress_MPa
    assert figure_26_inconel_600.points[-1].stress_MPa > figure_26_a286.points[-1].stress_MPa

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

    assert orange_pixels > 100


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