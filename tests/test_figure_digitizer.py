"""Smoke tests for automatic figure digitisation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spring_relaxation_ist.config import PDF_PATH
from spring_relaxation_ist.figure_digitizer import (
    SUPPORTED_FIGURE_SPECS,
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