from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import spring_relaxation_ist.calibration_ui as calibration_ui
from spring_relaxation_ist.calibration_ui import (
    _FigureCalibrationView,
    _build_figure_view,
    _calculate_fit_zoom,
    _format_axis_coordinates,
    _hit_test_control_marker,
    _preview_to_axis_point,
    _view_offset_for_pivot,
)
from spring_relaxation_ist.curve_calibration import CurveControlPoints
from spring_relaxation_ist.figure_digitizer import AxisCalibration, FigureAnalysisResult, FigureDigitizerSpec


def test_preview_to_axis_point_uses_parsed_axis() -> None:
    figure_view = _make_figure_view()

    point = _preview_to_axis_point(figure_view, 600.0, 500.0)

    assert point == (500.0, 10.0)
    assert _format_axis_coordinates("Cursor", point) == "Cursor: X 500.0 MPa | Y 10.000%"


def test_preview_to_axis_point_rejects_margin_coordinates() -> None:
    figure_view = _make_figure_view()

    assert _preview_to_axis_point(figure_view, 50.0, 500.0) is None
    assert _preview_to_axis_point(figure_view, 600.0, 950.0) is None


def test_build_figure_view_passes_manual_controls(monkeypatch) -> None:
    figure_view = _make_figure_view()
    captured: dict[str, object] = {}

    def fake_analyze(
        pdf_path: Path,
        spec: FigureDigitizerSpec,
        manual_curve_controls: dict[str, CurveControlPoints] | None = None,
    ) -> FigureAnalysisResult:
        captured["pdf_path"] = pdf_path
        captured["spec"] = spec
        captured["controls"] = manual_curve_controls
        return figure_view.analysis

    def fake_render(analysis: FigureAnalysisResult, show_curve_controls: bool = True) -> Image.Image:
        captured["render_analysis"] = analysis
        captured["show_curve_controls"] = show_curve_controls
        return figure_view.preview_image

    monkeypatch.setattr(calibration_ui, "_analyze_figure", fake_analyze)
    monkeypatch.setattr(calibration_ui, "_render_figure_preview", fake_render)

    controls = {"test-material": CurveControlPoints(start=(200.0, 4.5), end=(600.0, 3.0), line_style="dashed")}
    rebuilt_view = _build_figure_view(Path("dummy.pdf"), figure_view.analysis.spec, controls)

    assert captured["pdf_path"] == Path("dummy.pdf")
    assert captured["spec"] == figure_view.analysis.spec
    assert captured["controls"] == controls
    assert captured["render_analysis"] == figure_view.analysis
    assert captured["show_curve_controls"] is False
    assert rebuilt_view.analysis == figure_view.analysis
    assert rebuilt_view.preview_image == figure_view.preview_image


def test_calculate_fit_zoom_scales_to_full_extents() -> None:
    assert _calculate_fit_zoom((2000, 1000), (1000, 500)) == 0.5
    assert _calculate_fit_zoom((400, 300), (1000, 500)) == 1.0


def test_hit_test_control_marker_finds_nearest_marker() -> None:
    figure_view = _make_figure_view()
    controls = {
        "test-material": CurveControlPoints(start=(200.0, 10.0), end=(600.0, 5.0)),
        "other-material": CurveControlPoints(end=(800.0, 6.0)),
    }

    assert _hit_test_control_marker(figure_view, controls, 300.0, 500.0, 1.0) == ("test-material", "start")
    assert _hit_test_control_marker(figure_view, controls, 700.0, 700.0, 1.0) == ("test-material", "end")
    assert _hit_test_control_marker(figure_view, controls, 20.0, 20.0, 1.0) is None


def test_view_offset_for_pivot_clamps_to_scrollable_range() -> None:
    assert _view_offset_for_pivot(2000, 1000, 800.0, 300.0) == 500.0
    assert _view_offset_for_pivot(2000, 1000, 200.0, 300.0) == 0.0
    assert _view_offset_for_pivot(2000, 1000, 1900.0, 300.0) == 1000.0
    assert _view_offset_for_pivot(600, 1000, 400.0, 300.0) == 0.0


def _make_figure_view() -> _FigureCalibrationView:
    spec = FigureDigitizerSpec(
        figure_num=99,
        page=1,
        temperature_C=123.0,
        x_max=1000.0,
        material_ids_top_to_bottom=("test-material",),
        reference_fraction=0.5,
    )
    calibration = AxisCalibration(
        x_ticks=((0, 100.0), (1000, 1100.0)),
        y_ticks=((0, 900.0), (5, 700.0), (10, 500.0), (15, 300.0)),
        x_slope=1.0,
        x_intercept=100.0,
        y_slope=-40.0,
        y_intercept=900.0,
    )
    analysis = FigureAnalysisResult(
        spec=spec,
        page_image=np.zeros((1000, 1200), dtype=np.uint8),
        plot_bounds=(100, 300, 1100, 900),
        calibration=calibration,
        tracks=[],
        curves={},
        curve_controls={},
    )
    return _FigureCalibrationView(
        analysis=analysis,
        preview_image=Image.new("RGB", (1200, 1000), "white"),
        crop_box=(0, 0, 1200, 1000),
    )