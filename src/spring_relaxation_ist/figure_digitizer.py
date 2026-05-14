from __future__ import annotations

import atexit
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .config import GENERATED_DIR
from .schema import RelaxationCurve, RelaxationPoint

Y_AXIS_MAX = 15.0
PDF_RENDER_SCALE = 3.0
COARSE_RENDER_SCALE = 2.0
PREVIEW_AXIS_COLOR = (255, 140, 0, 255)
PREVIEW_LEGEND_BACKGROUND = (255, 255, 255, 224)
PREVIEW_CURVE_COLORS: tuple[tuple[int, int, int, int], ...] = (
    (0, 92, 184, 255),
    (220, 50, 47, 255),
    (42, 161, 152, 255),
    (181, 137, 0, 255),
    (108, 113, 196, 255),
    (211, 54, 130, 255),
)
MATERIAL_DISPLAY_NAMES: dict[str, str] = {
    "phosphor-bronze": "Phosphor Bronze",
    "titanium-alloy": "Titanium Alloy",
    "beryllium-copper": "Beryllium Copper",
    "patented-carbon-steel": "Patented Carbon Steel",
    "oil-hardened-and-tempered-steel": "Oil Hardened and Tempered Steel",
    "silicon-chromium-steel": "Silicon-Chromium Steel",
    "stainless-steel-18cr8ni": "18Cr/8Ni Stainless Steel",
    "inconel-600": "Inconel 600",
    "tungsten-steel": "Tungsten Steel",
    "18ni-co-mo-maraging-steel": "18Ni-Co-Mo Maraging Steel",
    "elgiloy": "Elgiloy",
    "a286": "A286",
    "inconel-x750": "Inconel X-750",
}

FIGURE_PAGE_HINTS: dict[int, int] = {
    18: 29,
    19: 30,
    20: 31,
    21: 32,
    22: 33,
    23: 34,
    24: 35,
    25: 36,
    26: 37,
    27: 38,
    28: 39,
    29: 40,
}


@dataclass(frozen=True)
class FigureDigitizerSpec:
    figure_num: int
    page: int
    temperature_C: float
    x_max: float
    material_ids_top_to_bottom: tuple[str, ...]
    reference_fraction: float


@dataclass(frozen=True)
class AxisCalibration:
    x_ticks: tuple[tuple[int, float], ...]
    y_ticks: tuple[tuple[int, float], ...]
    x_slope: float
    x_intercept: float
    y_slope: float
    y_intercept: float

    def stress_to_x(self, stress: float) -> float:
        return self.x_intercept + (self.x_slope * stress)

    def relaxation_to_y(self, relaxation: float) -> float:
        return self.y_intercept + (self.y_slope * relaxation)

    def x_to_stress(self, x_coord: float) -> float:
        return (x_coord - self.x_intercept) / self.x_slope

    def y_to_relaxation(self, y_coord: float) -> float:
        return (y_coord - self.y_intercept) / self.y_slope


@dataclass
class FigureAnalysisResult:
    spec: FigureDigitizerSpec
    page_image: np.ndarray[Any, np.dtype[np.uint8]]
    plot_bounds: tuple[int, int, int, int]
    calibration: AxisCalibration
    tracks: list[list[tuple[int, float]]]
    curves: dict[str, RelaxationCurve]


SUPPORTED_FIGURE_SPECS: tuple[FigureDigitizerSpec, ...] = (
    FigureDigitizerSpec(
        figure_num=18,
        page=29,
        temperature_C=100.0,
        x_max=700.0,
        material_ids_top_to_bottom=(
            "phosphor-bronze",
            "titanium-alloy",
            "beryllium-copper",
        ),
        reference_fraction=0.52,
    ),
    FigureDigitizerSpec(
        figure_num=19,
        page=30,
        temperature_C=150.0,
        x_max=800.0,
        material_ids_top_to_bottom=(
            "titanium-alloy",
            "beryllium-copper",
        ),
        reference_fraction=0.70,
    ),
    FigureDigitizerSpec(
        figure_num=20,
        page=31,
        temperature_C=100.0,
        x_max=1000.0,
        material_ids_top_to_bottom=(
            "patented-carbon-steel",
            "oil-hardened-and-tempered-steel",
            "silicon-chromium-steel",
        ),
        reference_fraction=0.72,
    ),
    FigureDigitizerSpec(
        figure_num=21,
        page=32,
        temperature_C=150.0,
        x_max=1000.0,
        material_ids_top_to_bottom=(
            "patented-carbon-steel",
            "oil-hardened-and-tempered-steel",
            "silicon-chromium-steel",
            "stainless-steel-18cr8ni",
        ),
        reference_fraction=0.78,
    ),
    FigureDigitizerSpec(
        figure_num=22,
        page=33,
        temperature_C=200.0,
        x_max=900.0,
        material_ids_top_to_bottom=(
            "patented-carbon-steel",
            "oil-hardened-and-tempered-steel",
            "silicon-chromium-steel",
            "stainless-steel-18cr8ni",
            "inconel-600",
        ),
        reference_fraction=0.56,
    ),
    FigureDigitizerSpec(
        figure_num=23,
        page=34,
        temperature_C=250.0,
        x_max=900.0,
        material_ids_top_to_bottom=(
            "silicon-chromium-steel",
            "stainless-steel-18cr8ni",
            "tungsten-steel",
            "inconel-600",
        ),
        reference_fraction=0.58,
    ),
    FigureDigitizerSpec(
        figure_num=24,
        page=35,
        temperature_C=300.0,
        x_max=1000.0,
        material_ids_top_to_bottom=(
            "stainless-steel-18cr8ni",
            "tungsten-steel",
            "inconel-600",
            "18ni-co-mo-maraging-steel",
        ),
        reference_fraction=0.60,
    ),
    FigureDigitizerSpec(
        figure_num=25,
        page=36,
        temperature_C=350.0,
        x_max=1000.0,
        material_ids_top_to_bottom=(
            "stainless-steel-18cr8ni",
            "tungsten-steel",
            "inconel-600",
            "18ni-co-mo-maraging-steel",
            "elgiloy",
        ),
        reference_fraction=0.60,
    ),
    FigureDigitizerSpec(
        figure_num=26,
        page=37,
        temperature_C=400.0,
        x_max=1000.0,
        material_ids_top_to_bottom=(
            "a286",
            "inconel-600",
            "tungsten-steel",
            "inconel-x750",
            "18ni-co-mo-maraging-steel",
        ),
        reference_fraction=0.58,
    ),
)

_PAGE_IMAGE_CACHE: dict[
    tuple[str, int, float, tuple[int, int, int, int] | None],
    np.ndarray[Any, np.dtype[np.uint8]],
] = {}
_FIGURE_ANALYSIS_CACHE: dict[tuple[str, int], FigureAnalysisResult] = {}
_PDF_DOCUMENT_CACHE: dict[str, Any] = {}
_RENDER_CACHE_DIR = GENERATED_DIR / "figure_render_cache"


def _close_cached_pdf_documents() -> None:
    for document in _PDF_DOCUMENT_CACHE.values():
        try:
            document.close()
        except Exception:  # noqa: BLE001
            continue


atexit.register(_close_cached_pdf_documents)


def digitize_supported_figure_curves(
    pdf_path: Path,
) -> tuple[dict[str, list[RelaxationCurve]], list[str], list[dict[str, Any]]]:
    curves_by_material: dict[str, list[RelaxationCurve]] = {}
    warnings: list[str] = []
    failed_items: list[dict[str, Any]] = []

    for spec in SUPPORTED_FIGURE_SPECS:
        try:
            digitized = _digitize_figure(pdf_path, spec)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Figure {spec.figure_num} digitisation failed: {exc}")
            failed_items.append(
                {
                    "source_ref": f"Figure {spec.figure_num}",
                    "temperature_C": spec.temperature_C,
                    "reason": f"Automatic raster digitisation failed: {exc}",
                }
            )
            continue

        for material_id, curve in digitized.items():
            curves_by_material.setdefault(material_id, []).append(curve)

    return curves_by_material, warnings, failed_items


def generate_supported_figure_previews(
    pdf_path: Path,
    output_dir: Path,
) -> tuple[list[Path], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_paths: list[Path] = []
    warnings: list[str] = []

    for spec in SUPPORTED_FIGURE_SPECS:
        try:
            analysis = _analyze_figure(pdf_path, spec)
            preview = _render_figure_preview(analysis)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Figure {spec.figure_num} preview generation failed: {exc}")
            continue

        preview_path = output_dir / f"figure_{spec.figure_num:02d}_overlay.png"
        preview.save(preview_path)
        preview_paths.append(preview_path)

    return preview_paths, warnings


def _digitize_figure(
    pdf_path: Path,
    spec: FigureDigitizerSpec,
) -> dict[str, RelaxationCurve]:
    return _analyze_figure(pdf_path, spec).curves


def _analyze_figure(
    pdf_path: Path,
    spec: FigureDigitizerSpec,
) -> FigureAnalysisResult:
    cache_key = (str(pdf_path.resolve()), spec.figure_num)
    cached = _FIGURE_ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    grayscale = _render_figure_grayscale(pdf_path, spec.page)
    dark = grayscale < 170
    bounds = _detect_plot_bounds(dark)
    calibration = _calibrate_axes(dark, bounds, spec.x_max)
    plot_mask = _extract_plot_mask(dark, bounds)
    tracks = _trace_monotonic_curves(
        plot_mask,
        expected_count=len(spec.material_ids_top_to_bottom),
        reference_fraction=spec.reference_fraction,
    )
    tracks = _reorder_tracks_for_spec(spec, tracks)

    if len(tracks) != len(spec.material_ids_top_to_bottom):
        raise ValueError(
            f"expected {len(spec.material_ids_top_to_bottom)} curves, found {len(tracks)}"
        )

    curves: dict[str, RelaxationCurve] = {}
    for material_id, track in zip(spec.material_ids_top_to_bottom, tracks):
        points = _track_to_points(track, bounds, calibration)
        if len(points) < 8:
            raise ValueError(
                f"curve for material {material_id} yielded only {len(points)} points"
            )

        curves[material_id] = RelaxationCurve(
            curve_id=f"{material_id}_fig{spec.figure_num}_T{int(spec.temperature_C)}C",
            source_type="figure",
            source_ref=f"Figure {spec.figure_num}",
            page=spec.page,
            temperature_C=spec.temperature_C,
            confidence_level=None,
            points=points,
            extraction_method="automatic",
            extraction_confidence=0.68,
            warnings=[
                "Automatically digitised from scanned figure using image-based tracing; verify against the source page before redistribution."
            ],
        )

    result = FigureAnalysisResult(
        spec=spec,
        page_image=grayscale,
        plot_bounds=bounds,
        calibration=calibration,
        tracks=tracks,
        curves=curves,
    )
    _FIGURE_ANALYSIS_CACHE[cache_key] = result
    return result


def _render_figure_preview(analysis: FigureAnalysisResult) -> Image.Image:
    base_image = Image.fromarray(analysis.page_image).convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    _draw_preview_axes(overlay_draw, analysis.calibration)

    for index, material_id in enumerate(analysis.spec.material_ids_top_to_bottom):
        color = PREVIEW_CURVE_COLORS[index % len(PREVIEW_CURVE_COLORS)]
        curve = analysis.curves[material_id]
        page_points = [
            (
                int(round(analysis.calibration.stress_to_x(point.stress_MPa))),
                int(round(analysis.calibration.relaxation_to_y(point.relaxation_percent))),
            )
            for point in curve.points
        ]
        if len(page_points) < 2:
            continue

        overlay_draw.line(page_points, fill=color, width=6)
        for point in _sample_track_points(page_points):
            overlay_draw.ellipse(
                (point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5),
                fill=color,
            )

    preview = Image.alpha_composite(base_image, overlay)
    preview = preview.crop(_preview_crop_box(base_image.size, analysis.plot_bounds))
    _draw_preview_legend(preview, analysis.spec.material_ids_top_to_bottom, analysis.spec)
    return preview.convert("RGB")


def _render_figure_grayscale(pdf_path: Path, page_number: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if COARSE_RENDER_SCALE >= PDF_RENDER_SCALE:
        return _render_page_grayscale(pdf_path, page_number, scale=PDF_RENDER_SCALE)

    coarse_grayscale = _render_page_grayscale(pdf_path, page_number, scale=COARSE_RENDER_SCALE)
    coarse_dark = coarse_grayscale < 170
    coarse_bounds = _detect_plot_bounds(coarse_dark)
    clip_rect = _clip_rect_from_bounds(pdf_path, page_number, coarse_bounds, COARSE_RENDER_SCALE)
    return _render_page_grayscale(pdf_path, page_number, scale=PDF_RENDER_SCALE, clip_rect=clip_rect)


def _render_page_grayscale(
    pdf_path: Path,
    page_number: int,
    scale: float,
    clip_rect: tuple[float, float, float, float] | None = None,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    clip_key: tuple[int, int, int, int] | None = None
    if clip_rect is not None:
        clip_key = (
            int(round(clip_rect[0] * 1000.0)),
            int(round(clip_rect[1] * 1000.0)),
            int(round(clip_rect[2] * 1000.0)),
            int(round(clip_rect[3] * 1000.0)),
        )

    cache_key = (str(pdf_path.resolve()), page_number, scale, clip_key)
    cached = _PAGE_IMAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_path = _render_cache_path(pdf_path, page_number, scale, clip_key)
    if cache_path.exists():
        grayscale = np.load(cache_path, allow_pickle=False)
        _PAGE_IMAGE_CACHE[cache_key] = grayscale
        return grayscale

    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf") from exc

    document = _get_pdf_document(pdf_path)

    page = document[page_number - 1]
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csGRAY,
        alpha=False,
        clip=None if clip_rect is None else fitz.Rect(*clip_rect),
    )

    image = np.frombuffer(pixmap.samples, dtype=np.uint8)
    grayscale = image.reshape(pixmap.height, pixmap.width)
    _PAGE_IMAGE_CACHE[cache_key] = grayscale
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, grayscale, allow_pickle=False)
    return grayscale


def _render_cache_path(
    pdf_path: Path,
    page_number: int,
    scale: float,
    clip_key: tuple[int, int, int, int] | None,
) -> Path:
    stat_result = pdf_path.stat()
    cache_token = "|".join(
        [
            str(pdf_path.resolve()),
            str(stat_result.st_size),
            str(stat_result.st_mtime_ns),
            str(page_number),
            f"{scale:.3f}",
            "full" if clip_key is None else ",".join(str(value) for value in clip_key),
        ]
    )
    digest = hashlib.sha1(cache_token.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _RENDER_CACHE_DIR / f"{digest}.npy"


def _get_pdf_document(pdf_path: Path) -> Any:
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf") from exc

    document_key = str(pdf_path.resolve())
    document = _PDF_DOCUMENT_CACHE.get(document_key)
    if document is None:
        document = fitz.open(document_key)
        _PDF_DOCUMENT_CACHE[document_key] = document
    return document


def _clip_rect_from_bounds(
    pdf_path: Path,
    page_number: int,
    bounds: tuple[int, int, int, int],
    scale: float,
) -> tuple[float, float, float, float]:
    document = _get_pdf_document(pdf_path)
    page = document[page_number - 1]
    page_rect = page.rect

    left, top, right, bottom = bounds
    plot_width = right - left
    plot_height = bottom - top
    margin_x = max(24, plot_width // 8)
    margin_y = max(24, plot_height // 8)

    clip_left = max((left - margin_x) / scale, float(page_rect.x0))
    clip_top = max((top - margin_y) / scale, float(page_rect.y0))
    clip_right = min((right + margin_x) / scale, float(page_rect.x1))
    clip_bottom = min((bottom + margin_y) / scale, float(page_rect.y1))
    return clip_left, clip_top, clip_right, clip_bottom


def _draw_preview_axes(
    draw: ImageDraw.ImageDraw,
    calibration: AxisCalibration,
) -> None:
    font = _load_preview_font(24)
    x_ticks = list(calibration.x_ticks)
    y_ticks = list(calibration.y_ticks)
    left = int(round(x_ticks[0][1]))
    right = int(round(x_ticks[-1][1]))
    bottom = int(round(y_ticks[0][1]))
    top = int(round(y_ticks[-1][1]))

    draw.line([(left, top), (left, bottom)], fill=PREVIEW_AXIS_COLOR, width=6)
    draw.line([(left, bottom), (right, bottom)], fill=PREVIEW_AXIS_COLOR, width=6)

    for tick_value, tick_x_float in x_ticks:
        tick_x = int(round(tick_x_float))
        draw.line(
            [(tick_x, bottom - 14), (tick_x, bottom + 14)],
            fill=PREVIEW_AXIS_COLOR,
            width=4,
        )
        tick_label = str(tick_value)
        label_box = draw.textbbox((0, 0), tick_label, font=font)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (tick_x - (label_width / 2), bottom + 20),
            tick_label,
            fill=PREVIEW_AXIS_COLOR,
            font=font,
        )

    for tick_value, tick_y_float in y_ticks:
        tick_y = int(round(tick_y_float))
        draw.line(
            [(left - 14, tick_y), (left + 14, tick_y)],
            fill=PREVIEW_AXIS_COLOR,
            width=4,
        )
        tick_label = str(tick_value)
        label_box = draw.textbbox((0, 0), tick_label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        draw.text(
            (left - label_width - 26, tick_y - (label_height / 2)),
            tick_label,
            fill=PREVIEW_AXIS_COLOR,
            font=font,
        )


def _x_tick_values(x_max: float) -> list[int]:
    max_value = int(round(x_max))
    tick_values = list(range(0, max_value + 1, 100))
    if tick_values[-1] != max_value:
        tick_values.append(max_value)
    return tick_values


def _preview_crop_box(
    image_size: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    page_width, page_height = image_size
    left, top, right, bottom = bounds
    plot_width = right - left
    plot_height = bottom - top

    margin_left = min(max(plot_width // 7, 120), 380)
    margin_top = min(max(plot_height // 8, 80), 240)
    margin_right = min(max(plot_width // 8, 120), 380)
    margin_bottom = min(max(plot_height // 4, 150), 360)

    return (
        max(left - margin_left, 0),
        max(top - margin_top, 0),
        min(right + margin_right, page_width),
        min(bottom + margin_bottom, page_height),
    )


def _draw_preview_legend(
    image: Image.Image,
    material_ids: tuple[str, ...],
    spec: FigureDigitizerSpec,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_preview_font(26)
    body_font = _load_preview_font(22)
    title = f"Figure {spec.figure_num} overlay | {int(spec.temperature_C)}C"

    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_height = title_box[3] - title_box[1]
    body_boxes = [draw.textbbox((0, 0), _material_display_name(material_id), font=body_font) for material_id in material_ids]
    body_width = max(box[2] - box[0] for box in body_boxes)
    body_height = max(box[3] - box[1] for box in body_boxes)
    legend_width = max(title_box[2] - title_box[0], body_width + 54) + 28
    line_height = body_height + 10
    legend_height = title_height + (line_height * len(material_ids)) + 28
    left = image.width - legend_width - 20
    top = 20

    draw.rounded_rectangle(
        (left, top, left + legend_width, top + legend_height),
        radius=16,
        fill=PREVIEW_LEGEND_BACKGROUND,
        outline=PREVIEW_AXIS_COLOR,
        width=3,
    )
    draw.text((left + 14, top + 10), title, fill=(0, 0, 0, 255), font=title_font)

    for index, material_id in enumerate(material_ids):
        entry_y = top + title_height + 20 + (index * line_height)
        color = PREVIEW_CURVE_COLORS[index % len(PREVIEW_CURVE_COLORS)]
        draw.rectangle((left + 14, entry_y + 4, left + 34, entry_y + 24), fill=color)
        draw.text(
            (left + 44, entry_y),
            _material_display_name(material_id),
            fill=(0, 0, 0, 255),
            font=body_font,
        )


def _sample_track_points(points: list[tuple[int, int]], sample_count: int = 12) -> list[tuple[int, int]]:
    if len(points) <= sample_count:
        return points

    step = max(1, len(points) // sample_count)
    return points[::step]


def _material_display_name(material_id: str) -> str:
    return MATERIAL_DISPLAY_NAMES.get(material_id, material_id.replace("-", " ").title())


def _load_preview_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue

    return ImageFont.load_default()


def _detect_plot_bounds(dark_pixels: np.ndarray[Any, np.dtype[np.bool_]]) -> tuple[int, int, int, int]:
    height, width = dark_pixels.shape
    search_limit = max(width // 3, 1)

    best_column = 0
    best_vertical_run = (-1, -1)
    for column_index in range(search_limit):
        run = _longest_dark_run(dark_pixels[:, column_index])
        if run is None:
            continue
        if (run[1] - run[0]) > (best_vertical_run[1] - best_vertical_run[0]):
            best_column = column_index
            best_vertical_run = run

    if best_vertical_run == (-1, -1):
        raise ValueError("could not detect y-axis")

    horizontal_search_start = best_vertical_run[1] - max(height // 40, 8)
    horizontal_search_end = min(best_vertical_run[1] + max(height // 40, 8), height)
    best_row = best_vertical_run[1]
    best_horizontal_run = (-1, -1)

    for row_index in range(horizontal_search_start, horizontal_search_end):
        run = _longest_dark_run(dark_pixels[row_index, best_column:])
        if run is None:
            continue
        shifted_run = (run[0] + best_column, run[1] + best_column)
        if (shifted_run[1] - shifted_run[0]) > (best_horizontal_run[1] - best_horizontal_run[0]):
            best_row = row_index
            best_horizontal_run = shifted_run

    if best_horizontal_run == (-1, -1):
        raise ValueError("could not detect x-axis")

    return best_column, best_vertical_run[0], best_horizontal_run[1], best_row


def _extract_plot_mask(
    dark_pixels: np.ndarray[Any, np.dtype[np.bool_]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    left, top, right, bottom = bounds
    plot_mask = dark_pixels[top : bottom + 1, left : right + 1].copy()
    plot_mask[:, :6] = False
    plot_mask[-6:, :] = False

    labels_array, component_count_int = _label_components(plot_mask)
    if component_count_int == 0:
        raise ValueError("no digitisation components found inside plot")

    component_sizes = np.bincount(labels_array.ravel())
    minimum_size = max(16, (plot_mask.shape[0] * plot_mask.shape[1]) // 8000)
    keep = component_sizes >= minimum_size
    keep[0] = False

    # Bridge short horizontal gaps so broken dashed traces remain trackable.
    filtered = keep[labels_array]
    return ndimage.binary_closing(filtered, structure=np.ones((1, 9), dtype=bool)).astype(bool)


def _trace_monotonic_curves(
    plot_mask: np.ndarray[Any, np.dtype[np.bool_]],
    expected_count: int,
    reference_fraction: float,
) -> list[list[tuple[int, float]]]:
    reference_x = int(plot_mask.shape[1] * reference_fraction)
    if plot_mask.shape[1] <= 256:
        try:
            tracks = _trace_curves_separately(plot_mask, expected_count, reference_fraction)
        except ValueError:
            tracks = []

        if _tracks_are_distinct_enough(
            tracks,
            expected_count=expected_count,
            reference_x=reference_x,
            tolerance=max(5.0, plot_mask.shape[0] * 0.012),
        ):
            return tracks

    return _trace_monotonic_curves_legacy(plot_mask, expected_count, reference_fraction)


def _trace_curves_separately(
    plot_mask: np.ndarray[Any, np.dtype[np.bool_]],
    expected_count: int,
    reference_fraction: float,
) -> list[list[tuple[int, float]]]:
    width = plot_mask.shape[1]
    column_centers_by_x = [_column_centers(plot_mask[:, column_index]) for column_index in range(width)]
    reference_x = int(width * reference_fraction)
    minimum_span = max(12, width // 6)
    viable_tracks: list[list[tuple[int, float]]] = []
    for seed_x, seed_y in _seed_curve_candidates(
        column_centers_by_x,
        plot_mask.shape[0],
        reference_x,
        expected_count,
    ):
        track = _trace_curve_from_seed(column_centers_by_x, plot_mask.shape[0], seed_x, seed_y)
        if not _is_viable_curve_track(track, minimum_span=minimum_span):
            continue
        if any(
            _tracks_are_similar(track, existing, reference_x=reference_x, tolerance=max(6.0, plot_mask.shape[0] * 0.01))
            for existing in viable_tracks
        ):
            continue
        viable_tracks.append(track)

    if not viable_tracks:
        raise ValueError("no viable curve tracks found")

    candidates = [
        track
        for track in viable_tracks
        if track[0][0] <= reference_x <= track[-1][0]
    ]
    if len(candidates) < expected_count:
        candidates = viable_tracks

    selected = _select_distinct_tracks(
        candidates,
        expected_count=expected_count,
        reference_x=reference_x,
        y_tolerance=max(5.0, plot_mask.shape[0] * 0.012),
    )
    if len(selected) != expected_count:
        raise ValueError(f"expected {expected_count} curve tracks, found {len(selected)}")

    ordered = sorted(selected, key=lambda track: _track_y_at(track, reference_x))
    return [_smooth_track(track) for track in ordered]


def _trace_monotonic_curves_legacy(
    plot_mask: np.ndarray[Any, np.dtype[np.bool_]],
    expected_count: int,
    reference_fraction: float,
) -> list[list[tuple[int, float]]]:
    tracks: list[dict[str, Any]] = []
    width = plot_mask.shape[1]
    column_centers_by_x = [_column_centers(plot_mask[:, column_index]) for column_index in range(width)]
    spawn_limit = int(width * 0.60)
    max_x_gap = max(6, width // 35)
    max_y_jump = max(20.0, plot_mask.shape[0] * 0.10)

    for column_index in range(width):
        centers = column_centers_by_x[column_index]
        used_indices: set[int] = set()

        for track in tracks:
            track["matched"] = False

        for track in sorted(tracks, key=lambda candidate: candidate["points"][-1][1]):
            last_x, last_y = track["points"][-1]
            if column_index - last_x > max_x_gap:
                continue

            candidates = [
                (idx, center)
                for idx, center in enumerate(centers)
                if idx not in used_indices and abs(center - last_y) <= max_y_jump
            ]
            if not candidates:
                continue

            group_index, center = min(candidates, key=lambda item: abs(item[1] - last_y))
            track["points"].append((column_index, center))
            track["missed"] = 0
            track["matched"] = True
            used_indices.add(group_index)

        for track in tracks:
            if not track["matched"]:
                track["missed"] += 1

        if column_index <= spawn_limit:
            for group_index, center in enumerate(centers):
                if group_index in used_indices:
                    continue
                if len(tracks) >= expected_count * 4:
                    break
                tracks.append(
                    {
                        "points": [(column_index, center)],
                        "missed": 0,
                        "matched": True,
                    }
                )

    minimum_points = max(8, width // 10)
    minimum_span = max(12, width // 6)
    viable_tracks = [
        track["points"]
        for track in tracks
        if len(track["points"]) >= minimum_points
        and (track["points"][-1][0] - track["points"][0][0]) >= minimum_span
    ]
    if not viable_tracks:
        raise ValueError("no viable curve tracks found")

    if len(viable_tracks) > expected_count:
        viable_tracks = _merge_track_fragments(
            viable_tracks,
            max_gap=max(40, width // 18),
            max_y_deviation=max(80.0, plot_mask.shape[0] * 0.08),
        )

    reference_x = int(width * reference_fraction)
    candidates = [
        track
        for track in viable_tracks
        if track[0][0] <= reference_x <= track[-1][0]
    ]
    if len(candidates) < expected_count:
        candidates = viable_tracks

    selected = sorted(
        candidates,
        key=lambda track: (track[-1][0] - track[0][0], len(track)),
        reverse=True,
    )[:expected_count]
    if len(selected) != expected_count:
        raise ValueError(f"expected {expected_count} curve tracks, found {len(selected)}")

    ordered = sorted(selected, key=lambda track: _track_y_at(track, reference_x))
    return [_smooth_track(track) for track in ordered]


def _tracks_are_distinct_enough(
    tracks: list[list[tuple[int, float]]],
    expected_count: int,
    reference_x: int,
    tolerance: float,
) -> bool:
    if len(tracks) != expected_count:
        return False

    ordered = sorted(tracks, key=lambda track: _track_y_at(track, reference_x))
    for left_track, right_track in zip(ordered, ordered[1:], strict=False):
        if abs(_track_y_at(left_track, reference_x) - _track_y_at(right_track, reference_x)) <= tolerance:
            return False

    return True


def _seed_curve_candidates(
    column_centers_by_x: list[list[float]],
    plot_height: int,
    reference_x: int,
    expected_count: int,
) -> list[tuple[int, float]]:
    width = len(column_centers_by_x)
    search_radius = max(12, width // 20)
    candidates: list[tuple[int, float]] = []

    for offset in range(search_radius + 1):
        column_indices = [reference_x] if offset == 0 else [reference_x - offset, reference_x + offset]
        for column_index in column_indices:
            if not 0 <= column_index < width:
                continue

            centers = column_centers_by_x[column_index]
            if len(centers) >= expected_count:
                candidates.extend((column_index, center) for center in sorted(centers)[:expected_count])
                break
        if candidates:
            break

    samples: list[tuple[int, float]] = []

    for offset in range(search_radius + 1):
        column_indices = [reference_x] if offset == 0 else [reference_x - offset, reference_x + offset]
        for column_index in column_indices:
            if not 0 <= column_index < width:
                continue
            for center in column_centers_by_x[column_index]:
                samples.append((column_index, center))

    if not samples:
        raise ValueError("no curve seed candidates found")

    y_tolerance = max(5.0, plot_height * 0.015)
    clusters: list[list[tuple[int, float]]] = []
    for sample in sorted(samples, key=lambda item: item[1]):
        if not clusters:
            clusters.append([sample])
            continue

        cluster_center = float(np.mean([point[1] for point in clusters[-1]]))
        if abs(sample[1] - cluster_center) <= y_tolerance:
            clusters[-1].append(sample)
            continue

        clusters.append([sample])

    selected_clusters = sorted(
        clusters,
        key=lambda cluster: (
            len(cluster),
            -min(abs(point[0] - reference_x) for point in cluster),
        ),
        reverse=True,
    )[: max(expected_count * 2, expected_count + 2)]

    if len(selected_clusters) < expected_count:
        for offset in range(search_radius + 1):
            column_indices = [reference_x] if offset == 0 else [reference_x - offset, reference_x + offset]
            for column_index in column_indices:
                if not 0 <= column_index < width:
                    continue
                centers = column_centers_by_x[column_index]
                if len(centers) >= expected_count:
                    return [(column_index, center) for center in centers]

    for cluster in selected_clusters:
        representative_x, _ = min(cluster, key=lambda point: abs(point[0] - reference_x))
        representative_y = float(np.mean([point[1] for point in cluster]))
        if any(abs(existing_y - representative_y) <= y_tolerance for _, existing_y in candidates):
            continue
        candidates.append((int(representative_x), representative_y))

    return sorted(candidates, key=lambda item: item[1])[: max(expected_count * 4, expected_count + 4)]


def _tracks_are_similar(
    left_track: list[tuple[int, float]],
    right_track: list[tuple[int, float]],
    reference_x: int,
    tolerance: float,
) -> bool:
    overlap_left = max(left_track[0][0], right_track[0][0])
    overlap_right = min(left_track[-1][0], right_track[-1][0])
    if overlap_left >= overlap_right:
        return False

    probe_x = min(max(reference_x, overlap_left), overlap_right)
    return (
        abs(_track_y_at(left_track, probe_x) - _track_y_at(right_track, probe_x)) <= tolerance
        and abs(left_track[0][0] - right_track[0][0]) <= 24
        and abs(left_track[-1][0] - right_track[-1][0]) <= 24
    )


def _trace_curve_from_seed(
    column_centers_by_x: list[list[float]],
    plot_height: int,
    seed_x: int,
    seed_y: float,
) -> list[tuple[int, float]]:
    prefer_dashed_mode = _seed_has_dashed_pattern(column_centers_by_x, seed_x, seed_y)
    left_track = _trace_curve_direction(
        column_centers_by_x,
        plot_height,
        seed_x,
        seed_y,
        direction=-1,
        prefer_dashed_mode=prefer_dashed_mode,
    )
    right_track = _trace_curve_direction(
        column_centers_by_x,
        plot_height,
        seed_x,
        seed_y,
        direction=1,
        prefer_dashed_mode=prefer_dashed_mode,
    )
    return _combine_tracks(list(reversed(left_track)), right_track)


def _trace_curve_direction(
    column_centers_by_x: list[list[float]],
    plot_height: int,
    seed_x: int,
    seed_y: float,
    direction: int,
    prefer_dashed_mode: bool,
) -> list[tuple[int, float]]:
    width = len(column_centers_by_x)
    max_gap = max(12, min(72, width // 20))
    max_y_deviation = max(18.0, plot_height * 0.08)
    track = [(seed_x, seed_y)]
    gap_history: list[int] = []
    segment_history: list[int] = []
    current_segment_length = 1

    while True:
        next_point = _select_direction_candidate(
            column_centers_by_x,
            track=track,
            direction=direction,
            max_gap=max_gap,
            max_y_deviation=max_y_deviation,
            gap_history=gap_history,
            segment_history=segment_history,
            current_segment_length=current_segment_length,
            prefer_dashed_mode=prefer_dashed_mode,
        )
        if next_point is None:
            break

        gap_length = abs(next_point[0] - track[-1][0]) - 1
        if gap_length > 0:
            gap_history.append(gap_length)
            segment_history.append(current_segment_length)
            current_segment_length = 1
        else:
            current_segment_length += 1

        track.append(next_point)

    return track


def _select_direction_candidate(
    column_centers_by_x: list[list[float]],
    track: list[tuple[int, float]],
    direction: int,
    max_gap: int,
    max_y_deviation: float,
    gap_history: list[int],
    segment_history: list[int],
    current_segment_length: int,
    prefer_dashed_mode: bool,
) -> tuple[int, float] | None:
    width = len(column_centers_by_x)
    current_x, current_y = track[-1]
    expected_gradient = _estimate_track_gradient(track)
    best_overall_candidate: tuple[int, float] | None = None
    best_overall_score: float | None = None

    for step in range(1, max_gap + 2):
        candidate_x = current_x + (direction * step)
        if not 0 <= candidate_x < width:
            break

        candidate_centers = column_centers_by_x[candidate_x]
        if not candidate_centers:
            continue

        best_step_candidate: tuple[int, float] | None = None
        best_step_score: float | None = None

        for center in candidate_centers:
            delta_x = float(candidate_x - current_x)
            predicted_y = current_y + (expected_gradient * delta_x)
            projection_error = abs(center - predicted_y)
            if projection_error > max_y_deviation:
                continue

            candidate_gradient = (center - current_y) / delta_x
            score = projection_error + (18.0 * abs(candidate_gradient - expected_gradient))
            score += _direction_turn_penalty(
                expected_gradient,
                candidate_gradient,
                point_count=len(track),
            )
            score += _dash_pattern_penalty(
                gap_length=step - 1,
                gap_history=gap_history,
                segment_history=segment_history,
                current_segment_length=current_segment_length,
            )

            if best_step_score is None or score < best_step_score:
                best_step_candidate = (candidate_x, float(center))
                best_step_score = score

        if best_step_candidate is not None:
            if not prefer_dashed_mode and not gap_history:
                return best_step_candidate

            if best_step_score is not None and (
                best_overall_score is None or best_step_score < best_overall_score
            ):
                best_overall_candidate = best_step_candidate
                best_overall_score = best_step_score

    return best_overall_candidate


def _seed_has_dashed_pattern(
    column_centers_by_x: list[list[float]],
    seed_x: int,
    seed_y: float,
) -> bool:
    width = len(column_centers_by_x)
    search_radius = max(18, min(96, width // 20))
    tolerance = 8.0
    hits: list[bool] = []

    for column_index in range(max(0, seed_x - search_radius), min(width, seed_x + search_radius + 1)):
        hits.append(any(abs(center - seed_y) <= tolerance for center in column_centers_by_x[column_index]))

    if not hits or all(hits):
        return False

    true_runs: list[int] = []
    false_runs: list[int] = []
    current_value = hits[0]
    current_length = 1
    for hit in hits[1:]:
        if hit == current_value:
            current_length += 1
            continue

        if current_value:
            true_runs.append(current_length)
        else:
            false_runs.append(current_length)
        current_value = hit
        current_length = 1

    if current_value:
        true_runs.append(current_length)
    else:
        false_runs.append(current_length)

    if len(true_runs) < 2 or not false_runs:
        return False

    coverage = sum(true_runs) / float(len(hits))
    return coverage < 0.8 and max(false_runs) >= 3


def _estimate_track_gradient(track: list[tuple[int, float]]) -> float:
    if len(track) < 2:
        return 0.0

    sample = track[-min(len(track), 6) :]
    x_values = np.array([point[0] for point in sample], dtype=float)
    y_values = np.array([point[1] for point in sample], dtype=float)
    if np.allclose(x_values, x_values[0]):
        return 0.0

    gradients = np.diff(y_values) / np.diff(x_values)
    return float(np.median(gradients))


def _direction_turn_penalty(
    expected_gradient: float,
    candidate_gradient: float,
    point_count: int,
) -> float:
    if point_count < 3:
        return 0.0

    if abs(expected_gradient) < 0.05 or abs(candidate_gradient) < 0.05:
        return 0.0

    if np.sign(expected_gradient) != np.sign(candidate_gradient):
        return 30.0 + (15.0 * abs(candidate_gradient - expected_gradient))

    return 0.0


def _dash_pattern_penalty(
    gap_length: int,
    gap_history: list[int],
    segment_history: list[int],
    current_segment_length: int,
) -> float:
    if gap_length == 0:
        return 0.0

    penalty = 2.0 * gap_length
    if gap_history:
        target_gap = float(np.median(np.array(gap_history[-3:], dtype=float)))
        gap_error = abs(gap_length - target_gap) / max(1.0, target_gap)
        penalty += max(0.0, (gap_error - 0.20) * 18.0)
        if gap_error <= 0.20:
            penalty -= 1.0

    if gap_history and segment_history:
        recent_segments = np.array(segment_history[-min(len(segment_history), 3) :], dtype=float)
        recent_gaps = np.array(gap_history[-min(len(gap_history), 3) :], dtype=float)
        target_ratio = float(np.median(recent_segments / np.maximum(recent_gaps, 1.0)))
        current_ratio = float(current_segment_length) / float(max(gap_length, 1))
        ratio_error = abs(current_ratio - target_ratio) / max(abs(target_ratio), 0.25)
        penalty += max(0.0, (ratio_error - 0.20) * 12.0)
        if ratio_error <= 0.20:
            penalty -= 1.0

    return max(0.0, penalty)


def _is_viable_curve_track(
    track: list[tuple[int, float]],
    minimum_span: int,
) -> bool:
    if len(track) < max(8, minimum_span // 2):
        return False

    span = track[-1][0] - track[0][0]
    if span < minimum_span:
        return False

    density = len(track) / max(1, span + 1)
    return density >= 0.35


def _label_components(
    plot_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> tuple[np.ndarray[Any, np.dtype[np.int32]], int]:
    labels_raw, component_count = cast(
        tuple[np.ndarray[Any, Any], int],
        ndimage.label(plot_mask),
    )
    return np.asarray(labels_raw, dtype=np.int32), int(component_count)


def _select_distinct_tracks(
    tracks: list[list[tuple[int, float]]],
    expected_count: int,
    reference_x: int,
    y_tolerance: float,
) -> list[list[tuple[int, float]]]:
    ordered_tracks = sorted(
        tracks,
        key=lambda track: (track[-1][0] - track[0][0], len(track)),
        reverse=True,
    )

    selected: list[list[tuple[int, float]]] = []
    for track in ordered_tracks:
        track_y = _track_y_at(track, reference_x)
        if any(abs(track_y - _track_y_at(existing, reference_x)) <= y_tolerance for existing in selected):
            continue

        selected.append(track)
        if len(selected) == expected_count:
            return selected

    return ordered_tracks[:expected_count]


def _track_to_points(
    track: list[tuple[int, float]],
    bounds: tuple[int, int, int, int],
    calibration: AxisCalibration,
) -> list[RelaxationPoint]:
    left, top, _, _ = bounds

    unique_points: dict[int, float] = {}
    for x_coord, y_coord in track:
        unique_points[int(x_coord)] = float(y_coord)

    x_values = np.array(sorted(unique_points), dtype=float)
    y_values = np.array([unique_points[int(x_coord)] for x_coord in x_values], dtype=float)
    sample_count = min(32, len(x_values))
    sample_x = np.linspace(x_values.min(), x_values.max(), sample_count)
    sample_y = np.interp(sample_x, x_values, y_values)

    page_x = left + sample_x
    page_y = top + sample_y
    stresses = [calibration.x_to_stress(float(x_coord)) for x_coord in page_x]
    relaxations = [calibration.y_to_relaxation(float(y_coord)) for y_coord in page_y]

    points: list[RelaxationPoint] = []
    for stress, relaxation in zip(stresses, relaxations, strict=True):
        clamped_relaxation = min(max(float(relaxation), 0.0), Y_AXIS_MAX)
        points.append(
            RelaxationPoint(
                stress_MPa=round(float(stress), 1),
                relaxation_percent=round(clamped_relaxation, 3),
            )
        )

    return points


def _column_centers(column: np.ndarray[Any, np.dtype[np.bool_]]) -> list[float]:
    indices = np.flatnonzero(column)
    if len(indices) == 0:
        return []

    split_points = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, split_points)
    return [float(group.mean()) for group in groups if len(group) >= 2]


def _track_y_at(track: list[tuple[int, float]], reference_x: int) -> float:
    x_values = np.array([point[0] for point in track], dtype=float)
    y_values = np.array([point[1] for point in track], dtype=float)
    return float(np.interp(reference_x, x_values, y_values))


def _reorder_tracks_for_spec(
    spec: FigureDigitizerSpec,
    tracks: list[list[tuple[int, float]]],
) -> list[list[tuple[int, float]]]:
    if spec.figure_num != 26 or len(tracks) != 5:
        return tracks

    upper_tracks = tracks[:3]
    lower_tracks = tracks[3:]
    tungsten_track = max(upper_tracks, key=lambda track: (_track_span(track), _track_density(track)))
    dashed_tracks = [track for track in upper_tracks if track is not tungsten_track]
    a286_track = min(dashed_tracks, key=lambda track: (_track_span(track), -_track_density(track)))
    inconel_600_track = next(track for track in dashed_tracks if track is not a286_track)
    return [a286_track, inconel_600_track, tungsten_track, *lower_tracks]


def _track_span(track: list[tuple[int, float]]) -> int:
    return int(track[-1][0] - track[0][0])


def _track_density(track: list[tuple[int, float]]) -> float:
    span = max(1, _track_span(track) + 1)
    return len(track) / float(span)


def _merge_track_fragments(
    tracks: list[list[tuple[int, float]]],
    max_gap: int,
    max_y_deviation: float,
) -> list[list[tuple[int, float]]]:
    merged_tracks = [list(track) for track in tracks]

    while True:
        best_pair: tuple[int, int] | None = None
        best_score: float | None = None

        for left_index, left_track in enumerate(merged_tracks):
            for right_index, right_track in enumerate(merged_tracks):
                if left_index == right_index:
                    continue

                score = _track_merge_score(
                    left_track,
                    right_track,
                    max_gap=max_gap,
                    max_y_deviation=max_y_deviation,
                )
                if score is None:
                    continue

                if best_score is None or score < best_score:
                    best_pair = (left_index, right_index)
                    best_score = score

        if best_pair is None:
            break

        left_index, right_index = best_pair
        merged_track = _combine_tracks(merged_tracks[left_index], merged_tracks[right_index])
        merged_tracks[left_index] = merged_track
        del merged_tracks[right_index]

    return merged_tracks


def _track_merge_score(
    left_track: list[tuple[int, float]],
    right_track: list[tuple[int, float]],
    max_gap: int,
    max_y_deviation: float,
) -> float | None:
    left_end_x = left_track[-1][0]
    right_start_x = right_track[0][0]
    right_end_x = right_track[-1][0]

    if right_end_x <= left_end_x:
        return None

    gap = right_start_x - left_end_x
    if gap > max_gap or gap < -max_gap:
        return None

    predicted_y = _predict_track_value(left_track, right_start_x)
    deviation = abs(predicted_y - right_track[0][1])
    if deviation > max_y_deviation:
        return None

    extension = right_end_x - left_end_x
    if extension < max(10, max_gap // 2):
        return None

    overlap_penalty = max(0, -gap)
    return deviation + (0.2 * overlap_penalty)


def _combine_tracks(
    left_track: list[tuple[int, float]],
    right_track: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    combined: dict[int, list[float]] = {}
    for x_coord, y_coord in left_track + right_track:
        combined.setdefault(int(x_coord), []).append(float(y_coord))

    return [
        (x_coord, float(sum(y_values) / len(y_values)))
        for x_coord, y_values in sorted(combined.items())
    ]


def _predict_track_value(track: list[tuple[int, float]], target_x: int) -> float:
    if len(track) == 1:
        return float(track[-1][1])

    sample = track[-min(len(track), 12) :]
    x_values = np.array([point[0] for point in sample], dtype=float)
    y_values = np.array([point[1] for point in sample], dtype=float)

    if np.allclose(x_values, x_values[0]):
        return float(y_values[-1])

    slope, intercept = np.polyfit(x_values, y_values, 1)
    return float((slope * float(target_x)) + intercept)


def _smooth_track(track: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if len(track) < 9:
        return track

    x_values = np.array([point[0] for point in track], dtype=float)
    y_values = np.array([point[1] for point in track], dtype=float)
    median_window = min(15, len(track) - (1 - len(track) % 2))
    if median_window >= 3:
        y_values = ndimage.median_filter(y_values, size=median_window, mode="nearest")

    sigma = max(1.2, len(track) / 180.0)
    y_values = ndimage.gaussian_filter1d(y_values, sigma=sigma, mode="nearest")

    if y_values[-1] <= y_values[0]:
        y_values = np.minimum.accumulate(y_values)
    else:
        y_values = np.maximum.accumulate(y_values)

    return [
        (int(round(x_coord)), float(y_coord))
        for x_coord, y_coord in zip(x_values, y_values, strict=True)
    ]


def _longest_dark_run(values: np.ndarray[Any, np.dtype[np.bool_]]) -> tuple[int, int] | None:
    if not values.any():
        return None

    padded = np.concatenate((np.array([False]), values, np.array([False])))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    lengths = ends - starts
    best_index = int(np.argmax(lengths))
    return int(starts[best_index]), int(ends[best_index] - 1)


def _calibrate_axes(
    dark_pixels: np.ndarray[Any, np.dtype[np.bool_]],
    bounds: tuple[int, int, int, int],
    x_max: float,
) -> AxisCalibration:
    left, top, right, bottom = bounds
    x_ticks = _detect_x_tick_positions(dark_pixels, left, right, bottom, x_max)
    y_ticks = _detect_y_tick_positions(dark_pixels, left, top, bottom)

    x_values = np.array([float(value) for value, _ in x_ticks], dtype=float)
    x_positions = np.array([position for _, position in x_ticks], dtype=float)
    x_slope, x_intercept = np.polyfit(x_values, x_positions, 1)

    y_values = np.array([float(value) for value, _ in y_ticks], dtype=float)
    y_positions = np.array([position for _, position in y_ticks], dtype=float)
    y_slope, y_intercept = np.polyfit(y_values, y_positions, 1)

    return AxisCalibration(
        x_ticks=tuple(x_ticks),
        y_ticks=tuple(y_ticks),
        x_slope=float(x_slope),
        x_intercept=float(x_intercept),
        y_slope=float(y_slope),
        y_intercept=float(y_intercept),
    )


def _detect_x_tick_positions(
    dark_pixels: np.ndarray[Any, np.dtype[np.bool_]],
    left: int,
    right: int,
    bottom: int,
    x_max: float,
) -> list[tuple[int, float]]:
    scan_start = left + 20
    scan_end = right - 4
    run_lengths = [
        _vertical_run_up_from_row(dark_pixels[:, column_index], bottom)
        for column_index in range(scan_start, scan_end + 1)
    ]
    tick_values = _x_tick_values(x_max)
    detected_positions = _cluster_tick_peaks(run_lengths, scan_start, min_length=14)
    expected_tick_count = len(tick_values) - 1
    if len(detected_positions) < expected_tick_count:
        step = (right - left) / max(expected_tick_count, 1)
        detected_positions = [left + (step * index) for index in range(1, len(tick_values))]
    else:
        detected_positions = detected_positions[:expected_tick_count]

    return [(0, float(left))] + [
        (tick_value, float(position))
        for tick_value, position in zip(tick_values[1:], detected_positions, strict=True)
    ]


def _detect_y_tick_positions(
    dark_pixels: np.ndarray[Any, np.dtype[np.bool_]],
    left: int,
    top: int,
    bottom: int,
) -> list[tuple[int, float]]:
    scan_end = max(bottom - 20, top)
    run_lengths = [
        _horizontal_run_right_from_column(dark_pixels[row_index, :], left)
        for row_index in range(top, scan_end + 1)
    ]
    detected_positions = _cluster_tick_peaks(run_lengths, top, min_length=14)
    if len(detected_positions) < 3:
        span = max(bottom - top, 1)
        detected_positions = [
            bottom - (span * (tick_value / Y_AXIS_MAX))
            for tick_value in [15, 10, 5]
        ]
    else:
        detected_positions = detected_positions[:3]

    return [
        (0, float(bottom)),
        (5, float(detected_positions[2])),
        (10, float(detected_positions[1])),
        (15, float(detected_positions[0])),
    ]


def _cluster_tick_peaks(
    run_lengths: list[int],
    offset: int,
    min_length: int,
) -> list[float]:
    groups: list[list[tuple[int, int]]] = []
    current_group: list[tuple[int, int]] = []

    for index, run_length in enumerate(run_lengths):
        if run_length < min_length:
            if current_group:
                groups.append(current_group)
                current_group = []
            continue

        current_group.append((offset + index, run_length))

    if current_group:
        groups.append(current_group)

    positions: list[float] = []
    for group in groups:
        weights = np.array([run_length for _, run_length in group], dtype=float)
        xs = np.array([index for index, _ in group], dtype=float)
        positions.append(float(np.average(xs, weights=weights)))

    return positions


def _horizontal_run_right_from_column(values: np.ndarray[Any, np.dtype[np.bool_]], start: int) -> int:
    run_length = 0
    index = start
    while index < len(values) and values[index]:
        run_length += 1
        index += 1
    return run_length


def _vertical_run_up_from_row(values: np.ndarray[Any, np.dtype[np.bool_]], start: int) -> int:
    run_length = 0
    index = start
    while index >= 0 and values[index]:
        run_length += 1
        index -= 1
    return run_length