from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .schema import RelaxationCurve, RelaxationPoint

Y_AXIS_MAX = 15.0
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
    grayscale = _render_page_grayscale(pdf_path, spec.page)
    dark = grayscale < 170
    bounds = _detect_plot_bounds(dark)
    calibration = _calibrate_axes(dark, bounds, spec.x_max)
    plot_mask = _extract_plot_mask(dark, bounds)
    tracks = _trace_monotonic_curves(
        plot_mask,
        expected_count=len(spec.material_ids_top_to_bottom),
        reference_fraction=spec.reference_fraction,
    )

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

    return FigureAnalysisResult(
        spec=spec,
        page_image=grayscale,
        plot_bounds=bounds,
        calibration=calibration,
        tracks=tracks,
        curves=curves,
    )


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


def _render_page_grayscale(pdf_path: Path, page_number: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf") from exc

    with fitz.open(str(pdf_path)) as document:
        page = document[page_number - 1]
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(4, 4),
            colorspace=fitz.csGRAY,
            alpha=False,
        )

    image = np.frombuffer(pixmap.samples, dtype=np.uint8)
    return image.reshape(pixmap.height, pixmap.width)


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

    labels, component_count = ndimage.label(plot_mask)
    if component_count == 0:
        raise ValueError("no digitisation components found inside plot")

    component_sizes = np.bincount(labels.ravel())
    minimum_size = max(16, (plot_mask.shape[0] * plot_mask.shape[1]) // 8000)
    keep = component_sizes >= minimum_size
    keep[0] = False

    # Bridge short horizontal gaps so broken dashed traces remain trackable.
    filtered = keep[labels]
    return ndimage.binary_closing(filtered, structure=np.ones((1, 9), dtype=bool))


def _trace_monotonic_curves(
    plot_mask: np.ndarray[Any, np.dtype[np.bool_]],
    expected_count: int,
    reference_fraction: float,
) -> list[list[tuple[int, float]]]:
    tracks: list[dict[str, Any]] = []
    width = plot_mask.shape[1]
    spawn_limit = int(width * 0.60)
    max_x_gap = max(6, width // 35)
    max_y_jump = max(20.0, plot_mask.shape[0] * 0.10)

    for column_index in range(width):
        centers = _column_centers(plot_mask[:, column_index])
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

    minimum_points = max(40, width // 10)
    minimum_span = max(80, width // 6)
    viable_tracks = [
        track["points"]
        for track in tracks
        if len(track["points"]) >= minimum_points
        and (track["points"][-1][0] - track["points"][0][0]) >= minimum_span
    ]
    if not viable_tracks:
        raise ValueError("no viable curve tracks found")

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

    return sorted(selected, key=lambda track: _track_y_at(track, reference_x))


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