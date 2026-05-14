from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .schema import RelaxationCurve, RelaxationPoint

Y_AXIS_MAX = 15.0

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


def _digitize_figure(
    pdf_path: Path,
    spec: FigureDigitizerSpec,
) -> dict[str, RelaxationCurve]:
    grayscale = _render_page_grayscale(pdf_path, spec.page)
    dark = grayscale < 170
    bounds = _detect_plot_bounds(dark)
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
        points = _track_to_points(track, plot_mask.shape, spec.x_max)
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

    return curves


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
    plot_shape: tuple[int, int],
    x_max: float,
) -> list[RelaxationPoint]:
    width = max(plot_shape[1] - 1, 1)
    height = max(plot_shape[0] - 1, 1)

    unique_points: dict[int, float] = {}
    for x_coord, y_coord in track:
        unique_points[int(x_coord)] = float(y_coord)

    x_values = np.array(sorted(unique_points), dtype=float)
    y_values = np.array([unique_points[int(x_coord)] for x_coord in x_values], dtype=float)
    sample_count = min(32, len(x_values))
    sample_x = np.linspace(x_values.min(), x_values.max(), sample_count)
    sample_y = np.interp(sample_x, x_values, y_values)

    stresses = (sample_x / width) * x_max
    relaxations = ((height - sample_y) / height) * Y_AXIS_MAX

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