"""
Relaxation curve extractor for IST Report 330.

NOTE: The IST Report 330 contains figures that are embedded as raster images.
Automated pixel-level digitisation of these curves would require calibrated
image-processing (axis detection, curve tracing) which is highly PDF-layout
specific and cannot be performed reliably without human validation.

This module performs the following best-effort extraction:
  - Identifies material names and families from surrounding text.
  - Extracts tabular data (Tables I–III) where pdfplumber can parse them.
  - Creates structured placeholder entries for figures that cannot be
    automatically digitised, with clear warnings.

No values are fabricated.  Placeholder curves carry extraction_confidence=0.0
and extraction_method="placeholder" so downstream code can filter them.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import IST_REPORT_URL, PACKAGE_VERSION
from .figure_digitizer import FIGURE_PAGE_HINTS, digitize_supported_figure_curves
from .pdf_parser import (
    detect_figure_references,
    detect_table_references,
    extract_tables_with_pdfplumber,
    extract_text_by_page,
)
from .schema import (
    LicenseNotice,
    Material,
    MetadataOutput,
    RelaxationCurve,
    RelaxationDataset,
    RelaxationPoint,
    SourceInfo,
)

# ---------------------------------------------------------------------------
# Known materials from IST Report 330 (Tables I–III / section headings).
# We use names/aliases that appear in the report text so they can be matched.
# ---------------------------------------------------------------------------
KNOWN_MATERIALS: list[dict[str, Any]] = [
    {
        "material_id": "carbon-steel",
        "material_name": "Carbon Steel",
        "aliases": ["hard drawn carbon steel", "high carbon steel", "patented carbon steel"],
        "family": "steel",
        "figure_range": list(range(1, 6)),
        "temperatures_C": [20, 100, 150, 200],
    },
    {
        "material_id": "alloy-steel",
        "material_name": "Alloy Steel",
        "aliases": ["chrome vanadium", "chrome silicon", "Cr-V", "Cr-Si"],
        "family": "steel",
        "figure_range": list(range(6, 10)),
        "temperatures_C": [20, 100, 150, 200],
    },
    {
        "material_id": "patented-carbon-steel",
        "material_name": "Patented Carbon Steel",
        "aliases": ["patented material", "patented"],
        "family": "steel",
        "figure_temperatures": [
            {"figure_num": 20, "temperature_C": 100.0},
            {"figure_num": 21, "temperature_C": 150.0},
            {"figure_num": 22, "temperature_C": 200.0},
        ],
    },
    {
        "material_id": "oil-hardened-and-tempered-steel",
        "material_name": "Oil Hardened and Tempered Steel",
        "aliases": ["oil hardened and tempered", "oil hardened"],
        "family": "steel",
        "figure_temperatures": [
            {"figure_num": 20, "temperature_C": 100.0},
            {"figure_num": 21, "temperature_C": 150.0},
            {"figure_num": 22, "temperature_C": 200.0},
        ],
    },
    {
        "material_id": "silicon-chromium-steel",
        "material_name": "Silicon-Chromium Steel",
        "aliases": ["silicon-chromium", "silicon chromium"],
        "family": "steel",
        "figure_temperatures": [
            {"figure_num": 20, "temperature_C": 100.0},
            {"figure_num": 21, "temperature_C": 150.0},
            {"figure_num": 22, "temperature_C": 200.0},
            {"figure_num": 23, "temperature_C": 250.0},
        ],
    },
    {
        "material_id": "stainless-steel-302-304",
        "material_name": "Stainless Steel 302/304",
        "aliases": ["302", "304", "austenitic stainless"],
        "family": "stainless_steel",
        "figure_range": list(range(10, 13)),
        "temperatures_C": [20, 100, 150, 200, 250],
    },
    {
        "material_id": "stainless-steel-18cr8ni",
        "material_name": "18Cr/8Ni Stainless Steel",
        "aliases": ["18cr/8ni stainless", "18cr 8ni stainless", "18/8 stainless"],
        "family": "stainless_steel",
        "figure_temperatures": [
            {"figure_num": 21, "temperature_C": 150.0},
            {"figure_num": 22, "temperature_C": 200.0},
            {"figure_num": 23, "temperature_C": 250.0},
            {"figure_num": 24, "temperature_C": 300.0},
            {"figure_num": 25, "temperature_C": 350.0},
        ],
    },
    {
        "material_id": "stainless-steel-17-7ph",
        "material_name": "Stainless Steel 17-7PH",
        "aliases": ["17-7 PH", "precipitation hardening stainless"],
        "family": "stainless_steel",
        "figure_range": list(range(13, 15)),
        "temperatures_C": [20, 100, 150, 200, 250, 300],
    },
    {
        "material_id": "inconel-x750",
        "material_name": "Inconel X-750",
        "aliases": ["Inconel X750", "X-750", "nickel alloy"],
        "family": "nickel_alloy",
        "figure_range": list(range(15, 18)),
        "temperatures_C": [20, 150, 200, 300, 400, 500],
        "figure_temperatures": [
            {"figure_num": 26, "temperature_C": 400.0},
            {"figure_num": 27, "temperature_C": 450.0},
            {"figure_num": 28, "temperature_C": 500.0},
            {"figure_num": 29, "temperature_C": 550.0},
        ],
    },
    {
        "material_id": "phosphor-bronze",
        "material_name": "Phosphor Bronze",
        "aliases": ["CuSnP", "bronze"],
        "family": "copper_alloy",
        "figure_range": [18, 19],
        "temperatures_C": [20, 100, 150],
        "figure_temperatures": [
            {"figure_num": 18, "temperature_C": 100.0},
        ],
    },
    {
        "material_id": "beryllium-copper",
        "material_name": "Beryllium Copper",
        "aliases": ["BeCu", "beryllium-copper", "CuBe"],
        "family": "copper_alloy",
        "figure_range": [20, 21],
        "temperatures_C": [20, 100, 150, 200],
        "figure_temperatures": [
            {"figure_num": 18, "temperature_C": 100.0},
            {"figure_num": 19, "temperature_C": 150.0},
        ],
    },
    {
        "material_id": "titanium-alloy",
        "material_name": "Titanium Alloy",
        "aliases": ["Ti alloy", "titanium"],
        "family": "titanium",
        "figure_range": [22, 23],
        "temperatures_C": [20, 100, 150, 200],
        "figure_temperatures": [
            {"figure_num": 18, "temperature_C": 100.0},
            {"figure_num": 19, "temperature_C": 150.0},
        ],
    },
    {
        "material_id": "tungsten-steel",
        "material_name": "Tungsten Steel",
        "aliases": ["tungsten steel"],
        "family": "steel",
        "figure_temperatures": [
            {"figure_num": 23, "temperature_C": 250.0},
            {"figure_num": 24, "temperature_C": 300.0},
            {"figure_num": 25, "temperature_C": 350.0},
            {"figure_num": 26, "temperature_C": 400.0},
        ],
    },
    {
        "material_id": "inconel-600",
        "material_name": "Inconel 600",
        "aliases": ["inconel 600"],
        "family": "nickel_alloy",
        "figure_temperatures": [
            {"figure_num": 22, "temperature_C": 200.0},
            {"figure_num": 23, "temperature_C": 250.0},
            {"figure_num": 24, "temperature_C": 300.0},
            {"figure_num": 25, "temperature_C": 350.0},
            {"figure_num": 26, "temperature_C": 400.0},
        ],
    },
    {
        "material_id": "elgiloy",
        "material_name": "Elgiloy",
        "aliases": ["Cobenium", "Phynox", "MP35N"],
        "family": "cobalt_alloy",
        "figure_range": [24, 25],
        "temperatures_C": [20, 100, 200, 300],
        "figure_temperatures": [
            {"figure_num": 25, "temperature_C": 350.0},
        ],
    },
    {
        "material_id": "18ni-co-mo-maraging-steel",
        "material_name": "18Ni-Co-Mo Maraging Steel",
        "aliases": ["18 ni-co-mo maraging", "maraging", "maraging steel"],
        "family": "steel",
        "figure_temperatures": [
            {"figure_num": 24, "temperature_C": 300.0},
            {"figure_num": 25, "temperature_C": 350.0},
            {"figure_num": 26, "temperature_C": 400.0},
        ],
    },
    {
        "material_id": "a286",
        "material_name": "A286",
        "aliases": ["a 286", "a286"],
        "family": "nickel_alloy",
        "figure_temperatures": [
            {"figure_num": 26, "temperature_C": 400.0},
        ],
    },
    {
        "material_id": "nb-alloy",
        "material_name": "Nimonic / Ni-based Superalloy",
        "aliases": ["Nimonic", "Ni superalloy"],
        "family": "nickel_alloy",
        "figure_range": [26, 27],
        "temperatures_C": [20, 200, 300, 400, 500],
        "figure_temperatures": [],
    },
]

STRICT_NUMBER_PATTERN = re.compile(r"^\s*-?\d+(?:[.,]\d+)?\s*%?\s*$")
TEMPERATURE_PATTERN = re.compile(
    r"(?<!\d)(-?\d+(?:[.,]\d+)?)\s*(?:°\s*C|deg(?:rees)?\.?\s*C|C)\b",
    re.IGNORECASE,
)


def _make_placeholder_curve(
    material_id: str,
    figure_num: int,
    temperature_C: float,
    page_hint: int,
    warnings: list[str],
) -> RelaxationCurve:
    """Create a structured placeholder for a figure that cannot be auto-digitised."""
    warn_msg = (
        f"Figure {figure_num} could not be automatically digitised. "
        "Pixel-level curve tracing was not attempted because it requires "
        "calibrated image processing with human validation. "
        "No values were fabricated."
    )
    warnings.append(warn_msg)
    return RelaxationCurve(
        curve_id=f"{material_id}_fig{figure_num}_T{int(temperature_C)}C",
        source_type="figure",
        source_ref=f"Figure {figure_num}",
        page=page_hint,
        temperature_C=temperature_C,
        confidence_level=None,
        points=[],
        extraction_method="placeholder",
        extraction_confidence=0.0,
        warnings=[warn_msg],
    )


def _iter_figure_temperature_pairs(mat_spec: dict[str, Any]) -> list[tuple[int, float]]:
    if "figure_temperatures" in mat_spec:
        figure_temperatures = mat_spec["figure_temperatures"]
        return [
            (int(item["figure_num"]), float(item["temperature_C"]))
            for item in figure_temperatures
        ]

    return [
        (int(figure_num), float(temperature_C))
        for figure_num in mat_spec["figure_range"]
        for temperature_C in mat_spec["temperatures_C"]
    ]


def _parse_table_row_to_point(row: list) -> RelaxationPoint | None:
    """
    Attempt to parse a table row into a RelaxationPoint.
    Expected columns: stress (MPa) | relaxation (%)
    Returns None if the row cannot be parsed.
    """
    if not row or len(row) < 2:
        return None
    stress = _parse_numeric_cell(row[0])
    relax = _parse_numeric_cell(row[1])
    if stress is not None and relax is not None and stress > 0 and 0.0 <= relax <= 100.0:
        return RelaxationPoint(stress_MPa=stress, relaxation_percent=relax)
    return None


def _parse_numeric_cell(cell: Any) -> float | None:
    if isinstance(cell, int | float):
        return float(cell)
    if cell is None:
        return None

    text = str(cell).strip()
    if not text or STRICT_NUMBER_PATTERN.fullmatch(text) is None:
        return None

    return float(text.replace("%", "").replace(",", ".").strip())


def _parse_temperature_cell(cell: Any) -> float | None:
    if cell is None:
        return None

    match = TEMPERATURE_PATTERN.search(str(cell))
    if match is None:
        return None

    return float(match.group(1).replace(",", "."))


def _format_temperature_token(temperature_C: float) -> str:
    if temperature_C.is_integer():
        return str(int(temperature_C))
    return str(temperature_C).replace(".", "_")


def _extract_temperature_columns(data: list[list[Any]]) -> dict[int, float]:
    temperature_columns: dict[int, float] = {}

    for row in data[:3]:
        if not row:
            continue

        for column_index, cell in enumerate(row):
            if column_index == 0:
                continue

            temperature_C = _parse_temperature_cell(cell)
            if temperature_C is not None:
                temperature_columns[column_index] = temperature_C

    return temperature_columns


def _extract_table_curves(
    material_id: str,
    table: dict[str, Any],
) -> list[RelaxationCurve]:
    data = table.get("data", [])
    if not data:
        return []

    page = table["page"]
    temperature_columns = _extract_temperature_columns(data)
    if temperature_columns:
        curves_by_temperature: dict[float, RelaxationCurve] = {}

        for row in data[1:]:
            if not row:
                continue

            stress = _parse_numeric_cell(row[0])
            if stress is None or stress <= 0:
                continue

            for column_index, temperature_C in temperature_columns.items():
                if column_index >= len(row):
                    continue

                relax = _parse_numeric_cell(row[column_index])
                if relax is None or not 0.0 <= relax <= 100.0:
                    continue

                curve = curves_by_temperature.setdefault(
                    temperature_C,
                    RelaxationCurve(
                        curve_id=(
                            f"{material_id}_table_p{page}_"
                            f"T{_format_temperature_token(temperature_C)}C"
                        ),
                        source_type="table",
                        source_ref=f"Table (page {page})",
                        page=page,
                        temperature_C=temperature_C,
                        confidence_level=None,
                        points=[],
                        extraction_method="table",
                        extraction_confidence=0.6,
                        warnings=[],
                    ),
                )
                curve.points.append(
                    RelaxationPoint(
                        stress_MPa=stress,
                        relaxation_percent=relax,
                    )
                )

        parsed_curves = [curve for curve in curves_by_temperature.values() if curve.points]
        for curve in parsed_curves:
            curve.points.sort(key=lambda point: point.stress_MPa)
        if parsed_curves:
            return parsed_curves

    fallback_points: list[RelaxationPoint] = []
    for row in data[1:]:
        point = _parse_table_row_to_point(row)
        if point is not None:
            fallback_points.append(point)

    if not fallback_points:
        return []

    fallback_points.sort(key=lambda point: point.stress_MPa)
    return [
        RelaxationCurve(
            curve_id=f"{material_id}_table_p{page}_T20C",
            source_type="table",
            source_ref=f"Table (page {page})",
            page=page,
            temperature_C=20.0,
            confidence_level=None,
            points=fallback_points,
            extraction_method="table",
            extraction_confidence=0.5,
            warnings=[
                "Temperature assumed 20°C — not confirmed from table header"
            ],
        )
    ]


def extract_curves(
    pdf_path: Path,
    pdf_sha256: str = "",
    pdf_url: str = IST_REPORT_URL,
) -> tuple[RelaxationDataset, MetadataOutput]:
    """
    Extract relaxation curves from IST Report 330.

    Returns (dataset, metadata).
    """
    now = datetime.now(timezone.utc).isoformat()
    global_warnings: list[str] = []
    failed_items: list[dict[str, Any]] = []

    # --- Extract text and tables from the PDF ---
    try:
        pages_text = extract_text_by_page(pdf_path)
        pages_processed = len(pages_text)
    except Exception as exc:  # noqa: BLE001
        global_warnings.append(f"Text extraction failed: {exc}")
        pages_text = {}
        pages_processed = 0

    try:
        raw_tables = extract_tables_with_pdfplumber(pdf_path)
        tables_detected = len(raw_tables)
    except Exception as exc:  # noqa: BLE001
        global_warnings.append(f"Table extraction failed: {exc}")
        raw_tables = []
        tables_detected = 0

    try:
        digitized_curves, digitization_warnings, digitization_failures = (
            digitize_supported_figure_curves(pdf_path)
        )
        global_warnings.extend(digitization_warnings)
        failed_items.extend(digitization_failures)
    except Exception as exc:  # noqa: BLE001
        global_warnings.append(f"Figure digitisation bootstrap failed: {exc}")
        digitized_curves = {}

    # --- Count figure references across all pages ---
    all_text = "\n".join(pages_text.values())
    fig_refs = detect_figure_references(all_text)
    figures_detected = len(set(fig_refs))
    detect_table_references(all_text)

    if figures_detected == 0 and tables_detected == 0 and pages_processed == 0:
        global_warnings.append(
            "No text, figures, or tables could be extracted from the PDF. "
            "The PDF may be image-only or failed to open."
        )

    # --- Build material entries with placeholder curves ---
    # We cannot reliably auto-digitise raster figure images without
    # calibrated image processing.  We create structured placeholders instead.
    materials: list[Material] = []

    for mat_spec in KNOWN_MATERIALS:
        mat_curves: list[RelaxationCurve] = []
        mat_warnings: list[str] = []

        # Attempt to match table data for this material (Tables I–III are
        # two-column stress/relaxation tables in the report appendix).
        for tbl in raw_tables:
            data = tbl.get("data", [])
            if not data:
                continue
            # Heuristic: look for the material name in the first row(s)
            header_text = " ".join(
                str(cell) for row in data[:3] for cell in (row or []) if cell
            ).lower()
            mat_name_lower = mat_spec["material_name"].lower()
            alias_match = any(
                a.lower() in header_text for a in mat_spec["aliases"]
            )
            if mat_name_lower not in header_text and not alias_match:
                continue

            mat_curves.extend(_extract_table_curves(mat_spec["material_id"], tbl))

        mat_curves.extend(digitized_curves.get(mat_spec["material_id"], []))
        existing_curve_ids = {curve.curve_id for curve in mat_curves}

        # Create placeholder curves for figures (cannot be auto-digitised)
        for fig_num, temp_C in _iter_figure_temperature_pairs(mat_spec):
            curve_id = f"{mat_spec['material_id']}_fig{fig_num}_T{int(temp_C)}C"
            if curve_id in existing_curve_ids:
                continue

            curve = _make_placeholder_curve(
                mat_spec["material_id"],
                fig_num,
                temp_C,
                page_hint=FIGURE_PAGE_HINTS.get(fig_num, 0),
                warnings=mat_warnings,
            )
            mat_curves.append(curve)
            existing_curve_ids.add(curve_id)
            failed_items.append(
                {
                    "material": mat_spec["material_name"],
                    "source_ref": f"Figure {fig_num}",
                    "temperature_C": temp_C,
                    "reason": "Figure not auto-digitised — requires calibrated image processing",
                }
            )

        materials.append(
            Material(
                material_id=mat_spec["material_id"],
                material_name=mat_spec["material_name"],
                aliases=mat_spec["aliases"],
                family=mat_spec["family"],
                curves=mat_curves,
            )
        )

    if not global_warnings:
        global_warnings.append(
            "A subset of stress-relaxation figures is automatically digitised from raster pages. "
            "Unsupported figure/material combinations remain placeholder entries; no values were fabricated for those placeholders."
        )

    # Build dataset
    dataset = RelaxationDataset(
        dataset="IST/SRAMA Report 330 relaxation curves",
        source=SourceInfo(
            title="A Design Guide to the Stress Relaxation of Spring Materials",
            report_number="330",
            url=pdf_url,
            local_pdf=str(pdf_path),
            sha256=pdf_sha256,
            generated_at=now,
        ),
        license_notice=LicenseNotice(
            code_license="MIT",
            data_notice=(
                "Generated locally from a user-downloaded public PDF. "
                "Do not redistribute unless the source data license allows it."
            ),
        ),
        materials=materials,
        warnings=global_warnings,
    )

    # Build metadata
    metadata = MetadataOutput(
        source_pdf=str(pdf_path),
        source_url=pdf_url,
        sha256=pdf_sha256,
        extraction_timestamp=now,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        package_version=PACKAGE_VERSION,
        pages_processed=pages_processed,
        figures_detected=figures_detected,
        tables_detected=tables_detected,
        extraction_warnings=global_warnings,
        failed_extraction_items=failed_items,
    )

    return dataset, metadata
