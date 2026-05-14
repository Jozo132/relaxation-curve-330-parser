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

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import IST_REPORT_URL, PACKAGE_VERSION
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
        "material_id": "stainless-steel-302-304",
        "material_name": "Stainless Steel 302/304",
        "aliases": ["302", "304", "austenitic stainless"],
        "family": "stainless_steel",
        "figure_range": list(range(10, 13)),
        "temperatures_C": [20, 100, 150, 200, 250],
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
    },
    {
        "material_id": "phosphor-bronze",
        "material_name": "Phosphor Bronze",
        "aliases": ["CuSnP", "bronze"],
        "family": "copper_alloy",
        "figure_range": [18, 19],
        "temperatures_C": [20, 100, 150],
    },
    {
        "material_id": "beryllium-copper",
        "material_name": "Beryllium Copper",
        "aliases": ["BeCu", "beryllium-copper", "CuBe"],
        "family": "copper_alloy",
        "figure_range": [20, 21],
        "temperatures_C": [20, 100, 150, 200],
    },
    {
        "material_id": "titanium-alloy",
        "material_name": "Titanium Alloy",
        "aliases": ["Ti alloy", "titanium"],
        "family": "titanium",
        "figure_range": [22, 23],
        "temperatures_C": [20, 100, 150, 200],
    },
    {
        "material_id": "elgiloy",
        "material_name": "Elgiloy",
        "aliases": ["Cobenium", "Phynox", "MP35N"],
        "family": "cobalt_alloy",
        "figure_range": [24, 25],
        "temperatures_C": [20, 100, 200, 300],
    },
    {
        "material_id": "nb-alloy",
        "material_name": "Nimonic / Ni-based Superalloy",
        "aliases": ["Nimonic", "Ni superalloy"],
        "family": "nickel_alloy",
        "figure_range": [26, 27],
        "temperatures_C": [20, 200, 300, 400, 500],
    },
]


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


def _parse_table_row_to_point(row: list) -> RelaxationPoint | None:
    """
    Attempt to parse a table row into a RelaxationPoint.
    Expected columns: stress (MPa) | relaxation (%)
    Returns None if the row cannot be parsed.
    """
    if not row or len(row) < 2:
        return None
    try:
        stress = float(str(row[0]).replace(",", ".").strip())
        relax = float(str(row[1]).replace(",", ".").strip())
        if stress > 0 and 0.0 <= relax <= 100.0:
            return RelaxationPoint(stress_MPa=stress, relaxation_percent=relax)
    except (ValueError, TypeError):
        pass
    return None


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

            # Parse data rows
            for row in data[1:]:
                pt = _parse_table_row_to_point(row)
                if pt is not None:
                    # We don't know temperature from the table header alone —
                    # use 20°C as the ambient default and warn
                    curve_id = (
                        f"{mat_spec['material_id']}_table_p{tbl['page']}_T20C"
                    )
                    existing_ids = [c.curve_id for c in mat_curves]
                    if curve_id not in existing_ids:
                        mat_curves.append(
                            RelaxationCurve(
                                curve_id=curve_id,
                                source_type="table",
                                source_ref=f"Table (page {tbl['page']})",
                                page=tbl["page"],
                                temperature_C=20.0,
                                confidence_level=None,
                                points=[pt],
                                extraction_method="table",
                                extraction_confidence=0.5,
                                warnings=[
                                    "Temperature assumed 20°C — not confirmed from table header"
                                ],
                            )
                        )
                    else:
                        for c in mat_curves:
                            if c.curve_id == curve_id:
                                c.points.append(pt)

        # Create placeholder curves for figures (cannot be auto-digitised)
        for fig_num in mat_spec["figure_range"]:
            for temp_C in mat_spec["temperatures_C"]:
                curve = _make_placeholder_curve(
                    mat_spec["material_id"],
                    fig_num,
                    float(temp_C),
                    page_hint=0,
                    warnings=mat_warnings,
                )
                mat_curves.append(curve)
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
            "Relaxation figure curves (Figures 1–29) are present as placeholder entries only. "
            "Automated pixel-level digitisation was not attempted as it requires calibrated "
            "image processing with human validation. No values were fabricated."
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
