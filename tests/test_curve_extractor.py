"""Tests for table-driven curve extraction behavior."""

from __future__ import annotations

from pathlib import Path

from spring_relaxation_ist import curve_extractor
from spring_relaxation_ist.schema import Material, RelaxationDataset


def _material_by_id(dataset: RelaxationDataset, material_id: str) -> Material:
    return next(material for material in dataset.materials if material.material_id == material_id)


def test_extract_curves_parses_temperature_columns_from_table_headers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_tables = [
        {
            "page": 12,
            "table_index": 0,
            "data": [
                ["Hard drawn carbon steel", "20°C", "150°C"],
                ["Stress (MPa)", "Relaxation (%)", "Relaxation (%)"],
                ["200", "2.0", "3.0"],
                ["100", "1.0", "1.8"],
            ],
        }
    ]

    monkeypatch.setattr(curve_extractor, "extract_text_by_page", lambda _: {1: "Table I"})
    monkeypatch.setattr(curve_extractor, "extract_tables_with_pdfplumber", lambda _: raw_tables)
    monkeypatch.setattr(curve_extractor, "detect_figure_references", lambda _: [])
    monkeypatch.setattr(curve_extractor, "detect_table_references", lambda _: ["Table I"])

    dataset, metadata = curve_extractor.extract_curves(
        tmp_path / "dummy.pdf",
        pdf_sha256="abc123",
    )

    material = _material_by_id(dataset, "carbon-steel")
    table_curves = [curve for curve in material.curves if curve.source_type == "table"]

    assert metadata.tables_detected == 1
    assert {curve.temperature_C for curve in table_curves} == {20.0, 150.0}

    curve_20 = next(curve for curve in table_curves if curve.temperature_C == 20.0)
    curve_150 = next(curve for curve in table_curves if curve.temperature_C == 150.0)

    assert [point.stress_MPa for point in curve_20.points] == [100.0, 200.0]
    assert [point.relaxation_percent for point in curve_20.points] == [1.0, 2.0]
    assert [point.relaxation_percent for point in curve_150.points] == [1.8, 3.0]
    assert curve_20.warnings == []
    assert curve_150.warnings == []


def test_extract_curves_falls_back_to_ambient_temperature_without_header_temperature(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_tables = [
        {
            "page": 7,
            "table_index": 0,
            "data": [
                ["Hard drawn carbon steel", "Relaxation (%)"],
                ["100", "1.0"],
                ["200", "2.0"],
            ],
        }
    ]

    monkeypatch.setattr(curve_extractor, "extract_text_by_page", lambda _: {1: "Table I"})
    monkeypatch.setattr(curve_extractor, "extract_tables_with_pdfplumber", lambda _: raw_tables)
    monkeypatch.setattr(curve_extractor, "detect_figure_references", lambda _: [])
    monkeypatch.setattr(curve_extractor, "detect_table_references", lambda _: ["Table I"])

    dataset, _ = curve_extractor.extract_curves(
        tmp_path / "dummy.pdf",
        pdf_sha256="abc123",
    )

    material = _material_by_id(dataset, "carbon-steel")
    table_curves = [curve for curve in material.curves if curve.source_type == "table"]

    assert len(table_curves) == 1
    assert table_curves[0].temperature_C == 20.0
    assert table_curves[0].warnings == [
        "Temperature assumed 20°C — not confirmed from table header"
    ]
    assert [point.relaxation_percent for point in table_curves[0].points] == [1.0, 2.0]


def test_iter_figure_temperature_pairs_respects_explicit_empty_mapping() -> None:
    assert curve_extractor._iter_figure_temperature_pairs({"figure_temperatures": []}) == []