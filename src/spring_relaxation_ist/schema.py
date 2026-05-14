"""Pydantic schemas for relaxation curve data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RelaxationPoint(BaseModel):
    """A single (stress, relaxation) data point."""

    stress_MPa: float
    relaxation_percent: float


class RelaxationCurve(BaseModel):
    """A relaxation curve extracted from a figure or table."""

    curve_id: str
    source_type: str  # "figure" | "table" | "placeholder"
    source_ref: str  # e.g. "Figure 18"
    page: int
    temperature_C: float
    confidence_level: str | None = None  # e.g. "expected", "upper", "lower"
    points: list[RelaxationPoint] = Field(default_factory=list)
    extraction_method: str  # "automatic" | "manual" | "table" | "placeholder"
    extraction_confidence: float = 0.0  # 0..1
    warnings: list[str] = Field(default_factory=list)


class Material(BaseModel):
    """A spring material with associated relaxation curves."""

    material_id: str
    material_name: str
    aliases: list[str] = Field(default_factory=list)
    family: str
    curves: list[RelaxationCurve] = Field(default_factory=list)


class SourceInfo(BaseModel):
    """Source PDF metadata."""

    title: str
    report_number: str
    url: str
    local_pdf: str
    sha256: str
    generated_at: str


class LicenseNotice(BaseModel):
    """License information."""

    code_license: str
    data_notice: str


class RelaxationDataset(BaseModel):
    """Top-level dataset structure."""

    dataset: str
    source: SourceInfo
    license_notice: LicenseNotice
    materials: list[Material] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MetadataOutput(BaseModel):
    """Metadata JSON output."""

    source_pdf: str
    source_url: str
    sha256: str
    extraction_timestamp: str
    python_version: str
    package_version: str
    pages_processed: int
    figures_detected: int
    tables_detected: int
    extraction_warnings: list[str] = Field(default_factory=list)
    failed_extraction_items: list[dict[str, Any]] = Field(default_factory=list)
