"""Command-line interface for spring_relaxation_ist."""

from __future__ import annotations

import json
import sys

import click

from .config import CURVES_JSON_PATH, FIGURE_PREVIEWS_DIR, GENERATED_DIR, METADATA_JSON_PATH, PDF_PATH


@click.group()
def main() -> None:
    """IST Report 330 relaxation curve parser."""


@main.command()
def download() -> None:
    """Download the IST Report 330 PDF if not already present."""
    from .downloader import download_pdf

    try:
        result = download_pdf()
        click.echo(f"PDF ready: {result['local_pdf']}")
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command()
def extract() -> None:
    """Extract relaxation curves from the PDF and write JSON output."""
    from .curve_extractor import extract_curves
    from .downloader import download_pdf, sha256_of_file
    from .figure_digitizer import generate_supported_figure_previews

    if not PDF_PATH.exists():
        click.echo("PDF not found. Downloading...")
        download_pdf()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    sha = sha256_of_file(PDF_PATH)
    dataset, metadata = extract_curves(PDF_PATH, pdf_sha256=sha)

    with open(CURVES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)
    click.echo(f"Generated relaxation curve JSON:\n{CURVES_JSON_PATH}")

    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)
    click.echo(f"Generated metadata JSON:\n{METADATA_JSON_PATH}")

    preview_paths, preview_warnings = generate_supported_figure_previews(
        PDF_PATH,
        FIGURE_PREVIEWS_DIR,
    )
    if preview_paths:
        click.echo(f"Generated figure overlay previews:\n{FIGURE_PREVIEWS_DIR}")
    for warning in preview_warnings:
        click.echo(f"Preview warning: {warning}", err=True)


@main.command()
def validate() -> None:
    """Validate the generated JSON files against the schema."""
    from .schema import MetadataOutput, RelaxationDataset

    errors: list[str] = []
    for path, schema in [
        (CURVES_JSON_PATH, RelaxationDataset),
        (METADATA_JSON_PATH, MetadataOutput),
    ]:
        if not path.exists():
            errors.append(f"Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        try:
            schema.model_validate(data)  # type: ignore[attr-defined]
            click.echo(f"OK: {path}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Invalid {path}: {exc}")

    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--material", required=True, help="Material name")
@click.option("--temperature", required=True, type=float, help="Temperature in °C")
@click.option("--stress", required=True, type=float, help="Stress in MPa")
@click.option(
    "--json-path",
    default=str(CURVES_JSON_PATH),
    help="Path to relaxation curves JSON",
)
def predict(material: str, temperature: float, stress: float, json_path: str) -> None:
    """Predict relaxation percent using the generated JSON."""
    from .prediction import predict_relaxation

    result = predict_relaxation(
        json_path=json_path,
        material=material,
        temperature_C=temperature,
        stress_MPa=stress,
    )
    click.echo(json.dumps(result, indent=2))
    if result.get("warnings"):
        for w in result["warnings"]:
            click.echo(f"Warning: {w}", err=True)
