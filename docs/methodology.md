# Methodology

## Overview

This project extracts relaxation curve data from IST/SRAMA Report No. 330:
*"A Design Guide to the Stress Relaxation of Spring Materials"*.

## Extraction approach

### Text extraction

PyMuPDF (`fitz`) is used to extract page text for detecting material names, table references,
and figure references in the report.

### Table extraction

`pdfplumber` is used to parse tabular data (Tables I–III) from the PDF. Where tables are
successfully parsed, stress and relaxation values are extracted directly. When table headers
contain explicit temperatures, separate curves are created per temperature column. If a table
can be parsed but its temperature header cannot be resolved, the extracted curve falls back to
an ambient 20°C assumption and carries a warning.

### Figure extraction (Figures 1–29)

The relaxation curves in Figures 1–29 are embedded as **raster images** in the PDF.
The current implementation has two figure-extraction modes:

- Figures 18 and 19 are automatically digitised for phosphor bronze, beryllium copper, and titanium alloy using image-based axis detection and curve tracing.
- Figures 20 and 21 are automatically digitised for patented carbon steel, oil hardened and tempered steel, silicon-chromium steel, and 18Cr/8Ni stainless steel.
- Figures 22 through 26 are automatically digitised for the supported exact materials shown on those pages, including Inconel 600, tungsten steel, 18Ni-Co-Mo maraging steel, A286, Inconel X-750, and Elgiloy where present.
- Unsupported figures remain structured placeholders.

General automated pixel-level digitisation still requires:
- Axis calibration (identifying axis tick values)
- Curve tracing (separating overlapping curve lines by colour/style)
- Human validation of traced points

That broader level of image processing is not implemented in the current version.
Unsupported figure entries are therefore represented as **structured placeholders** with:
- `extraction_method: "placeholder"`
- `extraction_confidence: 0.0`
- `points: []`
- A warning message

**No values are fabricated.** Placeholder entries exist so the schema is complete
and downstream code can detect which entries require manual digitisation.

## Adding real data

To add actual curve data:
1. Manually digitise figures using a tool like [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/).
2. Update the curve entries in `generated/ist_report_330_relaxation_curves.json`.
3. Set `extraction_method` to `"manual"` and `extraction_confidence` to an appropriate value (e.g. 0.8).

## Interpolation

Linear interpolation (`numpy.interp`) is used within the range of extracted data.
Extrapolation is not supported — requests outside the known range raise an error.
