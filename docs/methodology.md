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
successfully parsed, stress and relaxation values are extracted directly.

### Figure extraction (Figures 1–29)

The relaxation curves in Figures 1–29 are embedded as **raster images** in the PDF.
Automated pixel-level digitisation requires:
- Axis calibration (identifying axis tick values)
- Curve tracing (separating overlapping curve lines by colour/style)
- Human validation of traced points

This level of image processing is not implemented in the current version.
All figure entries are therefore represented as **structured placeholders** with:
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
