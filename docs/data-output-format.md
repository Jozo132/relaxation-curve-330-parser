# Data Output Format

## `generated/ist_report_330_relaxation_curves.json`

Top-level structure:

```json
{
  "dataset": "IST/SRAMA Report 330 relaxation curves",
  "source": { ... },
  "license_notice": { ... },
  "materials": [ ... ],
  "warnings": [ ... ]
}
```

### `source`

| Field | Description |
|-------|-------------|
| `title` | Report title |
| `report_number` | "330" |
| `url` | Original PDF URL |
| `local_pdf` | Local path to the downloaded PDF |
| `sha256` | SHA-256 hash of the PDF |
| `generated_at` | ISO 8601 timestamp |

### `materials[]`

Each material entry:

| Field | Description |
|-------|-------------|
| `material_id` | Kebab-case identifier |
| `material_name` | Full material name |
| `aliases` | Alternative names |
| `family` | Material family (e.g. "steel", "nickel_alloy") |
| `curves` | List of relaxation curves |

### `curves[]`

Each curve entry:

| Field | Description |
|-------|-------------|
| `curve_id` | Unique identifier |
| `source_type` | "figure", "table", or "placeholder" |
| `source_ref` | e.g. "Figure 18" |
| `page` | PDF page number (0 if unknown) |
| `temperature_C` | Temperature in °C |
| `confidence_level` | "expected", "upper", "lower", or null |
| `points` | List of `{stress_MPa, relaxation_percent}` pairs |
| `extraction_method` | "automatic", "manual", "table", "placeholder" |
| `extraction_confidence` | 0.0–1.0 |
| `warnings` | List of warning strings |

## `generated/ist_report_330_metadata.json`

| Field | Description |
|-------|-------------|
| `source_pdf` | Local path to the PDF |
| `source_url` | Download URL |
| `sha256` | SHA-256 of the PDF |
| `extraction_timestamp` | ISO 8601 |
| `python_version` | Python version used |
| `package_version` | Package version |
| `pages_processed` | Number of PDF pages processed |
| `figures_detected` | Number of figure references found |
| `tables_detected` | Number of tables extracted |
| `extraction_warnings` | List of warnings |
| `failed_extraction_items` | Items that could not be extracted |
