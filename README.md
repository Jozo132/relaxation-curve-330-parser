# relaxation-curve-330-parser

A Python-based open-source tool for local parsing and relaxation-curve extraction from
IST/SRAMA Report No. 330: *"A Design Guide to the Stress Relaxation of Spring Materials"*.

## License

The **code** in this repository is MIT licensed. See [LICENSE](LICENSE).

The **downloaded PDF** and **generated JSON data** are NOT committed to this repository:
- The PDF is downloaded to `data/` and ignored by `.gitignore`
- Generated JSON is written to `generated/` and ignored by `.gitignore`
- Generated curve data is for **local use only** — do not redistribute unless you have
  confirmed redistribution rights from IST

## Important notices

- **EN 13906-1 data is not included** in this project
- **Predictions are engineering estimates**, not guarantees
- **Figure data is placeholder only** — automated digitisation of raster figures was not
  attempted; no values are fabricated

## Getting started

### Prerequisites

- Node.js (for the npm workflow)
- Python 3.10+

### Setup

```bash
npm install
```

`npm install` now bootstraps the local `.venv` and installs the package in editable mode with
development dependencies. To rerun the bootstrap explicitly later, use:

```bash
npm run install
```

The npm scripts (`build`, `test`, `lint`, `typecheck`) use the project `.venv` automatically.

To launch the manual curve calibration UI, use:

```bash
npm run calibrate
```

The calibration tool opens a Python notebook-style UI with one tab per supported figure,
lets you select each legend item, and lets you place manual start and end points by clicking
the overlay image. `Apply` writes the saved overrides to `data/curve_calibration_overrides.json`
and refreshes the generated figure previews. `Cancel` exits without saving changes.

### Run the build

```bash
npm run build
```

Expected output:

```
PDF already exists locally: data/Research-Report-330-A-Design-Guide-To-The-Stress-Relaxation-of-Spring-Materials.pdf

Downloaded or found PDF:
data/Research-Report-330-A-Design-Guide-To-The-Stress-Relaxation-of-Spring-Materials.pdf

Generated relaxation curve JSON:
generated/ist_report_330_relaxation_curves.json

Generated metadata JSON:
generated/ist_report_330_metadata.json
```

Current extraction coverage:

- Table data from parsed tables is extracted directly, including header-derived temperatures when available.
- Figures 18 and 19 are automatically digitised for phosphor bronze, beryllium copper, and titanium alloy.
- Figures 20 and 21 are automatically digitised for patented carbon steel, oil hardened and tempered steel, silicon-chromium steel, and 18Cr/8Ni stainless steel where those curves appear.
- Figures 22 through 26 are automatically digitised for the supported exact materials on those pages, including Inconel 600, tungsten steel, 18Ni-Co-Mo maraging steel, A286, Inconel X-750, and Elgiloy where present.
- Remaining unsupported figure/material combinations stay as explicit placeholders with warnings.

### Run tests

```bash
npm run test
```

### Run linting

```bash
npm run lint
```

### Run type checking

```bash
npm run typecheck
```

## CLI usage

```bash
# Download the PDF
python -m spring_relaxation_ist download

# Extract curves
python -m spring_relaxation_ist extract

# Open the manual calibration UI
python -m spring_relaxation_ist.cli calibrate

# Validate generated JSON
python -m spring_relaxation_ist validate

# Predict relaxation
python -m spring_relaxation_ist predict \
  --material "Inconel X-750" \
  --temperature 300 \
  --stress 500
```

## Python API

```python
from spring_relaxation_ist.prediction import predict_relaxation

result = predict_relaxation(
    json_path="generated/ist_report_330_relaxation_curves.json",
    material="Inconel X-750",
    temperature_C=300,
    stress_MPa=500,
)
print(result)
```

## Methodology

See [docs/methodology.md](docs/methodology.md).

## Licensing details

See [docs/licensing.md](docs/licensing.md).
