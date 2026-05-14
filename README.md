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

- Node.js (for `npm run build`)
- Python 3.10+

### Setup

```bash
npm install
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

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
