"""Configuration constants for the spring relaxation IST parser."""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent.parent

# Data and output directories
DATA_DIR = ROOT_DIR / "data"
GENERATED_DIR = ROOT_DIR / "generated"

# PDF source
IST_REPORT_URL = (
    "https://ist.org.uk/wp-content/uploads/_pda/2024/07/"
    "Research-Report-330-A-Design-Guide-To-The-Stress-Relaxation-of-Spring-Materials.pdf"
)
PDF_FILENAME = "Research-Report-330-A-Design-Guide-To-The-Stress-Relaxation-of-Spring-Materials.pdf"
PDF_PATH = DATA_DIR / PDF_FILENAME

# Output files
CURVES_JSON_PATH = GENERATED_DIR / "ist_report_330_relaxation_curves.json"
METADATA_JSON_PATH = GENERATED_DIR / "ist_report_330_metadata.json"

# Package version
PACKAGE_VERSION = "1.0.0"
