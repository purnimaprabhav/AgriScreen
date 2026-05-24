import re
from pathlib import Path


BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
COMPANIES_DIR   = DATA_DIR / "companies"
REPORTS_DIR     = DATA_DIR / "reports"
NEWS_DIR        = DATA_DIR / "news"
MARKET_DATA_DIR = DATA_DIR / "market_data"


# ── CSV filenames ────────────────────────────────────────
FINANCIALS_CSV = "companies_financials_2023_2025.csv"
FUNDING_CSV    = "funding_rounds.csv"
LATEST_YEAR    = 2025

# ── Doc type labels ──────────────────────────────────────
DOC_TYPE_FACTSHEET  = "factsheet"
DOC_TYPE_NEWS       = "news"
DOC_TYPE_REPORT     = "report"
DOC_TYPE_FINANCIALS = "financials"
DOC_TYPE_FUNDING    = "funding"

# ── Factsheet parsing ────────────────────────────────────
# EDA finding: header line is always "COMPANY FACTSHEET — COMPANY NAME"
FACTSHEET_HEADER_PATTERN = re.compile(
    r'COMPANY FACTSHEET\s*[—-]\s*(.+)'
)

# EDA finding: section headers are ALL CAPS followed by a dashes line
SECTION_HEADER_PATTERN = re.compile(
    r'\n([A-Z][A-Z &/]+)\n-{3,}'
)

# EDA finding: SCORING INPUTS is the exact marker that splits narrative from data
SCORING_INPUTS_MARKER = "SCORING INPUTS"

# ── News parsing ─────────────────────────────────────────
# EDA finding: every article starts with [NEWS-XXX] — date on same line
NEWS_ARTICLE_PATTERN = re.compile(
    r'\[NEWS-(\d+)\]\s*[—-]\s*([^\n]+)\n([^\n]+)\n(.*?)(?=\[NEWS-|\Z)',
    re.DOTALL
)

# ── Company registry ─────────────────────────────────────
# Maps full company name (as written in factsheets) → CSV company_id
COMPANY_ID_MAP = {
    "Verdant Farms SA":          "VF-001",
    "GreenYield Technologies BV": "GY-002",
    "SoilSense AI Ltd":          "SS-003",
    "AquaGrow Solutions Ltd":    "AG-004",
    "HarvestLink GmbH":          "HL-005",
    "BioRoot Innovations SA":    "BR-006",
}

KNOWN_COMPANIES = list(COMPANY_ID_MAP.keys())