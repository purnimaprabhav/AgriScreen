import re
import pandas as pd
from pathlib import Path
from typing import Optional

from utils.constants import (
    COMPANIES_DIR, REPORTS_DIR, NEWS_DIR, MARKET_DATA_DIR,
    FINANCIALS_CSV, FUNDING_CSV, LATEST_YEAR,
    FACTSHEET_HEADER_PATTERN, SECTION_HEADER_PATTERN,
    SCORING_INPUTS_MARKER, NEWS_ARTICLE_PATTERN,
)


# ════════════════════════════════════════════════════════
# FACTSHEET LOADERS
# ════════════════════════════════════════════════════════

from utils.constants import COMPANY_ID_MAP

def _extract_company_name(text: str) -> str | None:
    match = FACTSHEET_HEADER_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(1).strip()
    # Cross-reference with known names to get correct casing
    for known_name in COMPANY_ID_MAP:
        if known_name.upper() == raw.upper():
            return known_name
    return raw  # fallback to whatever was extracted


def _parse_scoring_inputs(block: str) -> dict:
    """
    Converts the SCORING INPUTS text block into a typed Python dict.
    EDA finding: each line is 'key   : value' with consistent spacing.
    """
    result = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith(('=', '-', '#')):
            continue
        if ':' not in line:
            continue

        key, _, raw_value = line.partition(':')
        key        = key.strip()
        raw_value  = raw_value.strip()

        if not key or not raw_value:
            continue

        # Cast to correct Python type
        if raw_value.lower() == 'true':
            result[key] = True
        elif raw_value.lower() == 'false':
            result[key] = False
        else:
            try:
                result[key] = int(raw_value)
            except ValueError:
                try:
                    result[key] = float(raw_value)
                except ValueError:
                    result[key] = raw_value

    return result


def load_factsheet(path: Path) -> dict:
    """
    Loads one factsheet TXT file.

    EDA findings applied:
      - Split at SCORING INPUTS to separate narrative from structured data
      - Section headers are ALL CAPS + dashes — split into named sections
      - Company name is in the header line

    Returns:
        {
            'company_name'   : str,
            'source_file'    : str,
            'sections'       : { section_name: text },
            'scoring_inputs' : { key: typed_value },
        }
    """
    text = path.read_text(encoding='utf-8')

    # ── 1. Separate narrative from SCORING INPUTS ──
    if SCORING_INPUTS_MARKER in text:
        narrative, _, scoring_block = text.partition(SCORING_INPUTS_MARKER)
    else:
        narrative     = text
        scoring_block = ""

    # ── 2. Extract company name ──
    company_name = _extract_company_name(text)

    # ── 3. Split narrative into sections ──
    # re.split on the header pattern returns:
    # [preamble, HEADER_1, content_1, HEADER_2, content_2, ...]
    parts    = re.split(SECTION_HEADER_PATTERN, narrative)
    sections = {}

    i = 1  # skip preamble at index 0
    while i < len(parts) - 1:
        section_name    = parts[i].strip()
        section_content = parts[i + 1].strip()
        if section_name and section_content:
            sections[section_name] = section_content
        i += 2

    # ── 4. Parse scoring inputs ──
    scoring_inputs = _parse_scoring_inputs(scoring_block) if scoring_block else {}

    return {
        'company_name'   : company_name,
        'source_file'    : path.name,
        'sections'       : sections,
        'scoring_inputs' : scoring_inputs,
    }


def load_all_factsheets() -> list[dict]:
    """Loads every .txt file in the companies/ directory."""
    return [load_factsheet(p) for p in sorted(COMPANIES_DIR.glob("*.txt"))]


# ════════════════════════════════════════════════════════
# NEWS LOADERS
# ════════════════════════════════════════════════════════

def load_news_digest(path: Path) -> dict:
    """
    Loads one news digest TXT and splits it into individual articles.

    EDA finding: each article is delimited by [NEWS-XXX] — date
    and contains: headline on the next line, then body text.

    Returns:
        {
            'source_file': str,
            'articles'   : [
                {
                    'news_id'  : 'NEWS-006',
                    'date'     : '15 March 2026',
                    'headline' : str,
                    'body'     : str,
                }
            ]
        }
    """
    text     = path.read_text(encoding='utf-8')
    articles = []

    for match in NEWS_ARTICLE_PATTERN.finditer(text):
        num, date, headline, body = match.groups()
        articles.append({
            'news_id'  : f'NEWS-{num.zfill(3)}',
            'date'     : date.strip(),
            'headline' : headline.strip(),
            'body'     : body.strip(),
        })

    return {
        'source_file': path.name,
        'articles'   : articles,
    }


def load_all_news() -> list[dict]:
    """Loads every .txt file in the news/ directory."""
    return [load_news_digest(p) for p in sorted(NEWS_DIR.glob("*.txt"))]


# ════════════════════════════════════════════════════════
# REPORT LOADERS
# ════════════════════════════════════════════════════════

def load_report(path: Path) -> dict:
    """
    Loads one report TXT as a single text block.
    Sections are interconnected (cross-references between pillars),
    so we don't split here — chunker.py handles the splitting.

    Returns:
        {
            'source_file': str,
            'text'       : str,
        }
    """
    return {
        'source_file': path.name,
        'text'       : path.read_text(encoding='utf-8'),
    }


def load_all_reports() -> list[dict]:
    """Loads every .txt file in the reports/ directory."""
    return [load_report(p) for p in sorted(REPORTS_DIR.glob("*.txt"))]


# ════════════════════════════════════════════════════════
# CSV LOADERS
# ════════════════════════════════════════════════════════

def load_financials() -> pd.DataFrame:
    """Returns the full 3-year financials DataFrame (2023–2025)."""
    return pd.read_csv(MARKET_DATA_DIR / FINANCIALS_CSV)


def load_funding() -> pd.DataFrame:
    """Returns the full funding rounds DataFrame."""
    return pd.read_csv(MARKET_DATA_DIR / FUNDING_CSV)


def filter_latest_financials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to FY2025 only.
    EDA finding: CSV has exactly 3 rows per company; 2025 = latest.
    Used directly by the scoring layer.
    """
    return df[df['fy_year'] == LATEST_YEAR].reset_index(drop=True)


# ════════════════════════════════════════════════════════
# MASTER LOADER
# ════════════════════════════════════════════════════════

def load_all() -> dict:
    """
    Runs all loaders and returns everything in one dict.
    This is the only function chunker.py needs to call.

    Returns:
        {
            'factsheets'    : list[dict],
            'news'          : list[dict],
            'reports'       : list[dict],
            'financials_df' : pd.DataFrame,   # full 2023–2025
            'funding_df'    : pd.DataFrame,
        }
    """
    print("Loading factsheets...")
    factsheets = load_all_factsheets()
    print(f"  ✓ {len(factsheets)} factsheets loaded")

    print("Loading news digests...")
    news = load_all_news()
    total_articles = sum(len(d['articles']) for d in news)
    print(f"  ✓ {len(news)} files → {total_articles} articles")

    print("Loading reports...")
    reports = load_all_reports()
    print(f"  ✓ {len(reports)} reports loaded")

    print("Loading CSVs...")
    financials_df = load_financials()
    funding_df    = load_funding()
    print(f"  ✓ financials: {financials_df.shape} | funding: {funding_df.shape}")

    return {
        'factsheets'    : factsheets,
        'news'          : news,
        'reports'       : reports,
        'financials_df' : financials_df,
        'funding_df'    : funding_df,
    }