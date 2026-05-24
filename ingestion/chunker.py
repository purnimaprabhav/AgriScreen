"""
ingestion/chunker.py

Takes the raw output of loader.py and produces a flat list of Chunk objects.
One chunking strategy per data type — no one-size-fits-all splitting here.

Strategies:
  Factsheets → one Chunk per section + one Chunk for SCORING INPUTS
  News       → one Chunk per article (already natural boundaries)
  Reports    → sliding window (500 chars / 100 overlap) — no clean section breaks
  CSVs       → one narrative Chunk per company (human-readable, queryable)
"""

import re
import pandas as pd
from utils.constants import (
    KNOWN_COMPANIES, COMPANY_ID_MAP,
    DOC_TYPE_FACTSHEET, DOC_TYPE_NEWS,
    DOC_TYPE_REPORT, DOC_TYPE_FINANCIALS, DOC_TYPE_FUNDING,
)
from utils.metadata import Chunk, ChunkMetadata

# Report chunking config — tuned for 5-6KB report files
REPORT_CHUNK_SIZE    = 500   # characters
REPORT_CHUNK_OVERLAP = 100   # characters


# ════════════════════════════════════════════════════════
# FACTSHEET CHUNKER
# ════════════════════════════════════════════════════════

def chunk_factsheet(factsheet: dict) -> list[Chunk]:
    """
    Produces one Chunk per narrative section + one Chunk for SCORING INPUTS.

    EDA finding: sections range from 280–780 chars — small enough to keep whole.
    Keeping sections intact means a query about "technology" retrieves the full
    TECHNOLOGY section, not half of it.
    """
    chunks  = []
    company = factsheet['company_name']
    source  = factsheet['source_file']
    co_id   = COMPANY_ID_MAP.get(company)

    # ── Narrative sections ───────────────────────────────
    for section_name, section_text in factsheet['sections'].items():
        if not section_text.strip():
            continue

        chunks.append(Chunk(
            text=f"{section_name}\n\n{section_text}",
            metadata=ChunkMetadata(
                doc_type    = DOC_TYPE_FACTSHEET,
                source_file = source,
                company     = company,
                company_id  = co_id,
                section     = section_name,
            )
        ))

    # ── SCORING INPUTS block ─────────────────────────────
    # Stored as text so it's retrievable via natural language,
    # e.g. "what is AquaGrow's gross margin?"
    if factsheet['scoring_inputs']:
        scoring_text = "\n".join(
            f"{k}: {v}" for k, v in factsheet['scoring_inputs'].items()
        )
        chunks.append(Chunk(
            text=f"SCORING INPUTS for {company}\n\n{scoring_text}",
            metadata=ChunkMetadata(
                doc_type         = DOC_TYPE_FACTSHEET,
                source_file      = source,
                company          = company,
                company_id       = co_id,
                section          = "SCORING INPUTS",
                is_scoring_inputs= True,
            )
        ))

    return chunks


def chunk_all_factsheets(factsheets: list[dict]) -> list[Chunk]:
    chunks = []
    for fs in factsheets:
        chunks.extend(chunk_factsheet(fs))
    return chunks


# ════════════════════════════════════════════════════════
# NEWS CHUNKER
# ════════════════════════════════════════════════════════

def _detect_companies(text: str) -> list[str]:
    """
    Matches company names even when legal suffix (Ltd, GmbH, BV, SA) is omitted.
    EDA finding: news articles drop suffixes e.g. 'AquaGrow Solutions' not 'AquaGrow Solutions Ltd'
    """
    text_lower = text.lower()
    found = []
    for company in KNOWN_COMPANIES:
        # Strip legal suffixes for matching
        short = re.sub(r'\b(ltd|gmbh|bv|sa|llc)\b\.?', '', company, flags=re.I).strip()
        if short.lower() in text_lower:
            found.append(company)
    return found


def chunk_news_digest(digest: dict) -> list[Chunk]:
    """
    One Chunk per article — articles are already the right size (500–600 chars).

    The full text is: headline + body, so embeddings capture both.
    company field is set to the primary company mentioned (first match),
    so retriever.py can filter by company later.
    """
    chunks  = []
    source  = digest['source_file']

    for article in digest['articles']:
        full_text = f"{article['headline']}\n\n{article['body']}"
        mentioned = _detect_companies(full_text)

        chunks.append(Chunk(
            text=full_text,
            metadata=ChunkMetadata(
                doc_type    = DOC_TYPE_NEWS,
                source_file = source,
                company     = mentioned[0] if mentioned else None,
                company_id  = COMPANY_ID_MAP.get(mentioned[0]) if mentioned else None,
                news_id     = article['news_id'],
                date        = article['date'],
            )
        ))

    return chunks


def chunk_all_news(digests: list[dict]) -> list[Chunk]:
    chunks = []
    for digest in digests:
        chunks.extend(chunk_news_digest(digest))
    return chunks


# ════════════════════════════════════════════════════════
# REPORT CHUNKER
# ════════════════════════════════════════════════════════

def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """
    Splits text into overlapping windows.
    Used for reports where sections are too interconnected to split cleanly.
    """
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += size - overlap
    return chunks


def chunk_report(report: dict) -> list[Chunk]:
    """
    Splits report text into overlapping windows.
    No company tag — reports are cross-portfolio context.
    """
    chunks  = []
    source  = report['source_file']
    windows = _sliding_window(
        report['text'],
        size    = REPORT_CHUNK_SIZE,
        overlap = REPORT_CHUNK_OVERLAP,
    )

    for i, window in enumerate(windows):
        chunks.append(Chunk(
            text=window,
            metadata=ChunkMetadata(
                doc_type    = DOC_TYPE_REPORT,
                source_file = source,
            )
        ))

    return chunks


def chunk_all_reports(reports: list[dict]) -> list[Chunk]:
    chunks = []
    for report in reports:
        chunks.extend(chunk_report(report))
    return chunks


# ════════════════════════════════════════════════════════
# CSV CHUNKERS
# ════════════════════════════════════════════════════════

def _format_financials_row(row: pd.Series) -> str:
    """Formats one year of financials as a readable sentence."""
    runway = "profitable (no runway constraint)" if row['runway_months'] == 999 \
             else f"{row['runway_months']} months"
    return (
        f"FY{row['fy_year']}: Revenue €{row['revenue_eur_k']:,}K "
        f"(growth {row['revenue_growth_pct']:+}%), "
        f"Gross Margin {row['gross_margin_pct']}%, "
        f"EBITDA €{row['ebitda_eur_k']:,}K, "
        f"ARR €{row['arr_eur_k']:,}K, "
        f"Cash €{row['cash_eur_k']:,}K, "
        f"Runway: {runway}, "
        f"FTEs: {row['fte_count']}"
    )


def chunk_financials(df: pd.DataFrame) -> list[Chunk]:
    """
    One Chunk per company covering all 3 years of financials.

    EDA finding: 3 rows per company (2023–2025) — grouping them keeps
    trend context together, so queries like "how has revenue grown?" work.
    """
    chunks = []

    for company_id, group in df.groupby('company_id'):
        group       = group.sort_values('fy_year')
        company     = group.iloc[0]['company_name']
        year_lines  = [_format_financials_row(row) for _, row in group.iterrows()]

        text = (
            f"Financial history for {company} ({company_id}):\n\n"
            + "\n".join(year_lines)
        )

        chunks.append(Chunk(
            text=text,
            metadata=ChunkMetadata(
                doc_type    = DOC_TYPE_FINANCIALS,
                source_file = "companies_financials_2023_2025.csv",
                company     = company,
                company_id  = company_id,
            )
        ))

    return chunks


def chunk_funding(df: pd.DataFrame) -> list[Chunk]:
    """
    One Chunk per company covering full funding history.
    """
    chunks = []

    for company_id, group in df.groupby('company_id'):
        group   = group.sort_values('round_date')
        company = group.iloc[0]['company_name']

        round_lines = []
        for _, row in group.iterrows():
            investors = row['participating_investors'] \
                if pd.notna(row['participating_investors']) else "—"
            notes     = f" Note: {row['notes']}" \
                if pd.notna(row['notes']) else ""
            round_lines.append(
                f"{row['round_type']} ({row['round_date']}): "
                f"€{row['amount_eur_m']}M — Lead: {row['lead_investor']}, "
                f"Others: {investors}. "
                f"Post-money: €{row['post_money_valuation_eur_m']}M.{notes}"
            )

        text = (
            f"Funding history for {company} ({company_id}):\n\n"
            + "\n".join(round_lines)
        )

        chunks.append(Chunk(
            text=text,
            metadata=ChunkMetadata(
                doc_type    = DOC_TYPE_FUNDING,
                source_file = "funding_rounds.csv",
                company     = company,
                company_id  = company_id,
            )
        ))

    return chunks


# ════════════════════════════════════════════════════════
# MASTER CHUNKER
# ════════════════════════════════════════════════════════

def chunk_all(data: dict) -> list[Chunk]:
    """
    Receives the output of loader.load_all() and returns
    a flat list of Chunk objects ready for embeddings.py.
    """
    chunks = []

    print("Chunking factsheets...")
    fs_chunks = chunk_all_factsheets(data['factsheets'])
    chunks.extend(fs_chunks)
    print(f"  ✓ {len(fs_chunks)} chunks from {len(data['factsheets'])} factsheets")

    print("Chunking news...")
    news_chunks = chunk_all_news(data['news'])
    chunks.extend(news_chunks)
    print(f"  ✓ {len(news_chunks)} chunks from news digests")

    print("Chunking reports...")
    report_chunks = chunk_all_reports(data['reports'])
    chunks.extend(report_chunks)
    print(f"  ✓ {len(report_chunks)} chunks from reports")

    print("Chunking financials CSV...")
    fin_chunks = chunk_financials(data['financials_df'])
    chunks.extend(fin_chunks)
    print(f"  ✓ {len(fin_chunks)} chunks from financials")

    print("Chunking funding CSV...")
    fund_chunks = chunk_funding(data['funding_df'])
    chunks.extend(fund_chunks)
    print(f"  ✓ {len(fund_chunks)} chunks from funding rounds")

    print(f"\nTotal chunks ready for embedding: {len(chunks)}")
    return chunks