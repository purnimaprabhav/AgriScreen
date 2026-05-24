"""
monitoring/alerts.py
Rule-based alert engine for portfolio monitoring.

Seven alert types evaluated against:
  - Structured data  : CSV financials, scoring results
  - Unstructured data: news + factsheet chunks via FAISS retrieval

Every alert is backed by a RAG-retrieved source quote.
No LLM calls — all rules are deterministic.
"""

import pandas as pd
from ingestion.vector_store import load_index, search
from ingestion.embeddings import load_model

# ── Alert definitions ─────────────────────────────────────────
ALERT_DEFINITIONS = {
    'RUNWAY_CRITICAL': {
        'severity'   : 'HIGH',
        'description': 'Runway below 12 months',
        'action'     : 'Immediate flag — fundraising or exit urgency',
    },
    'REVENUE_DECLINE': {
        'severity'   : 'HIGH',
        'description': 'Revenue declined year-over-year',
        'action'     : 'Investigate cause — structural or temporary?',
    },
    'ESG_ALERT': {
        'severity'   : 'MEDIUM',
        'description': 'ESG composite score below 60/100',
        'action'     : 'Enhanced ESG due diligence required per framework',
    },
    'GOVERNANCE_FLAG': {
        'severity'   : 'HIGH',
        'description': 'Senior leadership departure or governance risk detected',
        'action'     : 'Governance review — assess management continuity',
    },
    'FUNDRAISE_ACTIVE': {
        'severity'   : 'LOW',
        'description': 'Active fundraising process detected in news',
        'action'     : 'Priority contact — engage before round closes',
    },
    'STRATEGIC_EXIT': {
        'severity'   : 'HIGH',
        'description': 'M&A or strategic sale process detected in news',
        'action'     : 'Time-sensitive — escalate to senior team immediately',
    },
    'SCORE_PRIORITY': {
        'severity'   : 'LOW',
        'description': 'Composite score reached PRIORITY threshold (≥70)',
        'action'     : 'Advance to formal due diligence',
    },
}

# ── News signal keywords ──────────────────────────────────────
GOVERNANCE_KEYWORDS = [
    'resigned', 'resignation', 'cfo vacancy', 'departed',
    'departure', 'headcount reduction', 'layoff',
]
FUNDRAISE_KEYWORDS = [
    'series b', 'series a', 'series c', 'fundraising process',
    'targeting eur', 'targeting gbp', 'launch its', 'planned for q',
]
EXIT_KEYWORDS = [
    'acquisition', 'acquirer', 'strategic sale', 'm&a',
    'expressions of interest', 'strategic options',
    'mandate', 'explore strategic',
]

SEVERITY_ORDER = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}


# ── RAG enrichment ────────────────────────────────────────────

def _enrich_with_corpus(
    company_name     : str,
    enrichment_query : str,
    index,
    chunks           : list,
    model,
    doc_type         : str = 'factsheet',
    k                : int = 2,
    min_score        : float = 0.30,
) -> dict | None:
    """
    Queries the FAISS index for narrative evidence to back a structured alert.
    Returns dict with excerpt + source filename, or None if no good match.
    """
    results = search(
        query    = enrichment_query,
        index    = index,
        chunks   = chunks,
        model    = model,
        k        = k,
        company  = company_name,
        doc_type = doc_type,
    )

    if not results or results[0]['score'] < min_score:
        return None

    top = results[0]
    excerpt = top['chunk'].text.strip()

    # Trim to ~250 chars, ending at a sentence boundary if possible
    if len(excerpt) > 250:
        excerpt = excerpt[:250]
        last_break = max(excerpt.rfind('.'), excerpt.rfind('\n'))
        if last_break > 100:
            excerpt = excerpt[:last_break + 1]

    return {
        'excerpt'    : excerpt.strip(),
        'source_file': top['chunk'].metadata.source_file,
        'section'    : top['chunk'].metadata.section or '',
        'score'      : top['score'],
    }


# ── News signal detection ─────────────────────────────────────

def _keyword_search(
    company_name : str,
    keywords     : list[str],
    index,
    chunks       : list,
    model,
) -> tuple[bool, str, str]:
    """
    Searches indexed news chunks for keyword signals for a company.
    Returns (found, evidence, source_file).
    """
    results = search(
        query    = f"{company_name} " + " ".join(keywords[:3]),
        index    = index,
        chunks   = chunks,
        model    = model,
        k        = 6,
        company  = company_name,
        doc_type = 'news',
    )

    for r in results:
        text_lower = r['chunk'].text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                headline = r['chunk'].text.split('\n')[0][:200]
                return True, headline, r['chunk'].metadata.source_file

    return False, '', ''


# ── Alert builder ─────────────────────────────────────────────

def _make_alert(
    company    : str,
    alert_type : str,
    trigger    : str,
    evidence   : str,
    source     : str,
) -> dict:
    meta = ALERT_DEFINITIONS[alert_type]
    return {
        'company'    : company,
        'alert_type' : alert_type,
        'severity'   : meta['severity'],
        'description': meta['description'],
        'trigger'    : trigger,
        'evidence'   : evidence,
        'source'     : source,
        'action'     : meta['action'],
    }


# ── Master alert runner ───────────────────────────────────────

def run_alerts(
    scores_df     : pd.DataFrame,
    financials_df : pd.DataFrame,
    index_dir     : str = "outputs/index",
) -> list[dict]:
    """
    Evaluates all alert rules for every company in scores_df.
    Every alert is backed by a RAG-retrieved corpus quote where available.
    """
    print("Loading index for alert signal detection...")
    model         = load_model()
    index, chunks = load_index(index_dir)
    print(f"  ✓ {len(chunks)} chunks available for retrieval\n")

    alerts = []

    for _, row in scores_df.iterrows():
        company = row['company']
        print(f"  Evaluating alerts → {company}")

        fy2025_df = financials_df[
            financials_df['company_name'].str.lower() == company.lower()
        ]
        if fy2025_df.empty:
            continue

        fy2025    = fy2025_df[fy2025_df['fy_year'] == 2025].iloc[0]
        fy2024_df = fy2025_df[fy2025_df['fy_year'] == 2024]

        # ── RUNWAY_CRITICAL ────────────────────────
        runway = int(fy2025['runway_months'])
        if 0 < runway < 12:
            csv_evidence = (
                f"Monthly burn €{fy2025['burn_eur_k_monthly']}K, "
                f"cash €{fy2025['cash_eur_k']}K → {runway} months runway."
            )
            csv_source = 'companies_financials_2023_2025.csv'

            corpus = _enrich_with_corpus(
                company,
                f"{company} cash runway burn rate financial position",
                index, chunks, model,
            )
            if corpus:
                evidence = (
                    f"{csv_evidence}\n\n"
                    f'Narrative context (from {corpus["source_file"]}): '
                    f'"{corpus["excerpt"]}"'
                )
                source = f"{csv_source} + {corpus['source_file']}"
            else:
                evidence = csv_evidence
                source   = csv_source

            alerts.append(_make_alert(
                company    = company,
                alert_type = 'RUNWAY_CRITICAL',
                trigger    = f"runway_months = {runway}",
                evidence   = evidence,
                source     = source,
            ))

        # ── REVENUE_DECLINE ────────────────────────
        if not fy2024_df.empty:
            rev_2025 = float(fy2025['revenue_eur_k'])
            rev_2024 = float(fy2024_df.iloc[0]['revenue_eur_k'])
            if rev_2025 < rev_2024:
                pct = round((rev_2025 - rev_2024) / rev_2024 * 100, 1)
                csv_evidence = (
                    f"Revenue fell from €{rev_2024:.0f}K (FY2024) "
                    f"to €{rev_2025:.0f}K (FY2025)."
                )
                csv_source = 'companies_financials_2023_2025.csv'

                corpus = _enrich_with_corpus(
                    company,
                    f"{company} revenue customers churn market position",
                    index, chunks, model,
                )
                if corpus:
                    evidence = (
                        f"{csv_evidence}\n\n"
                        f'Narrative context (from {corpus["source_file"]}): '
                        f'"{corpus["excerpt"]}"'
                    )
                    source = f"{csv_source} + {corpus['source_file']}"
                else:
                    evidence = csv_evidence
                    source   = csv_source

                alerts.append(_make_alert(
                    company    = company,
                    alert_type = 'REVENUE_DECLINE',
                    trigger    = f"revenue_growth = {pct}% YoY",
                    evidence   = evidence,
                    source     = source,
                ))

        # ── ESG_ALERT ──────────────────────────────
        e_breakdown   = row.get('e_breakdown', {})
        esg_composite = (
            e_breakdown.get('composite (0-100)', row['esg'] * 4)
            if isinstance(e_breakdown, dict) else row['esg'] * 4
        )
        if esg_composite < 60:
            scoring_evidence = (
                f"ESG composite of {esg_composite:.0f}/100 is below "
                f"the 60/100 minimum required by the ESG framework."
            )
            scoring_source = 'ESG scoring engine'

            corpus = _enrich_with_corpus(
                company,
                f"{company} ESG environmental social governance impact sustainability",
                index, chunks, model,
            )
            if corpus:
                evidence = (
                    f"{scoring_evidence}\n\n"
                    f'ESG narrative (from {corpus["source_file"]}): '
                    f'"{corpus["excerpt"]}"'
                )
                source = f"{scoring_source} + {corpus['source_file']}"
            else:
                evidence = scoring_evidence
                source   = scoring_source

            alerts.append(_make_alert(
                company    = company,
                alert_type = 'ESG_ALERT',
                trigger    = f"ESG composite = {esg_composite:.0f}/100",
                evidence   = evidence,
                source     = source,
            ))

        # ── GOVERNANCE_FLAG ────────────────────────
        gov_found, gov_evidence, gov_source = _keyword_search(
            company, GOVERNANCE_KEYWORDS, index, chunks, model
        )
        if gov_found:
            alerts.append(_make_alert(
                company    = company,
                alert_type = 'GOVERNANCE_FLAG',
                trigger    = 'Leadership departure or governance risk in news',
                evidence   = gov_evidence,
                source     = f'news digest ({gov_source})' if gov_source else 'news digest',
            ))

        # ── FUNDRAISE_ACTIVE ───────────────────────
        fund_found, fund_evidence, fund_source = _keyword_search(
            company, FUNDRAISE_KEYWORDS, index, chunks, model
        )
        if fund_found:
            alerts.append(_make_alert(
                company    = company,
                alert_type = 'FUNDRAISE_ACTIVE',
                trigger    = 'Active fundraising process detected in news',
                evidence   = fund_evidence,
                source     = f'news digest ({fund_source})' if fund_source else 'news digest',
            ))

        # ── STRATEGIC_EXIT ─────────────────────────
        exit_found, exit_evidence, exit_source = _keyword_search(
            company, EXIT_KEYWORDS, index, chunks, model
        )
        if exit_found:
            alerts.append(_make_alert(
                company    = company,
                alert_type = 'STRATEGIC_EXIT',
                trigger    = 'M&A or strategic sale signal in news',
                evidence   = exit_evidence,
                source     = f'news digest ({exit_source})' if exit_source else 'news digest',
            ))

        # ── SCORE_PRIORITY ─────────────────────────
        if row['total'] >= 70:
            scoring_evidence = (
                f"Score of {row['total']:.1f}/100 crossed PRIORITY threshold. "
                f"F={row['financial']:.1f} T={row['technology']:.1f} "
                f"M={row['market']:.1f} E={row['esg']:.1f}"
            )
            scoring_source = 'scoring engine'

            corpus = _enrich_with_corpus(
                company,
                f"{company} strengths growth competitive advantage market position",
                index, chunks, model,
            )
            if corpus:
                evidence = (
                    f"{scoring_evidence}\n\n"
                    f'Supporting narrative (from {corpus["source_file"]}): '
                    f'"{corpus["excerpt"]}"'
                )
                source = f"{scoring_source} + {corpus['source_file']}"
            else:
                evidence = scoring_evidence
                source   = scoring_source

            alerts.append(_make_alert(
                company    = company,
                alert_type = 'SCORE_PRIORITY',
                trigger    = f"total_score = {row['total']:.1f} ≥ 70",
                evidence   = evidence,
                source     = source,
            ))

    # Sort HIGH → MEDIUM → LOW, then alphabetically
    alerts.sort(key=lambda a: (
        SEVERITY_ORDER.get(a['severity'], 3),
        a['company'],
    ))

    return alerts


# ── Pretty printer ────────────────────────────────────────────

def print_alerts(alerts: list[dict]) -> None:
    """Prints alerts grouped by severity."""
    icons = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🔵'}

    print(f"\n{'='*65}")
    print(f"MONITORING ALERTS  ({len(alerts)} total)")
    print(f"{'='*65}")

    current_sev = None
    for a in alerts:
        if a['severity'] != current_sev:
            current_sev = a['severity']
            print(f"\n{icons.get(current_sev, '⚪')}  {current_sev}")
            print('-' * 40)
        print(f"  [{a['alert_type']}]  {a['company']}")
        print(f"  Trigger : {a['trigger']}")
        print(f"  Evidence: {a['evidence'][:150]}")
        print(f"  Source  : {a['source']}")
        print(f"  Action  : {a['action']}")
        print()