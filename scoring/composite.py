"""
scoring/composite.py
Orchestrates all four scoring dimensions → final composite score per company.

Usage:
    from scoring.composite import run_scoring
    results_df = run_scoring()
"""

import os
import pandas as pd
from rag.qa_engine import QAEngine
from ingestion.loader import load_all_factsheets, load_financials
from scoring import financial, technology, market, esg

PRIORITY_THRESHOLD = 70
WATCH_THRESHOLD    = 50


def _flag(total: float) -> str:
    if total >= PRIORITY_THRESHOLD: return "PRIORITY"
    if total >= WATCH_THRESHOLD:    return "WATCH"
    return "LOW PRIORITY"


def load_scoring_inputs() -> dict:
    """Returns {company_name: scoring_inputs_dict} from all factsheets."""
    return {
        fs['company_name']: fs['scoring_inputs']
        for fs in load_all_factsheets()
    }


def run_scoring(
    index_dir: str        = "outputs/index",
    api_key  : str | None = None,
) -> pd.DataFrame:
    """
    Runs the full scoring pipeline for all companies.

    Returns a DataFrame sorted by total score (descending) with columns:
        company, financial, technology, market, esg,
        total, flag, + full breakdown columns
    """
    print("Initialising scoring pipeline...")

    scoring_inputs = load_scoring_inputs()
    financials_df  = load_financials()
    qa_engine      = QAEngine(index_dir=index_dir, api_key=api_key)

    results = []
    print(f"\nScoring {len(scoring_inputs)} companies...\n")

    for company_name, inputs in scoring_inputs.items():
        print(f"  → {company_name}")

        f = financial.score(company_name, financials_df)
        t = technology.score(
            company_name = company_name,
            has_patent   = inputs.get('has_patent', False),
            qa_engine    = qa_engine,
        )
        m = market.score(
            company_name       = company_name,
            market_coverage_ha = inputs.get('market_coverage_ha', 0),
            total_raised_eur_m = inputs.get('total_raised_eur_m', 0),
            qa_engine          = qa_engine,
        )
        e = esg.score(company_name, qa_engine)

        total = round(f['score'] + t['score'] + m['score'] + e['score'], 2)
        flag  = _flag(total)

        print(
            f"     F={f['score']:.1f}  T={t['score']:.1f}  "
            f"M={m['score']:.1f}  E={e['score']:.1f}  "
            f"→ Total={total:.1f}  [{flag}]"
        )

        results.append({
            'company'     : company_name,
            'financial'   : f['score'],
            'technology'  : t['score'],
            'market'      : m['score'],
            'esg'         : e['score'],
            'total'       : total,
            'flag'        : flag,
            'f_breakdown' : f['breakdown'],
            't_breakdown' : t['breakdown'],
            'm_breakdown' : m['breakdown'],
            'e_breakdown' : e['breakdown'],
            'esg_flags'   : e['flags'],
            't_signals'   : t.get('extracted_signals', {}),
            'm_signals'   : m.get('extracted_signals', {}),
            'e_extracted' : e.get('extracted', {}),
        })

    df = (
        pd.DataFrame(results)
        .sort_values('total', ascending=False)
        .reset_index(drop=True)
    )
    df.index = df.index + 1

    print("\n" + "=" * 65)
    print("FINAL SCORES")
    print("=" * 65)
    print(df[['company', 'financial', 'technology', 'market', 'esg', 'total', 'flag']].to_string())

    return df