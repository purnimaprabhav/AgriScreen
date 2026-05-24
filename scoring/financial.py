"""
scoring/financial.py
Financial dimension score (0-25).
Fully deterministic — CSV data only, no LLM.

Sub-components:
  Revenue CAGR 3y   : 0-10
  Gross margin      : 0-7
  Runway            : 0-5
  Burn efficiency   : 0-3
"""

import pandas as pd

CAGR_MAX   = 10
MARGIN_MAX = 7
RUNWAY_MAX = 5
BURN_MAX   = 3



def _revenue_cagr(df_company: pd.DataFrame) -> float:
    """3-year CAGR from FY2023 to FY2025."""
    try:
        rev_2023 = df_company[df_company['fy_year'] == 2023]['revenue_eur_k'].values[0]
        rev_2025 = df_company[df_company['fy_year'] == 2025]['revenue_eur_k'].values[0]
        if rev_2023 <= 0:
            return 0.0
        return round((rev_2025 / rev_2023) ** 0.5 - 1, 4)
    except (IndexError, ZeroDivisionError):
        return 0.0



def _cagr_score(cagr: float) -> float:
    """50%+ CAGR = full 10 pts."""
    return round(min(max(cagr, 0) / 0.5, 1.0) * CAGR_MAX, 2)


def _margin_score(margin_pct: float) -> float:
    """80%+ gross margin = full 7 pts."""
    return round(min(margin_pct / 80.0, 1.0) * MARGIN_MAX, 2)


def _runway_score(runway_months: int) -> float:
    """Tiered runway score 0-5. 999 = profitable sentinel."""
    if runway_months >= 999: return 5.0
    if runway_months >= 36:  return 4.0
    if runway_months >= 24:  return 3.0
    if runway_months >= 18:  return 2.0
    if runway_months >= 12:  return 1.0
    return 0.0


def _burn_efficiency_score(ebitda_k: float, revenue_k: float) -> float:
    """EBITDA margin tiers: positive→3, >-20%→2, >-40%→1, else→0."""
    if revenue_k <= 0:
        return 0.0
    margin = ebitda_k / revenue_k
    if margin > 0:     return 3.0
    if margin > -0.20: return 2.0
    if margin > -0.40: return 1.0
    return 0.0

def score(company_name: str, financials_df: pd.DataFrame) -> dict:
    # Case-insensitive match — factsheet names are uppercase, CSV is mixed case
    df = financials_df[
        financials_df['company_name'].str.lower() == company_name.lower()
    ].copy()

    if df.empty:
        return {
            'score': 0.0, 'breakdown': {},
            'inputs_used': {}, 'error': f'{company_name} not found in CSV'
        }

    fy2025  = df[df['fy_year'] == 2025].iloc[0]
    cagr    = _revenue_cagr(df)
    margin  = float(fy2025['gross_margin_pct'])
    runway  = int(fy2025['runway_months'])
    ebitda  = float(fy2025['ebitda_eur_k'])
    revenue = float(fy2025['revenue_eur_k'])

    cagr_pts   = _cagr_score(cagr)
    margin_pts = _margin_score(margin)
    runway_pts = _runway_score(runway)
    burn_pts   = _burn_efficiency_score(ebitda, revenue)
    total      = round(min(cagr_pts + margin_pts + runway_pts + burn_pts, 25.0), 2)

    return {
        'score': total,
        'breakdown': {
            'revenue_cagr_3y (0-10)' : cagr_pts,
            'gross_margin (0-7)'     : margin_pts,
            'runway (0-5)'           : runway_pts,
            'burn_efficiency (0-3)'  : burn_pts,
        },
        'inputs_used': {
            'revenue_cagr_3y_pct' : round(cagr * 100, 1),
            'gross_margin_pct'    : margin,
            'runway_months'       : runway,
            'ebitda_eur_k'        : ebitda,
            'revenue_eur_k'       : revenue,
            'ebitda_margin_pct'   : round(ebitda / revenue * 100, 1),
        },
    }