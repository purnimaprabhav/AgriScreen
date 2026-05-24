"""
scoring/market.py
Market dimension score (0-25).
Hybrid: structured signals + LLM extraction → Python scoring.

Sub-components:
  market_traction_ha  : 0-7  (log-scaled coverage_ha)
  investor_conviction : 0-8  (tiered by total_raised_eur_m)
  tam_strength        : 0-3  (LLM extracted)
  geographic_presence : 0-3  (LLM extracted)
  competitive_position: 0-2  (LLM extracted)
  partnership_strength: 0-2  (LLM extracted)
"""

import json
import re
import math
from rag.qa_engine import QAEngine, GROQ_MODEL

MAX_COVERAGE_HA = 890_000   # AquaGrow — highest in portfolio
TRACTION_MAX    = 7
CONVICTION_MAX  = 8
TAM_MAX         = 3
GEO_MAX         = 3
COMPETITION_MAX = 2
PARTNERSHIP_MAX = 2

EXTRACTION_PROMPT = """\
Analyze the company context below and extract market signals.
Return ONLY a valid JSON object — no explanation, no markdown.

{{
  "tam_strength": <0-10>,
  "geographic_presence": <0-10>,
  "competitive_position": <0-10>,
  "partnership_strength": <0-10>,
  "key_market_insights": ["<string>", "<string>"]
}}

Scoring guide:
- tam_strength: size and growth rate of addressable market (0=niche/declining, 10=large fast-growing)
- geographic_presence: countries and strategic importance (0=single country, 10=multi-continent)
- competitive_position: differentiation vs competitors (0=undifferentiated, 10=clear moat/leader)
- partnership_strength: quality of strategic partnerships (0=none, 10=Tier-1 partners providing distribution)

Context:
{context}

Return ONLY the JSON object."""


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _traction_score(coverage_ha: float) -> float:
    """Log-scaled. 0 ha = 0 pts, 890,000 ha = full 7 pts."""
    if coverage_ha <= 0:
        return 0.0
    return round(
        (math.log10(coverage_ha + 1) / math.log10(MAX_COVERAGE_HA + 1)) * TRACTION_MAX, 2
    )


def _conviction_score(raised_eur_m: float) -> float:
    """Tiered by total equity raised."""
    if raised_eur_m >= 20: return float(CONVICTION_MAX)
    if raised_eur_m >= 15: return 6.0
    if raised_eur_m >= 10: return 5.0
    if raised_eur_m >= 5:  return 3.0
    return 1.0


def _score_signals(signals: dict) -> tuple[float, dict]:
    tam_pts   = round(min(signals.get('tam_strength', 0) / 10, 1.0) * TAM_MAX, 2)
    geo_pts   = round(min(signals.get('geographic_presence', 0) / 10, 1.0) * GEO_MAX, 2)
    comp_pts  = round(min(signals.get('competitive_position', 0) / 10, 1.0) * COMPETITION_MAX, 2)
    part_pts  = round(min(signals.get('partnership_strength', 0) / 10, 1.0) * PARTNERSHIP_MAX, 2)

    breakdown = {
        'tam_strength (0-3)'          : tam_pts,
        'geographic_presence (0-3)'   : geo_pts,
        'competitive_position (0-2)'  : comp_pts,
        'partnership_strength (0-2)'  : part_pts,
    }
    return tam_pts + geo_pts + comp_pts + part_pts, breakdown


def score(
    company_name       : str,
    market_coverage_ha : float,
    total_raised_eur_m : float,
    qa_engine          : QAEngine,
) -> dict:
    """
    Computes Market score for one company.

    Returns:
        {
            'score'            : float (0-25),
            'breakdown'        : dict,
            'extracted_signals': dict,
            'inputs_used'      : dict,
        }
    """
    # 1. Structured signals
    traction_pts   = _traction_score(market_coverage_ha)
    conviction_pts = _conviction_score(total_raised_eur_m)

    # 2. Retrieve context
    context, _ = qa_engine.retriever.retrieve(
        query   = f"{company_name} market TAM partnerships geographic expansion competition position",
        k       = 5,
        company = company_name,
    )

    # 3. Single LLM extraction call
    response = qa_engine.client.chat.completions.create(
        model      = GROQ_MODEL,
        max_tokens = 512,
        messages   = [{"role": "user", "content": EXTRACTION_PROMPT.format(context=context)}],
    )
    signals = _parse_json(response.choices[0].message.content)

    # 4. Python scoring
    llm_pts, llm_breakdown = _score_signals(signals)
    total = round(min(traction_pts + conviction_pts + llm_pts, 25.0), 2)

    breakdown = {
        'market_traction_ha (0-7)'   : traction_pts,
        'investor_conviction (0-8)'  : conviction_pts,
    }
    breakdown.update(llm_breakdown)

    return {
        'score'            : total,
        'breakdown'        : breakdown,
        'extracted_signals': signals,
        'inputs_used': {
            'market_coverage_ha' : market_coverage_ha,
            'total_raised_eur_m' : total_raised_eur_m,
        },
    }