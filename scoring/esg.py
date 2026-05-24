"""
scoring/esg.py
ESG dimension score (0-25).
RAG extracts evidence → Python applies ESG framework rules.

Framework source: reports/esg_scoring_framework_agriculture_v2_1.txt
Three pillars (E, S, G) each scored 0-100.
Composite = (E + S + G) / 3 → dimension = composite / 4 → 0-25.

Red flags applied from framework:
  Unverified ESG claims        → Environmental: -20
  CFO vacancy                  → Governance:    -10
  Revenue declining            → Social:        -10, Governance: -10
  Leadership departures > 1    → Governance:    -10
  Customer concentration > 50% → Governance:    -10
"""

import json
import re
from rag.qa_engine import QAEngine, GROQ_MODEL

EXTRACTION_PROMPT = """\
Analyze the company context below and extract ESG evidence.
Return ONLY a valid JSON object — no explanation, no markdown.

{{
  "environmental": {{
    "water_reduction_pct": <number or null>,
    "ghg_reduction_pct": <number or null>,
    "pesticide_reduction_pct": <number or null>,
    "certifications": ["<cert>"],
    "verified_claims": <true or false>
  }},
  "social": {{
    "farmer_impact": "<high|medium|low|none>",
    "food_security_contribution": <true or false>,
    "sdg_alignment": ["SDG X"],
    "community_engagement": "<high|medium|low|none>"
  }},
  "governance": {{
    "cfo_vacancy": <true or false>,
    "leadership_departures_12mo": <integer>,
    "revenue_declining": <true or false>,
    "customer_concentration_risk": <true or false>,
    "reporting_quality": "<high|medium|low>"
  }}
}}

Context:
{context}

Return ONLY the JSON object."""


# ── Helpers ───────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Robustly extracts JSON from LLM response."""
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


def _safe_num(val, default: float = 0.0) -> float:
    """
    Safely converts LLM-extracted values to float.
    Handles null, 'N/A', 'none', 'unknown', and other non-numeric strings.
    """
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """Safe integer cast for LLM-extracted values."""
    return int(_safe_num(val, default))


def _safe_bool(val, default: bool = False) -> bool:
    """
    Safe boolean cast. Handles True/False, 'yes'/'no', 'true'/'false', 1/0.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ('true', 'yes', '1')
    return default


def _safe_str(val, default: str = 'none') -> str:
    """Returns lowercased string or default if None/non-string."""
    if val is None:
        return default
    return str(val).strip().lower()


# ── Pillar scorers ────────────────────────────────────────────

def _score_environmental(env: dict) -> tuple[float, list]:
    """Scores Environmental pillar 0-100 with red flags."""
    score = 50.0
    flags = []

    water     = _safe_num(env.get('water_reduction_pct'))
    ghg       = _safe_num(env.get('ghg_reduction_pct'))
    pesticide = _safe_num(env.get('pesticide_reduction_pct'))

    if water >= 30:      score += 15
    elif water >= 15:    score += 8
    elif water > 0:      score += 4

    if ghg >= 20:        score += 10
    elif ghg > 0:        score += 5

    if pesticide >= 30:  score += 10
    elif pesticide > 0:  score += 5

    certs = env.get('certifications') or []
    if isinstance(certs, list):
        score += min(len(certs) * 5, 15)

    # Red flag: unverified claims
    if not _safe_bool(env.get('verified_claims'), default=True):
        score -= 20
        flags.append("Unverified ESG claims: -20 (Environmental)")

    return round(min(max(score, 0), 100), 1), flags


def _score_social(soc: dict) -> tuple[float, list]:
    """Scores Social pillar 0-100."""
    score = 50.0
    flags = []

    impact_map = {'high': 15, 'medium': 8, 'low': 3, 'none': 0}
    farmer_impact = _safe_str(soc.get('farmer_impact'), default='none')
    score += impact_map.get(farmer_impact, 0)

    if _safe_bool(soc.get('food_security_contribution')):
        score += 10

    sdgs = soc.get('sdg_alignment') or []
    if isinstance(sdgs, list):
        score += min(len(sdgs) * 3, 12)

    community_map = {'high': 10, 'medium': 5, 'low': 2, 'none': 0}
    community = _safe_str(soc.get('community_engagement'), default='none')
    score += community_map.get(community, 0)

    # Red flag: revenue decline
    if _safe_bool(soc.get('revenue_declining')):
        score -= 10
        flags.append("Revenue declining: -10 (Social)")

    return round(min(max(score, 0), 100), 1), flags


def _score_governance(gov: dict) -> tuple[float, list]:
    """Scores Governance pillar 0-100 with red flags from ESG framework."""
    score = 60.0
    flags = []

    if _safe_bool(gov.get('cfo_vacancy')):
        score -= 10
        flags.append("CFO vacancy: -10 (Governance)")

    departures = _safe_int(gov.get('leadership_departures_12mo', 0))
    if departures > 1:
        score -= 10
        flags.append(f"{departures} leadership departures in 12mo: -10 (Governance)")

    if _safe_bool(gov.get('revenue_declining')):
        score -= 10
        flags.append("Revenue declining: -10 (Governance)")

    if _safe_bool(gov.get('customer_concentration_risk')):
        score -= 10
        flags.append("Customer concentration >50%: -10 (Governance)")

    reporting_map = {'high': 15, 'medium': 5, 'low': 0}
    reporting = _safe_str(gov.get('reporting_quality'), default='medium')
    score += reporting_map.get(reporting, 5)

    return round(min(max(score, 0), 100), 1), flags


# ── Main scorer ───────────────────────────────────────────────

def score(company_name: str, qa_engine: QAEngine) -> dict:
    """
    Computes ESG score for one company.

    Returns:
        {
            'score'    : float (0-25),
            'e_score'  : float (0-100),
            's_score'  : float (0-100),
            'g_score'  : float (0-100),
            'composite': float (0-100),
            'flags'    : list[str],
            'breakdown': dict,
            'extracted': dict,
        }
    """
    # 1. Retrieve ESG context
    context, _ = qa_engine.retriever.retrieve(
        query = (
            f"{company_name} ESG sustainability environmental water emissions "
            "certifications governance leadership social farmer impact SDG"
        ),
        k       = 6,
        company = company_name,
    )

    # 2. Single LLM extraction call
    response = qa_engine.client.chat.completions.create(
        model      = GROQ_MODEL,
        max_tokens = 600,
        messages   = [{"role": "user", "content": EXTRACTION_PROMPT.format(context=context)}],
    )
    extracted = _parse_json(response.choices[0].message.content)

    # 3. Python scoring per pillar
    e_score, e_flags = _score_environmental(extracted.get('environmental', {}))
    s_score, s_flags = _score_social(extracted.get('social', {}))
    g_score, g_flags = _score_governance(extracted.get('governance', {}))

    all_flags = e_flags + s_flags + g_flags
    composite = round((e_score + s_score + g_score) / 3, 1)
    dimension = round(min(composite / 4, 25.0), 2)

    return {
        'score'    : dimension,
        'e_score'  : e_score,
        's_score'  : s_score,
        'g_score'  : g_score,
        'composite': composite,
        'flags'    : all_flags,
        'breakdown': {
            'environmental (0-100)' : e_score,
            'social (0-100)'        : s_score,
            'governance (0-100)'    : g_score,
            'composite (0-100)'     : composite,
            'dimension (0-25)'      : dimension,
        },
        'extracted': extracted,
    }