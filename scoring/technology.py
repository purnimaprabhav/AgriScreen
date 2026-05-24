"""
scoring/technology.py
Technology dimension score (0-25).
Hybrid: structured signal (has_patent) + LLM extraction → Python scoring.

Sub-components:
  IP / Patents       : 0-8  (structured — from SCORING INPUTS)
  ip_strength        : 0-5  (LLM extracted)
  data_moat          : 0-4  (LLM extracted)
  integration_maturity: 0-4 (LLM extracted)
  benchmark_evidence : 0-4  (LLM extracted)
"""

import json
import re
from rag.qa_engine import QAEngine, GROQ_MODEL

PATENT_MAX      = 8
IP_MAX          = 5
MOAT_MAX        = 4
INTEGRATION_MAX = 4
BENCHMARK_MAX   = 4

EXTRACTION_PROMPT = """\
Analyze the company context below and extract technology signals.
Return ONLY a valid JSON object — no explanation, no markdown.

{{
  "ip_strength": <0-10>,
  "data_moat": <0-10>,
  "integration_maturity": <0-10>,
  "benchmark_evidence": <0-10>,
  "key_strengths": ["<string>", "<string>"],
  "key_risks": ["<string>", "<string>"]
}}

Scoring guide:
- ip_strength: patents, proprietary algorithms (0=none, 10=multiple granted patents)
- data_moat: unique proprietary datasets competitors cannot replicate (0=none, 10=large unique dataset)
- integration_maturity: third-party platform integrations (0=none, 10=deep integrations with major platforms)
- benchmark_evidence: quantified performance claims and accuracy metrics (0=none, 10=multiple verified benchmarks)

Context:
{context}

Return ONLY the JSON object."""


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


def _score_signals(signals: dict) -> tuple[float, dict]:
    """Maps extracted 0-10 signals to weighted sub-scores."""
    ip_pts    = round(min(signals.get('ip_strength', 0) / 10, 1.0) * IP_MAX, 2)
    moat_pts  = round(min(signals.get('data_moat', 0) / 10, 1.0) * MOAT_MAX, 2)
    integ_pts = round(min(signals.get('integration_maturity', 0) / 10, 1.0) * INTEGRATION_MAX, 2)
    bench_pts = round(min(signals.get('benchmark_evidence', 0) / 10, 1.0) * BENCHMARK_MAX, 2)

    breakdown = {
        'ip_strength (0-5)'           : ip_pts,
        'data_moat (0-4)'             : moat_pts,
        'integration_maturity (0-4)'  : integ_pts,
        'benchmark_evidence (0-4)'    : bench_pts,
    }
    return ip_pts + moat_pts + integ_pts + bench_pts, breakdown


def score(company_name: str, has_patent: bool, qa_engine: QAEngine) -> dict:
    """
    Computes Technology score for one company.

    Returns:
        {
            'score'            : float (0-25),
            'breakdown'        : dict,
            'extracted_signals': dict from LLM,
        }
    """
    # 1. Hard signal
    patent_pts = float(PATENT_MAX) if has_patent else 0.0

    # 2. Retrieve context
    context, _ = qa_engine.retriever.retrieve(
        query   = f"{company_name} technology patents IP integrations accuracy benchmarks proprietary",
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
    total = round(min(patent_pts + llm_pts, 25.0), 2)

    breakdown = {'has_patent (0-8)': patent_pts}
    breakdown.update(llm_breakdown)

    return {
        'score'            : total,
        'breakdown'        : breakdown,
        'extracted_signals': signals,
    }