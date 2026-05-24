"""
monitoring/company_notes.py
Generates structured investment notes per company using RAG + LLM.
"""

import os
import pandas as pd
from rag.qa_engine import QAEngine, GROQ_MODEL
from monitoring.alerts import run_alerts
from ingestion.loader import load_financials
from utils.company_meta import COMPANY_INFO


ACTION_MAP = {
    'PRIORITY'    : 'ADVANCE TO DUE DILIGENCE',
    'WATCH'       : 'MONITOR — NEXT CONTACT IN 90 DAYS',
    'LOW PRIORITY': 'DEPRIORITISE — REVIEW IN 6 MONTHS',
}

SUGGESTED_ACTIONS = {
    'PRIORITY': [
        "Schedule a first call with management within 2 weeks",
        "Request data room access and detailed financials",
        "Brief the investment committee on the opportunity",
    ],
    'WATCH': [
        "Set up news monitoring alerts for catalyst events",
        "Schedule a 90-day check-in call",
        "Track key milestones (regulatory, fundraising, partnerships)",
    ],
    'LOW PRIORITY': [
        "Add to passive watchlist",
        "Re-evaluate in 6 months unless major catalyst emerges",
        "Skip outreach unless analyst flag changes",
    ],
}

NOTE_PROMPT = """\
You are a senior investment analyst at Pivot & Co, an agricultural impact fund.
Write a clear, professional investment note for {company_name}.

Use ONLY the facts in the CONTEXT section. Do not invent information.
Write in plain English an analyst could read in 60 seconds.
Use specific numbers and dates wherever possible.

COMPANY OVERVIEW:
- Company: {company_name}
- Country: {country_name}
- Sub-sector: {sub_sector}
- Founded: {founded}

SCORES:
{score_block}

ACTIVE ALERTS:
{alerts_block}

CONTEXT:
{context}

Output the note in EXACTLY this markdown format. Do not add commentary outside the structure:

## {company_name}
**{country_name}** · {sub_sector} · Founded {founded} · Score: **{total}/100** · {flag}

### Investment Thesis
[2-3 sentences explaining what the company does, why it's interesting (or not), and the key insight an analyst needs to know. Cite specific numbers.]

### Key Positive Signals
1. [Specific signal — must include a number or named entity from the context]
2. [Specific signal — must include a number or named entity from the context]
3. [Specific signal — must include a number or named entity from the context]

### Key Risks
1. [Specific risk — must include a number or named entity from the context]
2. [Specific risk — must include a number or named entity from the context]
3. [Specific risk — must include a number or named entity from the context]

### Active Alerts
{alerts_block}

### Recommended Action
**{action}** — [One sentence explaining why, referencing the specific scores or alerts above.]

### Suggested Next Steps
{suggested_steps}
"""


def generate_note(company_name, score_row, alerts, qa_engine):
    """Generates a structured investment note for one company."""
    context, _ = qa_engine.retriever.retrieve(
        query = (
            f"{company_name} investment technology market position "
            "ESG financial risks strategic partnerships customers"
        ),
        k=7, company=company_name,
    )

    score_block = (
        f"Financial: {score_row['financial']:.1f}/25  |  "
        f"Technology: {score_row['technology']:.1f}/25  |  "
        f"Market: {score_row['market']:.1f}/25  |  "
        f"ESG: {score_row['esg']:.1f}/25  |  "
        f"Total: {score_row['total']:.1f}/100  |  Flag: {score_row['flag']}"
    )

    co_alerts = [a for a in alerts if a['company'] == company_name]
    alerts_block = (
        "\n".join(f"- [{a['severity']}] {a['alert_type']}: {a['trigger']}" for a in co_alerts)
        if co_alerts else "No active alerts."
    )

    meta   = COMPANY_INFO.get(company_name, {})
    flag   = score_row['flag']
    action = ACTION_MAP.get(flag, 'MONITOR')
    suggested = SUGGESTED_ACTIONS.get(flag, [])
    suggested_steps = "\n".join(f"- {s}" for s in suggested)

    prompt = NOTE_PROMPT.format(
        company_name    = company_name,
        country_name    = meta.get('country_name', 'N/A'),
        sub_sector      = meta.get('sub_sector', 'AgTech'),
        founded         = meta.get('founded', 'N/A'),
        score_block     = score_block,
        alerts_block    = alerts_block,
        context         = context,
        total           = f"{score_row['total']:.1f}",
        flag            = flag,
        action          = action,
        suggested_steps = suggested_steps,
    )

    response = qa_engine.client.chat.completions.create(
        model       = GROQ_MODEL,
        max_tokens  = 1100,
        temperature = 0,
        messages    = [{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_all_notes(scores_df, qa_engine,
                        index_dir="outputs/index", output_dir="outputs/notes"):
    """Generates and saves investment notes for all companies."""
    os.makedirs(output_dir, exist_ok=True)
    financials_df = load_financials()
    alerts        = run_alerts(scores_df, financials_df, index_dir)

    notes = {}
    print(f"\nGenerating notes for {len(scores_df)} companies...\n")
    for _, row in scores_df.iterrows():
        company = row['company']
        print(f"  → {company}")
        note = generate_note(company, row, alerts, qa_engine)
        notes[company] = note
        safe = company.replace(' ', '_').replace('/', '_')
        with open(os.path.join(output_dir, f"{safe}.md"), 'w') as f:
            f.write(note)
        print(f"     ✓ Saved → {safe}.md")
    return notes