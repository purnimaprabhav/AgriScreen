"""
tests/test_validation.py
Runs all validation checks from the checklist programmatically.
"""
import os
import sys

# Add project root to path so imports work from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

from dotenv import load_dotenv
load_dotenv()

# rest of imports stay the same...
from rag.qa_engine import QAEngine
from scoring.composite import run_scoring
from ingestion.loader import load_financials
from monitoring.alerts import run_alerts

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

qa_engine     = QAEngine(index_dir="outputs/index", api_key=os.getenv("GROQ_API_KEY"))
scores_df     = run_scoring(index_dir="outputs/index", api_key=os.getenv("GROQ_API_KEY"))
financials_df = load_financials()
alerts        = run_alerts(scores_df, financials_df, index_dir="outputs/index")

# ══════════════════════════════════════════════════════════════
# 1. RAG VALIDATION
# ══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("1. RAG VALIDATION")
print("="*65)

RAG_QUERIES = [
    {
        "id"      : "Q1",
        "query"   : "Which company has the strongest revenue growth trajectory?",
        "expect"  : ["soilsense", "bioroot", "verdant", "greenyield", "growth", "revenue"],
        "must_not": [],
    },
    {
        "id"      : "Q2",
        "query"   : "What are the main technology risks across the portfolio?",
        "expect"  : ["risk", "technology", "competition"],
        "must_not": [],
    },
    {
        "id"      : "Q3",
        "query"   : "Which companies are actively fundraising and how much are they targeting?",
        "expect"  : ["greenyield", "verdant", "series", "eur"],
        "must_not": [],
    },
    {
        "id"      : "Q4",
        "query"   : "Which company has the strongest ESG profile?",
        "expect"  : ["esg", "environmental", "social", "governance"],
        "must_not": [],
    },
    {
        "id"      : "Q5 (impossible)",
        "query"   : "Which company plans expansion into South America in 2028?",
        "expect"  : ["cannot", "not", "available", "documents"],
        "must_not": ["2028", "south america", "expansion plan"],
    },
]

for q in RAG_QUERIES:
    result = qa_engine.ask(q["query"], k=5)
    answer_lower = result["answer"].lower()

    # Check expected keywords present
    hits     = [kw for kw in q["expect"] if kw.lower() in answer_lower]
    misses   = [kw for kw in q["expect"] if kw.lower() not in answer_lower]

    # Check hallucination guard for impossible question
    bad_hits = [kw for kw in q["must_not"] if kw.lower() in answer_lower]

    sources_ok = len(result["sources"]) > 0

    print(f"\n[{q['id']}] {q['query'][:60]}…")
    print(f"  Sources retrieved : {PASS if sources_ok else FAIL} ({len(result['sources'])} chunks)")
    print(f"  Expected keywords : {PASS if not misses else WARN} hits={hits} misses={misses}")

    if q["must_not"]:
        if bad_hits:
            print(f"  Hallucination check: {FAIL} Found forbidden content: {bad_hits}")
        else:
            print(f"  Hallucination guard : {PASS} No hallucinated content")

    print(f"  Answer preview    : {result['answer'][:200].strip()}…")

    # Source inspection
    print(f"  Top sources:")
    for src in result["sources"][:3]:
        meta = src["chunk"].metadata
        print(
            f"    [{src['rank']}] {meta.source_file} "
            f"| {meta.doc_type} "
            f"| company={meta.company} "
            f"| sim={src['score']:.3f}"
        )


# ══════════════════════════════════════════════════════════════
# 2. SCORING VALIDATION
# ══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("2. SCORING VALIDATION")
print("="*65)

# A. F + T + M + E = Total
print("\nA. Sub-score totals match composite:")
all_match = True
for _, row in scores_df.iterrows():
    computed = round(row['financial'] + row['technology'] + row['market'] + row['esg'], 2)
    stored   = round(row['total'], 2)
    match    = abs(computed - stored) < 0.1
    if not match:
        all_match = False
    status = PASS if match else FAIL
    print(f"  {status} {row['company']:<35} F+T+M+E={computed:.1f}  stored={stored:.1f}")

# B. Rankings make sense
print("\nB. Rankings make logical sense:")
last = scores_df.iloc[-1]
first = scores_df.iloc[0]
print(f"  {PASS if 'harvest' in last['company'].lower() else FAIL} "
      f"HarvestLink is lowest  → {last['company']} ({last['total']:.1f})")
print(f"  {PASS if first['total'] >= 70 else FAIL} "
      f"Top company is PRIORITY → {first['company']} ({first['total']:.1f})")

# C. HarvestLink financial score < 10
hl = scores_df[scores_df['company'].str.contains('Harvest')].iloc[0]
print(f"  {PASS if hl['financial'] < 10 else FAIL} "
      f"HarvestLink F score < 10 → F={hl['financial']:.1f} (declining revenue)")

# D. AquaGrow profitable → runway=999 → max runway score
ag = scores_df[scores_df['company'].str.contains('AquaGrow')].iloc[0]
f_breakdown = ag.get('f_breakdown', {})
runway_pts  = f_breakdown.get('runway (0-5)', 0)
print(f"  {PASS if runway_pts == 5.0 else FAIL} "
      f"AquaGrow runway = 5.0/5 (profitable) → got {runway_pts}")

# E. Score explanation sample
print("\nC. Score explanation — BioRoot Technology:")
br = scores_df[scores_df['company'].str.contains('BioRoot')].iloc[0]
t_breakdown = br.get('t_breakdown', {})
print(f"  Total T = {br['technology']:.1f}")
for k, v in t_breakdown.items():
    print(f"    {k}: {v}")


# ══════════════════════════════════════════════════════════════
# 3. ALERT VALIDATION
# ══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("3. ALERT VALIDATION")
print("="*65)

EXPECTED_ALERTS = {
    "RUNWAY_CRITICAL" : "HarvestLink GmbH",
    "REVENUE_DECLINE" : "HarvestLink GmbH",
    "GOVERNANCE_FLAG" : "HarvestLink GmbH",
    "STRATEGIC_EXIT"  : "AquaGrow Solutions Ltd",
    "FUNDRAISE_ACTIVE": "GreenYield Technologies BV",
    "ESG_ALERT"       : "HarvestLink GmbH",
    "SCORE_PRIORITY"  : None,  # multiple companies
}

print(f"\nTotal alerts: {len(alerts)}")
alert_types_found = {(a['alert_type'], a['company']) for a in alerts}

for alert_type, expected_company in EXPECTED_ALERTS.items():
    if expected_company:
        found = any(
            a['alert_type'] == alert_type and
            expected_company.lower() in a['company'].lower()
            for a in alerts
        )
        print(f"  {PASS if found else FAIL} {alert_type:<20} → {expected_company}")
    else:
        found = any(a['alert_type'] == alert_type for a in alerts)
        print(f"  {PASS if found else FAIL} {alert_type:<20} → (any company)")

# Check every alert has required fields
print("\nAlert field completeness:")
required_fields = ['company', 'alert_type', 'severity', 'trigger', 'evidence', 'source', 'action']
incomplete = []
for a in alerts:
    missing = [f for f in required_fields if not a.get(f)]
    if missing:
        incomplete.append((a['alert_type'], a['company'], missing))

if incomplete:
    for at, co, missing in incomplete:
        print(f"  {FAIL} [{at}] {co} — missing: {missing}")
else:
    print(f"  {PASS} All {len(alerts)} alerts have complete fields")

# Check alerts are NOT hardcoded (derived from data)
print("\nAlerts derived dynamically:")
runway_alert = next((a for a in alerts if a['alert_type'] == 'RUNWAY_CRITICAL'), None)
if runway_alert:
    has_number = any(c.isdigit() for c in runway_alert['evidence'])
    print(f"  {PASS if has_number else FAIL} RUNWAY_CRITICAL evidence contains numeric value")
    print(f"  Evidence: {runway_alert['evidence']}")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("VALIDATION SUMMARY")
print("="*65)
print("""
Manual checks still required in the UI:
  1. Open each tab — Dashboard / Chat / Alerts / Notes
  2. Run Q1-Q5 in the Chat tab and inspect source expanders
  3. Test Alert filters (severity, company, type dropdowns)
  4. Generate a company note and confirm it's grounded
  5. Confirm Refresh Scores button triggers a re-run
  6. Test an impossible question in Chat

If all automated checks passed and manual checks look good:
  ✅ Submission is ready.
""")