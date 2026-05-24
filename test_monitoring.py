import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from scoring.composite import run_scoring
from ingestion.loader import load_financials
from monitoring.alerts import run_alerts, print_alerts
from monitoring.company_notes import generate_all_notes
from rag.qa_engine import QAEngine

# 1. Run scoring
scores_df     = run_scoring(index_dir="outputs/index", api_key=os.getenv("GROQ_API_KEY"))
financials_df = load_financials()

# 2. Run alerts
alerts = run_alerts(scores_df, financials_df, index_dir="outputs/index")
print_alerts(alerts)

# 3. Generate notes for top 2 companies only (saves time)
qa_engine = QAEngine(index_dir="outputs/index", api_key=os.getenv("GROQ_API_KEY"))
top2      = scores_df.head(2)
notes     = generate_all_notes(top2, qa_engine, index_dir="outputs/index")

# 4. Print one note
first_company = scores_df.iloc[0]['company']
print(f"\n{'='*65}")
print(f"INVESTMENT NOTE — {first_company}")
print('='*65)
print(notes[first_company])