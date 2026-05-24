# test_scoring.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from scoring.composite import run_scoring

results = run_scoring(
    index_dir = "outputs/index",
    api_key   = os.getenv("GROQ_API_KEY"),
)


# Print detailed breakdown — use exact name from results dataframe
print("\nCompanies in results:", results['company'].tolist())

# Pick first row (highest scorer)
row = results.iloc[0]
print(f"\nDetailed breakdown — {row['company']}")
print("Financial :", row['f_breakdown'])
print("Technology:", row['t_breakdown'])
print("Market    :", row['m_breakdown'])
print("ESG       :", row['e_breakdown'])
print("ESG flags :", row['esg_flags'])
