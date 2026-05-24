# test_rag.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from rag.qa_engine import QAEngine, print_result

load_dotenv()

engine = QAEngine(
    index_dir = "outputs/index",
    api_key   = os.getenv("GROQ_API_KEY"),
)

# Test 1 — specific company question
print_result(engine.ask(
    "What is AquaGrow's gross margin and when did they become profitable?"
))

# Test 2 — cross-company comparison
print_result(engine.ask(
    "Which companies have an active fundraising process and how much are they targeting?"
))

# Test 3 — out-of-corpus question (should say cannot answer)
print_result(engine.ask(
    "What is the share price of Verdant Farms?"
))