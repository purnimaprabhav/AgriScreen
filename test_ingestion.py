# test_ingestion.py
# Run from project root: python test_ingestion.py

from ingestion.loader import load_all
from ingestion.chunker import chunk_all
from ingestion.embeddings import load_model
from ingestion.vector_store import build_and_save, search
from collections import Counter

print("=" * 50)
print("STEP 1: Loading files")
print("=" * 50)
data = load_all()

print("\n" + "=" * 50)
print("STEP 2: Chunking")
print("=" * 50)
chunks = chunk_all(data)

print("\nChunks by type:")
for doc_type, count in Counter(c.metadata.doc_type for c in chunks).items():
    print(f"  {doc_type:<15} {count}")

print("\n" + "=" * 50)
print("STEP 3: Embedding + indexing")
print("=" * 50)
model = load_model()
index, chunks = build_and_save(
    chunks,
    output_dir="outputs/index",
    model=model
)

print("\n" + "=" * 50)
print("STEP 4: Test search")
print("=" * 50)
results = search(
    "AquaGrow water saving technology",
    index, chunks, model, k=3
)
for r in results:
    print(f"[{r['rank']}] score={r['score']:.3f} | {r['chunk'].metadata.doc_type} | {r['chunk'].metadata.company}")
    print(f"    {r['chunk'].text[:120]}...")
    print()

