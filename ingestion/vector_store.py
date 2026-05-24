"""
ingestion/vector_store.py

Builds a FAISS index from embedded chunks, and provides
search with optional metadata filtering.

FAISS stores vectors only — chunks (text + metadata) are stored
separately as a pickle file and kept in sync by list index.

Files written to disk:
  {output_dir}/index.faiss   ← FAISS binary index
  {output_dir}/chunks.pkl    ← list[Chunk], aligned with FAISS index
"""

import pickle
import numpy as np
import faiss
from pathlib import Path

from utils.metadata import Chunk
from ingestion.embeddings import embed_query, load_model

INDEX_FILE  = "index.faiss"
CHUNKS_FILE = "chunks.pkl"


# ════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a FAISS flat inner-product index from pre-normalised vectors.
    IndexFlatIP + L2-normalised vectors = exact cosine similarity search.

    Args:
        vectors: shape (n_chunks, embed_dim), float32, L2-normalised

    Returns:
        faiss.IndexFlatIP with all vectors added
    """
    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"  ✓ FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ════════════════════════════════════════════════════════
# SAVE / LOAD
# ════════════════════════════════════════════════════════

def save_index(
    index   : faiss.IndexFlatIP,
    chunks  : list[Chunk],
    output_dir: str | Path,
) -> None:
    """
    Saves the FAISS index and chunk list to disk.
    Both files must always be saved together — they're aligned by position.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / INDEX_FILE))

    with open(output_dir / CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  ✓ Index saved to {output_dir}/")
    print(f"    {INDEX_FILE}: {index.ntotal} vectors")
    print(f"    {CHUNKS_FILE}: {len(chunks)} chunks")


def load_index(output_dir: str | Path) -> tuple[faiss.IndexFlatIP, list[Chunk]]:
    """
    Loads FAISS index and chunk list from disk.

    Returns:
        (index, chunks) — ready to pass directly to search()
    """
    output_dir = Path(output_dir)

    index = faiss.read_index(str(output_dir / INDEX_FILE))

    with open(output_dir / CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    print(f"  ✓ Index loaded: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


# ════════════════════════════════════════════════════════
# SEARCH
# ════════════════════════════════════════════════════════

def search(
    query      : str,
    index      : faiss.IndexFlatIP,
    chunks     : list[Chunk],
    model,
    k          : int = 5,
    doc_type   : str | None = None,
    company    : str | None = None,
) -> list[dict]:
    """
    Retrieves the top-k most relevant chunks for a query.

    Optional filters (applied post-retrieval):
      doc_type  : restrict to "factsheet" | "news" | "report" | "financials" | "funding"
      company   : restrict to a specific company name (case-insensitive)

    Returns:
        list of dicts, each containing:
          'chunk' : Chunk object
          'score' : cosine similarity (float, higher = more relevant)
          'rank'  : 1-based rank
    """
    # Embed and search with extra candidates to allow for filtering
    query_vec       = embed_query(query, model)
    search_k        = min(k * 6, index.ntotal)
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = chunks[idx]

        # Apply optional filters
        if doc_type and chunk.metadata.doc_type != doc_type:
            continue
        if company and (
            chunk.metadata.company is None or
            company.lower() not in chunk.metadata.company.lower()
        ):
            continue

        results.append({
            'chunk': chunk,
            'score': float(score),
        })

        if len(results) >= k:
            break

    # Add 1-based rank
    for i, r in enumerate(results):
        r['rank'] = i + 1

    return results


# ════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ════════════════════════════════════════════════════════

def build_and_save(
    chunks     : list[Chunk],
    output_dir : str | Path,
    model      = None,
) -> tuple[faiss.IndexFlatIP, list[Chunk]]:
    """
    Full pipeline: embed chunks → build index → save to disk.
    Call this once after chunker.chunk_all().

    Args:
        chunks     : output of chunker.chunk_all()
        output_dir : where to save index.faiss + chunks.pkl
        model      : SentenceTransformer (loads automatically if None)

    Returns:
        (index, chunks) ready for search()
    """
    from ingestion.embeddings import embed_chunks

    if model is None:
        model = load_model()

    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_chunks(chunks, model)

    print("Building FAISS index...")
    index = build_index(vectors)

    print("Saving to disk...")
    save_index(index, chunks, output_dir)

    return index, chunks