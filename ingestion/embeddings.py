"""
ingestion/embeddings.py

Loads the embedding model and encodes Chunk text into vectors.
Vectors are L2-normalised so cosine similarity = dot product in FAISS.

Model: all-MiniLM-L6-v2
  - 384-dimensional output
  - ~80MB, fast on CPU, no GPU needed
  - Strong performance on short-to-medium texts
"""

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from utils.metadata import Chunk

# Prevents macOS fork/segfault when tokenizers and torch
# parallelism conflict with HTTP client threads (e.g. Groq)
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM        = 384


def load_model() -> SentenceTransformer:
    """Loads and returns the embedding model. Call once and reuse."""
    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print("  ✓ Model loaded")
    return model


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Encodes a list of strings into L2-normalised float32 vectors.

    Returns:
        np.ndarray of shape (len(texts), EMBED_DIM), dtype float32
    """
    vectors = model.encode(
        texts,
        batch_size        = 64,
        show_progress_bar = True,
        convert_to_numpy  = True,
    ).astype("float32")

    # L2 normalise — inner product == cosine similarity in FAISS
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1e-9, norms)
    vectors = vectors / norms

    return vectors


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> np.ndarray:
    """
    Convenience wrapper: extracts text from Chunk objects then embeds.
    Returns vectors in the same order as chunks.
    """
    texts = [chunk.text for chunk in chunks]
    return embed_texts(texts, model)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Encodes a single query string.
    Returns shape (1, EMBED_DIM) — matches what FAISS expects for search.
    """
    vectors = model.encode(
        [query],
        convert_to_numpy  = True,
    ).astype("float32")

    # L2 normalise
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1e-9, norms)
    vectors = vectors / norms

    return vectors.reshape(1, -1)