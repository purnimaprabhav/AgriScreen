"""
rag/retriever.py
Retrieves relevant chunks from the vector index and formats them
as a numbered context string ready for the LLM prompt.
"""

from ingestion.vector_store import search, load_index
from ingestion.embeddings import load_model


def format_context(results: list[dict]) -> str:
    """
    Converts retrieved chunks into a numbered [SOURCE N] context block.
    The LLM is instructed to cite these by number in its answer.
    """
    parts = []

    for r in results:
        chunk = r['chunk']
        meta  = chunk.metadata

        # Build a descriptive source label
        label_parts = [f"SOURCE {r['rank']}"]

        if meta.company:
            label_parts.append(meta.company)
        if meta.section:
            label_parts.append(f"section: {meta.section}")
        if meta.news_id:
            label_parts.append(meta.news_id)
        if meta.date:
            label_parts.append(meta.date)

        label_parts.append(meta.source_file)

        label = " | ".join(label_parts)
        parts.append(f"[{label}]\n{chunk.text}")

    return "\n\n---\n\n".join(parts)


class Retriever:
    """
    Wraps the FAISS index and embedding model.
    Initialise once and reuse across queries.
    """

    def __init__(self, index_dir: str = "outputs/index"):
        print("Loading embedding model...")
        self.model = load_model()

        print("Loading FAISS index...")
        self.index, self.chunks = load_index(index_dir)

        print("  ✓ Retriever ready\n")

    def retrieve(
        self,
        query    : str,
        k        : int        = 5,
        company  : str | None = None,
        doc_type : str | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Retrieves top-k chunks for a query with optional filters.

        Args:
            query    : natural language question
            k        : number of chunks to retrieve
            company  : filter by company name (partial match)
            doc_type : filter by type — factsheet | news | report | financials | funding

        Returns:
            context  : formatted [SOURCE N] string for the LLM prompt
            results  : raw list of dicts with chunk + score (for citation display)
        """
        results = search(
            query    = query,
            index    = self.index,
            chunks   = self.chunks,
            model    = self.model,
            k        = k,
            company  = company,
            doc_type = doc_type,
        )

        context = format_context(results)
        return context, results