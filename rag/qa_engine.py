"""
rag/qa_engine.py
Combines the Retriever and Groq LLM into a single ask() interface.
Supports both standard (ask) and streaming (ask_stream) responses.
"""

import os
from groq import Groq
from rag.retriever import Retriever
from rag.prompts import SYSTEM_PROMPT, build_user_prompt

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024


class QAEngine:
    """Full RAG pipeline: retrieve → prompt → generate → return cited answer."""

    def __init__(self, index_dir="outputs/index", api_key=None):
        self.retriever = Retriever(index_dir)
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found.")
        self.client = Groq(api_key=key)
        print("  ✓ QA engine ready")

    def ask(self, query, k=5, company=None, doc_type=None):
        """Non-streaming RAG. Returns full answer at once."""
        context, results = self.retriever.retrieve(
            query=query, k=k, company=company, doc_type=doc_type
        )
        if not results:
            return {'answer': "No relevant documents found for this query.",
                    'sources': [], 'query': query}

        user_prompt = build_user_prompt(query, context)
        response = self.client.chat.completions.create(
            model       = GROQ_MODEL,
            max_tokens  = MAX_TOKENS,
            temperature = 0,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ]
        )
        return {'answer': response.choices[0].message.content,
                'sources': results, 'query': query}

    def ask_stream(self, query, k=5, company=None, doc_type=None):
        """
        Streaming RAG. Returns (generator, sources).
        Generator yields tokens. Use with st.write_stream() in Streamlit.
        """
        context, results = self.retriever.retrieve(
            query=query, k=k, company=company, doc_type=doc_type
        )
        if not results:
            def empty():
                yield "No relevant documents found for this query."
            return empty(), []

        user_prompt = build_user_prompt(query, context)

        def token_generator():
            stream = self.client.chat.completions.create(
                model       = GROQ_MODEL,
                max_tokens  = MAX_TOKENS,
                temperature = 0,
                stream      = True,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return token_generator(), results


def print_result(result):
    print(f"QUERY: {result['query']}")
    print("=" * 60)
    print(result['answer'])
    print()
    print(f"SOURCES USED ({len(result['sources'])}):")
    for r in result['sources']:
        meta = r['chunk'].metadata
        print(f"  [{r['rank']}] score={r['score']:.3f} "
              f"| {meta.doc_type} | {meta.company or 'n/a'} "
              f"| {meta.section or meta.news_id or ''}")