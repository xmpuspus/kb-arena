"""Vector store adapters behind one interface.

ChromaDB stays the default and ships in the core install. Qdrant, pgvector,
and LanceDB are optional adapters behind extras, so `pip install kb-arena`
alone never pulls in their SDKs. Every adapter lazy-imports its SDK inside a
method body, the same pattern kb_arena.strategies.embeddings uses for
embedding providers, so importing this package pulls in none of them.
"""

from __future__ import annotations

from kb_arena.vectorstores.base import VectorMatch, VectorStore
from kb_arena.vectorstores.chroma_store import ChromaVectorStore
from kb_arena.vectorstores.lancedb_store import LanceDBVectorStore
from kb_arena.vectorstores.pgvector_store import PgVectorStore
from kb_arena.vectorstores.qdrant_store import QdrantVectorStore

__all__ = [
    "VectorMatch",
    "VectorStore",
    "ChromaVectorStore",
    "QdrantVectorStore",
    "PgVectorStore",
    "LanceDBVectorStore",
]
