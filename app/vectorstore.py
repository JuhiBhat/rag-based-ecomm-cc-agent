"""Qdrant vector store: build the collection at ingest time, open it at query time.

Mode is chosen by configuration:
* QDRANT_URL set      -> Qdrant server (the Docker Compose `qdrant` service, or Qdrant Cloud)
* QDRANT_URL unset    -> embedded Qdrant persisted to QDRANT_PATH (no server needed)

Embedded mode locks its folder, so only one process can open it at a time. Run ingestion
before starting the API, and use server mode when several processes need the index.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _client(url: str | None, api_key: str | None, path: str) -> QdrantClient:
    if url:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(path=path)


def get_client(settings: Settings) -> QdrantClient:
    if not settings.qdrant_url:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return _client(settings.qdrant_url, settings.qdrant_api_key, str(settings.qdrant_path))


def build_vectorstore(
    chunks: list[Document], embeddings: Embeddings, settings: Settings
) -> QdrantVectorStore:
    """(Re)create the collection and index all chunks with cosine distance."""
    client = get_client(settings)
    if client.collection_exists(settings.collection):
        logger.info("Dropping existing collection %s", settings.collection)
        client.delete_collection(settings.collection)

    dimension = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=settings.collection,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=embeddings,
        distance=models.Distance.COSINE,
    )
    store.add_documents(chunks)
    return store


def load_vectorstore(embeddings: Embeddings, settings: Settings) -> QdrantVectorStore:
    client = get_client(settings)
    if not client.collection_exists(settings.collection):
        raise FileNotFoundError(
            f"Qdrant collection '{settings.collection}' not found. Run `python ingest.py` first."
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=embeddings,
        distance=models.Distance.COSINE,
    )


def count_points(settings: Settings) -> int:
    client = get_client(settings)
    if not client.collection_exists(settings.collection):
        return 0
    return client.count(settings.collection, exact=True).count
