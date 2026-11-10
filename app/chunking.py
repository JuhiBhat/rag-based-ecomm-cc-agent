"""Split documents into semantically coherent chunks.

Two strategies:
* ``recursive`` (default): structure-aware splitting that prefers section breaks, then
  paragraphs, then sentences, so a chunk rarely cuts a policy clause in half. Cheap and
  deterministic.
* ``semantic``: LangChain's SemanticChunker, which embeds sentences and starts a new chunk
  where the topic shifts (embedding distance jumps). Costs extra embedding calls.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", ""]


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def split_documents(
    documents: list[Document], settings: Settings, embeddings: Embeddings | None = None
) -> list[Document]:
    cleaned = [Document(page_content=_clean(d.page_content), metadata=d.metadata) for d in documents]

    if settings.chunking_strategy == "semantic":
        if embeddings is None:
            raise ValueError("Semantic chunking needs an embeddings model.")
        try:
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError as exc:  # optional dependency
            raise ImportError(
                "CHUNKING_STRATEGY=semantic needs `pip install langchain-experimental`."
            ) from exc
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=SEPARATORS,
        )

    chunks = splitter.split_documents(cleaned)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 40]  # drop headers/footers
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
