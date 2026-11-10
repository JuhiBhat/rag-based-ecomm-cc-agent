"""Document ingestion: load PDFs/TXTs -> chunk -> embed with OpenAI -> index in Qdrant.

Usage:
    python ingest.py                 # uses DOCS_DIR (default data/documents)
    python ingest.py --docs-dir path/to/folder
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path

from app.chunking import split_documents
from app.config import get_settings
from app.loaders import load_documents
from app.models import get_embeddings
from app.vectorstore import build_vectorstore, count_points

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ingest")


def run(docs_dir: Path | None = None) -> int:
    settings = get_settings()
    if docs_dir:
        settings = replace(settings, docs_dir=docs_dir.resolve())

    logger.info("Loading documents from %s", settings.docs_dir)
    documents = load_documents(settings.docs_dir, settings.sources_file)

    embeddings = get_embeddings(settings)
    chunks = split_documents(documents, settings, embeddings)
    per_file = Counter(c.metadata["source"] for c in chunks)
    for name, n in sorted(per_file.items()):
        logger.info("  %-55s %4d chunks", name, n)

    logger.info("Embedding %d chunks with %s ...", len(chunks), settings.embedding_id)
    if settings.embedding_provider == "ollama":
        logger.info("  (local embeddings: this can take a few minutes on a laptop)")
    build_vectorstore(chunks, embeddings, settings)

    target = settings.qdrant_url or settings.qdrant_path
    points = count_points(settings)
    logger.info("Indexed %d chunks into Qdrant collection '%s' (%s)", points, settings.collection, target)
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-dir", type=Path, default=None, help="Folder with PDF/TXT files")
    run(parser.parse_args().docs_dir)
