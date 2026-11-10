"""Load PDF and TXT documents from a local folder into LangChain Documents."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_source_catalog(sources_file: Path) -> dict[str, dict]:
    """Map file name -> {title, url, ...} so citations can show a readable title and link."""
    if not sources_file.exists():
        return {}
    entries = json.loads(sources_file.read_text(encoding="utf-8")).get("documents", [])
    return {entry["filename"]: entry for entry in entries}


def _base_metadata(path: Path, catalog: dict[str, dict]) -> dict:
    entry = catalog.get(path.name, {})
    return {
        "source": path.name,
        "title": entry.get("title", path.stem.replace("_", " ").replace("-", " ").title()),
        "url": entry.get("url", ""),
        "doc_type": entry.get("doc_type", "document"),
    }


def _load_pdf(path: Path, base: dict) -> list[Document]:
    reader = PdfReader(str(path))
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={**base, "page": page_number}))
    return docs


def _load_txt(path: Path, base: dict) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [Document(page_content=text, metadata={**base, "page": None})] if text else []


def load_documents(docs_dir: Path, sources_file: Path | None = None) -> list[Document]:
    """Load every supported file in ``docs_dir``. PDFs yield one Document per page."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents folder not found: {docs_dir}")

    catalog = load_source_catalog(sources_file) if sources_file else {}
    documents: list[Document] = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        base = _base_metadata(path, catalog)
        loaded = _load_pdf(path, base) if path.suffix.lower() == ".pdf" else _load_txt(path, base)
        if not loaded:
            logger.warning("No extractable text in %s (scanned PDF?) - skipped", path.name)
        documents.extend(loaded)
        logger.info("Loaded %s (%d page/section(s))", path.name, len(loaded))

    if not documents:
        raise ValueError(
            f"No PDF/TXT content found in {docs_dir}. Run scripts/download_documents.py "
            "or add your own files."
        )
    return documents
