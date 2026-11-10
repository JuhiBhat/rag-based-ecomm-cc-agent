"""Central configuration, read from environment variables (and an optional .env file).

No secrets are hard-coded: OPENAI_API_KEY must come from the environment.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The exact sentence the assistant must use when the documents do not contain the answer.
REFUSAL_MESSAGE = "I don't have enough information in the provided documents."


def _path(env_name: str, default: str) -> Path:
    value = Path(os.getenv(env_name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


def _provider(env_name: str) -> str:
    value = os.getenv(env_name, "openai").strip().lower()
    if value not in {"openai", "ollama"}:
        raise ValueError(f"{env_name} must be 'openai' or 'ollama', got '{value}'")
    return value


def _default_chat_model() -> str:
    if _provider("LLM_PROVIDER") == "ollama":
        return os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:9b")
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _default_embedding_model() -> str:
    if _provider("EMBEDDING_PROVIDER") == "ollama":
        return os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _optional_bool(env_name: str, default: str) -> bool | None:
    value = os.getenv(env_name, default).strip().lower()
    if value in {"", "none", "auto"}:
        return None
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Model providers: "openai" (cloud API) or "ollama" (local models on your computer).
    # They are independent, e.g. local embeddings with OpenAI answers is allowed.
    llm_provider: str = field(default_factory=lambda: _provider("LLM_PROVIDER"))
    embedding_provider: str = field(default_factory=lambda: _provider("EMBEDDING_PROVIDER"))
    chat_model: str = field(default_factory=_default_chat_model)
    embedding_model: str = field(default_factory=_default_embedding_model)
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))

    # Ollama-only settings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    # Ollama's default context window is small; prompts with 20 rerank passages need more.
    ollama_num_ctx: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", "16384")))
    # "Thinking" models (Qwen 3.5, Gemma 4) are slower with reasoning on. false = off,
    # true = on, empty = model default (use empty if a model rejects the setting).
    ollama_reasoning: bool | None = field(default_factory=lambda: _optional_bool("OLLAMA_REASONING", "false"))

    # Paths
    docs_dir: Path = field(default_factory=lambda: _path("DOCS_DIR", "data/documents"))
    sources_file: Path = field(default_factory=lambda: _path("SOURCES_FILE", "data/sources.json"))
    feedback_file: Path = field(default_factory=lambda: _path("FEEDBACK_FILE", "logs/feedback.jsonl"))
    interactions_file: Path = field(
        default_factory=lambda: _path("INTERACTIONS_FILE", "logs/interactions.jsonl")
    )

    # Qdrant vector store. If QDRANT_URL is set (e.g. http://qdrant:6333 in Docker) a Qdrant
    # server is used; otherwise Qdrant runs embedded, persisted on disk at QDRANT_PATH.
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL") or None)
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    qdrant_path: Path = field(default_factory=lambda: _path("QDRANT_PATH", "vectorstore/qdrant"))
    collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "ecommerce_support_docs")
    )

    # Chunking
    chunking_strategy: str = field(default_factory=lambda: os.getenv("CHUNKING_STRATEGY", "recursive"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))

    # Retrieval (stage 1: vector search) - FETCH_K candidates from Qdrant.
    fetch_k: int = field(default_factory=lambda: int(os.getenv("FETCH_K", "20")))
    # Cosine similarity below which a candidate is dropped before reranking (loose recall filter).
    min_relevance: float = field(default_factory=lambda: float(os.getenv("MIN_RELEVANCE", "0.20")))

    # Reranking (stage 2) - keep the best TOP_K candidates scoring >= MIN_RERANK_SCORE (0-1).
    reranker: str = field(default_factory=lambda: os.getenv("RERANKER", "llm"))
    # Model for the llm reranker; empty = same as chat_model (on the same provider).
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_LLM_MODEL", ""))
    cross_encoder_model: str = field(
        default_factory=lambda: os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    min_rerank_score: float = field(default_factory=lambda: float(os.getenv("MIN_RERANK_SCORE", "0.40")))

    # Conversation memory: number of past user/assistant exchanges kept per session.
    history_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_TURNS", "6")))

    @property
    def embedding_id(self) -> str:
        return f"{self.embedding_provider}:{self.embedding_model}"

    @property
    def collection(self) -> str:
        """Qdrant collection actually used: one per embedding model.

        Vectors from different embedding models are not comparable (and often differ in
        size), so switching models gets its own collection instead of corrupting the old one.
        Switching back reuses the earlier index without re-ingesting.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", self.embedding_id.lower()).strip("-")
        return f"{self.collection_name}__{slug}"


def get_settings() -> Settings:
    return Settings()


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export it in your shell."
        )
