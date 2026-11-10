"""Factories for the AI models used by the pipeline: OpenAI (cloud) or Ollama (local).

LLM_PROVIDER picks the chat model (condense, rerank, answer); EMBEDDING_PROVIDER picks the
embedding model (ingestion + search). Everything else in the pipeline is provider-agnostic.
"""
from __future__ import annotations

import re

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, require_api_key

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks some local reasoning models put in their output."""
    return _THINK_RE.sub("", text).strip()


def _ollama_chat(settings: Settings, model: str, json_mode: bool = False) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    kwargs = dict(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature if not json_mode else 0,
        num_ctx=settings.ollama_num_ctx,
    )
    if settings.ollama_reasoning is not None:
        kwargs["reasoning"] = settings.ollama_reasoning
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)

    from langchain_openai import OpenAIEmbeddings

    require_api_key()
    return OpenAIEmbeddings(model=settings.embedding_model)


def get_chat_model(settings: Settings, model: str | None = None) -> BaseChatModel:
    model = model or settings.chat_model
    if settings.llm_provider == "ollama":
        return _ollama_chat(settings, model)

    from langchain_openai import ChatOpenAI

    require_api_key()
    return ChatOpenAI(model=model, temperature=settings.temperature)


def get_rerank_model(settings: Settings) -> BaseChatModel:
    """Chat model for the llm reranker, in JSON mode so its scores always parse."""
    model = settings.rerank_model or settings.chat_model
    if settings.llm_provider == "ollama":
        return _ollama_chat(settings, model, json_mode=True)

    from langchain_openai import ChatOpenAI

    require_api_key()
    return ChatOpenAI(
        model=model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def describe(settings: Settings) -> dict:
    """Human-readable summary of which models are active (shown in /health and logs)."""
    return {
        "llm": f"{settings.llm_provider}:{settings.chat_model}",
        "embeddings": settings.embedding_id,
        "reranker": settings.reranker,
        "collection": settings.collection,
    }
