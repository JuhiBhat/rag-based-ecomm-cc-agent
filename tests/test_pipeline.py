"""Offline tests of the full pipeline (no API key, no network).

OpenAI models are replaced by a deterministic bag-of-words embedding and a scripted fake
LLM, so the tests check the wiring: loading, chunking, Qdrant indexing, retrieval,
relevance gating, refusal, history-aware condensation, citations, the API and feedback.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda

from app.api import create_app
from app.chunking import split_documents
from app.config import REFUSAL_MESSAGE, get_settings
from app.loaders import load_documents
from app.rag import SupportAssistant
from app.reranker import LLMReranker
from app.vectorstore import build_vectorstore, count_points

FIXTURES = Path(__file__).parent / "fixtures"
DIM = 256
STOP = {"the", "a", "an", "of", "to", "is", "are", "and", "or", "in", "on", "for", "what",
        "does", "do", "this", "that", "it", "be", "can", "i", "my", "how", "with", "within"}


class KeywordEmbeddings(Embeddings):
    """Hashed bag-of-words vectors: texts sharing words get high cosine similarity."""

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in STOP:
                continue
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class ScriptedLLM:
    """Stands in for ChatOpenAI. Records every prompt it receives."""

    def __init__(self):
        self.calls: list[list[BaseMessage]] = []

    def __call__(self, prompt: ChatPromptValue) -> str:
        messages = prompt.to_messages()
        self.calls.append(messages)
        system = messages[0].content
        question = messages[-1].content
        if system.startswith("You are a relevance judge"):
            # Rerank step: score passages by keyword overlap with the question (0-10).
            q_part, passages_part = question.split("\n\nPassages:\n", 1)
            q_words = set(re.findall(r"[a-z]+", q_part.lower())) - STOP
            scores = []
            for block in re.split(r"\n\n(?=\[\d+\] )", passages_part):
                pid = int(re.match(r"\[(\d+)\]", block).group(1))
                overlap = len(q_words & set(re.findall(r"[a-z]+", block.lower())))
                scores.append({"id": pid, "score": min(10, overlap * 2)})
            return json.dumps({"scores": scores})
        if system.startswith("You rewrite"):
            # Condense step: resolve "this" using the previous user question.
            previous = [m.content for m in messages[1:-1] if m.type == "human"][-1]
            follow_up = question.replace("Latest message: ", "").split("\n")[0]
            return f"{follow_up} (regarding: {previous})"
        if "transferable" in question.lower():  # docs don't say -> model declines
            return f"{REFUSAL_MESSAGE}"
        if "discounted" in question.lower() or "clearance" in question.lower():
            return "No, according to the Demo Store Return Policy, clearance items are final sale [2]."
        if "30 days" in system:
            return "Based on the Demo Store Return Policy, items can be returned within 30 days [1]."
        return REFUSAL_MESSAGE


@pytest.fixture()
def settings(tmp_path):
    return replace(
        get_settings(),
        docs_dir=FIXTURES,
        sources_file=FIXTURES / "sources.json",
        qdrant_url=None,
        qdrant_path=tmp_path / "qdrant",
        collection_name="test_docs",
        chunk_size=300,
        chunk_overlap=30,
        min_relevance=0.2,
        interactions_file=tmp_path / "interactions.jsonl",
        feedback_file=tmp_path / "feedback.jsonl",
    )


@pytest.fixture()
def assistant(settings):
    embeddings = KeywordEmbeddings()
    chunks = split_documents(load_documents(settings.docs_dir, settings.sources_file), settings)
    store = build_vectorstore(chunks, embeddings, settings)
    llm = ScriptedLLM()
    runnable = RunnableLambda(llm)
    bot = SupportAssistant(settings, runnable, store, reranker=LLMReranker(runnable))
    bot.fake_llm = llm
    return bot


def _answer_calls(llm: ScriptedLLM):
    return [c for c in llm.calls if c[0].content.startswith("You are a customer-support")]


def test_ingestion_loads_chunks_and_indexes(settings):
    docs = load_documents(settings.docs_dir, settings.sources_file)
    assert {d.metadata["source"] for d in docs} == {"demo_return_policy.txt", "demo_blender_manual.txt"}
    assert docs[0].metadata["title"]  # title comes from sources.json
    chunks = split_documents(docs, settings)
    assert len(chunks) > len(docs)
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)
    build_vectorstore(chunks, KeywordEmbeddings(), settings)
    assert count_points(settings) == len(chunks)


def test_answer_is_grounded_and_cited(assistant):
    result = assistant.ask("What is the return policy window for returning items?")
    assert not result.refused
    assert result.answer.startswith("Based on the Demo Store Return Policy")
    assert result.sources and result.sources[0].source == "demo_return_policy.txt"
    # Retrieved + reranked context really was injected into the prompt.
    system_prompt = _answer_calls(assistant.fake_llm)[-1][0].content
    assert "30 days" in system_prompt
    assert result.retrieval["reranker"] == "llm"
    assert result.retrieval["kept"] <= assistant.settings.top_k
    assert 0 < result.sources[0].rerank_score <= 1


def test_reranker_reorders_and_filters(assistant):
    query = "clearance items final sale"
    candidates = assistant.vectorstore.similarity_search_with_score(query, k=10)
    ranked = assistant.reranker.rerank(query, candidates)
    assert len(ranked) == len(candidates)
    assert [c.rerank_score for c in ranked] == sorted((c.rerank_score for c in ranked), reverse=True)
    assert "clearance" in ranked[0].document.page_content.lower()
    kept, diag = assistant.retrieve(query)
    assert all(c.rerank_score >= assistant.settings.min_rerank_score for c in kept)
    assert diag["candidates"] >= diag["kept"]


def test_reranker_failure_falls_back_to_vector_order(settings):
    from app.reranker import LLMReranker as R

    broken = R(RunnableLambda(lambda _: "not json"))
    store_hits = [(type("D", (), {"page_content": "x", "metadata": {}})(), 0.9)]
    ranked = broken.rerank("q", store_hits)
    assert ranked[0].rerank_score == 0.9  # vector score carried over


def test_out_of_scope_question_is_refused_without_llm_call(assistant):
    result = assistant.ask("Who won the football championship yesterday?")
    assert result.refused
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []
    assert _answer_calls(assistant.fake_llm) == []  # gate stopped it before generation


def test_follow_up_uses_history(assistant):
    first = assistant.ask("What is the return policy window for returning items?")
    follow = assistant.ask("Does this apply to discounted clearance items?", first.session_id)
    assert "regarding: What is the return policy" in follow.standalone_question
    assert follow.answer.startswith("No, according to")
    # History was passed to the answer prompt.
    answer_messages = _answer_calls(assistant.fake_llm)[-1]
    assert any("return policy window" in m.content for m in answer_messages[1:-1])


def test_llm_refusal_is_normalised(assistant):
    # Relevant chunks are found (warranty words appear in the docs) but the scripted LLM
    # declines, so the assistant must return the exact refusal and no sources.
    result = assistant.ask("Is the blender motor warranty transferable to a new owner?")
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []


def test_provider_switch_builds_ollama_models(monkeypatch):
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    from app.models import get_chat_model, get_embeddings, get_rerank_model

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # Ollama must not need a key
    s = get_settings()
    assert s.chat_model == "qwen3.5:9b" and s.embedding_model == "bge-m3"
    chat, rerank, emb = get_chat_model(s), get_rerank_model(s), get_embeddings(s)
    assert isinstance(chat, ChatOllama) and chat.num_ctx == s.ollama_num_ctx
    assert rerank.format == "json"
    assert isinstance(emb, OllamaEmbeddings)
    # Different embedding models never share a Qdrant collection.
    assert s.collection.endswith("ollama-bge-m3")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    assert get_settings().collection != s.collection


def test_reasoning_blocks_are_stripped():
    from app.models import strip_reasoning

    assert strip_reasoning("<think>let me see...</think>\nBased on the policy [1].") == "Based on the policy [1]."


def test_api_chat_and_feedback(assistant, settings, monkeypatch):
    monkeypatch.setattr("app.api.get_settings", lambda: settings)
    with TestClient(create_app(assistant)) as client:
        assert client.get("/health").json()["ready"] is True
        r = client.post("/chat", json={"question": "What is the return policy window for returning items?"})
        assert r.status_code == 200
        body = r.json()
        assert body["sources"] and body["message_id"]

        r2 = client.post("/chat", json={"question": "Does this apply to clearance items?",
                                        "session_id": body["session_id"]})
        assert r2.json()["answer"].startswith("No,")

        fb = client.post("/feedback", json={"session_id": body["session_id"],
                                            "message_id": body["message_id"],
                                            "rating": "up", "comment": "clear answer"})
        assert fb.json()["saved"] is True
        assert client.get("/feedback/summary").json() == {"up": 1, "down": 0, "total": 1}
        assert client.post("/feedback", json={"session_id": "s", "message_id": "m",
                                              "rating": "meh"}).status_code == 422

    lines = settings.interactions_file.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["sources"]
