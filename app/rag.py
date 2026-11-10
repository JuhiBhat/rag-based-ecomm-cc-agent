"""Retrieval-augmented, history-aware support assistant with reranking.

Pipeline for each user message:
    1. Condense   - rewrite a follow-up ("does this apply to discounted items?") into a
                    standalone question using the conversation history.
    2. Retrieve   - FETCH_K candidate chunks from Qdrant by cosine similarity; candidates
                    below MIN_RELEVANCE are dropped (cheap recall filter).
    3. Rerank     - a reranker (LLM or cross-encoder) scores each candidate against the
                    question; keep the best TOP_K with score >= MIN_RERANK_SCORE.
    4. Gate       - if nothing survives, refuse without calling the answer model.
    5. Generate   - the LLM answers from the numbered passages only, citing [n].
    6. Verify     - refusals are normalised; cited passage numbers become source citations.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict, dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStore

from app.config import REFUSAL_MESSAGE, Settings
from app.feedback import log_interaction
from app.memory import ConversationStore
from app.models import strip_reasoning
from app.prompts import ANSWER_PROMPT, CONDENSE_PROMPT
from app.reranker import NoOpReranker, RankedChunk, Reranker

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    ref: int
    title: str
    source: str
    page: int | None
    url: str
    score: float  # vector (cosine) similarity from Qdrant
    rerank_score: float  # 0-1 relevance from the reranker
    excerpt: str


@dataclass
class ChatResult:
    message_id: str
    session_id: str
    question: str
    standalone_question: str
    answer: str
    refused: bool
    sources: list[Source] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)  # diagnostics: candidates, reranker, kept

    def to_dict(self) -> dict:
        return asdict(self)


def _format_context(chunks: list[RankedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.document.metadata
        location = f", page {meta['page']}" if meta.get("page") else ""
        title = meta.get("title", meta.get("source"))
        blocks.append(f"[{i}] (Document: {title}{location})\n{chunk.document.page_content}")
    return "\n\n---\n\n".join(blocks)


def _is_refusal(answer: str) -> bool:
    normalised = answer.strip().replace("’", "'").lower()
    return normalised.startswith(REFUSAL_MESSAGE.lower().rstrip("."))


class SupportAssistant:
    def __init__(
        self,
        settings: Settings,
        llm: Runnable,
        vectorstore: VectorStore,
        reranker: Reranker | None = None,
        memory: ConversationStore | None = None,
    ):
        self.settings = settings
        self.vectorstore = vectorstore
        self.reranker = reranker or NoOpReranker()
        self.memory = memory or ConversationStore(settings.history_turns)
        clean = RunnableLambda(strip_reasoning)  # drop <think> blocks from local reasoning models
        self._condense_chain = CONDENSE_PROMPT | llm | StrOutputParser() | clean
        self._answer_chain = ANSWER_PROMPT | llm | StrOutputParser() | clean

    # -- pipeline steps ---------------------------------------------------------------
    def condense(self, question: str, history: list) -> str:
        if not history:
            return question
        standalone = self._condense_chain.invoke({"history": history, "question": question}).strip()
        return standalone or question

    def retrieve(self, query: str) -> tuple[list[RankedChunk], dict]:
        """Stage 1 vector search + stage 2 rerank. Returns kept chunks and diagnostics."""
        s = self.settings
        hits = self.vectorstore.similarity_search_with_score(query, k=max(s.fetch_k, s.top_k))
        candidates = [(d, sc) for d, sc in hits if sc >= s.min_relevance]

        ranked = self.reranker.rerank(query, candidates) if candidates else []
        kept = [c for c in ranked if c.rerank_score >= s.min_rerank_score][: s.top_k]

        diagnostics = {
            "reranker": self.reranker.name,
            "retrieved": len(hits),
            "candidates": len(candidates),
            "kept": len(kept),
            "top_rerank_scores": [round(c.rerank_score, 3) for c in ranked[:5]],
        }
        logger.info("Retrieval %s", diagnostics)
        return kept, diagnostics

    @staticmethod
    def _sources(chunks: list[RankedChunk], answer: str) -> list[Source]:
        cited = {int(n) for n in _CITATION_RE.findall(answer)}
        selected = [i for i in range(1, len(chunks) + 1) if i in cited] or list(range(1, len(chunks) + 1))
        sources = []
        for i in selected:
            chunk = chunks[i - 1]
            meta = chunk.document.metadata
            sources.append(
                Source(
                    ref=i,
                    title=meta.get("title", meta.get("source", "")),
                    source=meta.get("source", ""),
                    page=meta.get("page"),
                    url=meta.get("url", ""),
                    score=round(float(chunk.vector_score), 4),
                    rerank_score=round(float(chunk.rerank_score), 4),
                    excerpt=chunk.document.page_content[:400],
                )
            )
        return sources

    # -- public API -------------------------------------------------------------------
    def ask(self, question: str, session_id: str | None = None) -> ChatResult:
        question = question.strip()
        session_id = session_id or str(uuid.uuid4())
        history = self.memory.get(session_id)

        standalone = self.condense(question, history)
        chunks, diagnostics = self.retrieve(standalone)

        if not chunks:
            answer, refused, sources = REFUSAL_MESSAGE, True, []
        else:
            answer = self._answer_chain.invoke(
                {"context": _format_context(chunks), "history": history, "question": question}
            ).strip()
            refused = _is_refusal(answer)
            if refused:
                answer, sources = REFUSAL_MESSAGE, []
            else:
                sources = self._sources(chunks, answer)

        self.memory.append(session_id, question, answer)
        result = ChatResult(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            question=question,
            standalone_question=standalone,
            answer=answer,
            refused=refused,
            sources=sources,
            retrieval=diagnostics,
        )
        log_interaction(self.settings.interactions_file, result.to_dict())
        return result

    def reset(self, session_id: str) -> None:
        self.memory.clear(session_id)


def build_assistant(settings: Settings) -> SupportAssistant:
    """Wire up the production assistant: OpenAI models, Qdrant collection, reranker."""
    from app.models import get_chat_model, get_embeddings, get_rerank_model
    from app.reranker import build_reranker
    from app.vectorstore import load_vectorstore

    embeddings = get_embeddings(settings)
    rerank_llm = get_rerank_model(settings) if settings.reranker == "llm" else None
    return SupportAssistant(
        settings,
        get_chat_model(settings),
        load_vectorstore(embeddings, settings),
        reranker=build_reranker(settings, rerank_llm),
    )
