"""FastAPI service exposing the support assistant.

Endpoints
    GET    /health                 liveness + index status
    GET    /documents              documents in the knowledge base
    POST   /chat                   ask a question (send session_id to keep context)
    DELETE /sessions/{session_id}  clear a conversation
    POST   /feedback               thumbs up/down (+ comment) on an answer
    GET    /feedback/summary       aggregate feedback counts

Run locally:  uvicorn app.api:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.feedback import feedback_summary, save_feedback
from app.loaders import load_source_catalog
from app.models import describe
from app.rag import SupportAssistant, build_assistant
from app.vectorstore import count_points

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("api")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, description="Omit to start a new conversation")


class SourceOut(BaseModel):
    ref: int
    title: str
    source: str
    page: int | None
    url: str
    score: float = Field(description="Vector (cosine) similarity from Qdrant")
    rerank_score: float = Field(description="0-1 relevance assigned by the reranker")
    excerpt: str


class ChatResponse(BaseModel):
    message_id: str
    session_id: str
    answer: str
    refused: bool
    standalone_question: str
    sources: list[SourceOut]
    retrieval: dict = Field(default_factory=dict, description="Candidates retrieved / kept, reranker used")


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(None, max_length=2000)


def create_app(assistant: SupportAssistant | None = None) -> FastAPI:
    """``assistant`` can be injected (tests); otherwise it is built at startup."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if assistant is not None:
            app.state.assistant = assistant
        else:
            try:
                app.state.assistant = build_assistant(settings)
                logger.info("Assistant ready (%d chunks indexed) %s", count_points(settings), describe(settings))
            except Exception as exc:  # keep /health up so the problem is visible
                logger.error("Assistant not ready: %s", exc)
                app.state.assistant = None
                app.state.startup_error = str(exc)
        yield

    app = FastAPI(title="E-commerce Support Assistant", version="1.0.0", lifespan=lifespan)

    def _assistant(request: Request) -> SupportAssistant:
        bot = getattr(request.app.state, "assistant", None)
        if bot is None:
            detail = getattr(request.app.state, "startup_error", "Assistant not initialised")
            raise HTTPException(status_code=503, detail=detail)
        return bot

    @app.get("/health")
    def health(request: Request):
        bot = getattr(request.app.state, "assistant", None)
        return {
            "status": "ok" if bot else "degraded",
            "ready": bot is not None,
            "models": describe(settings),
            "error": getattr(request.app.state, "startup_error", None),
        }

    @app.get("/documents")
    def documents():
        catalog = load_source_catalog(settings.sources_file)
        present = {p.name for p in settings.docs_dir.glob("*")} if settings.docs_dir.exists() else set()
        return [
            {"filename": name, "title": e.get("title"), "url": e.get("url"), "available": name in present}
            for name, e in catalog.items()
        ]

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request):
        data = _assistant(request).ask(body.question, body.session_id).to_dict()
        data.pop("question", None)
        return ChatResponse(**data)

    @app.delete("/sessions/{session_id}")
    def reset_session(session_id: str, request: Request):
        _assistant(request).reset(session_id)
        return {"session_id": session_id, "cleared": True}

    @app.post("/feedback")
    def feedback(body: FeedbackRequest):
        record = save_feedback(
            settings.feedback_file,
            session_id=body.session_id,
            message_id=body.message_id,
            rating=body.rating,
            comment=body.comment,
        )
        return {"saved": True, "feedback": record}

    @app.get("/feedback/summary")
    def summary():
        return feedback_summary(settings.feedback_file)

    return app


app = create_app()
