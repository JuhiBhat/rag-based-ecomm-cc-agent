"""Streamlit chat UI for the support assistant.

Talks to the FastAPI service at API_URL (default http://localhost:8000).
Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 90

EXAMPLES = [
    "What is Meesho's return and exchange policy?",
    "Who pays the return shipping fee on Meesho?",
    "What must an e-commerce marketplace in India disclose about returns and refunds?",
    "What rights does a consumer have under the Consumer Protection Act, 2019?",
    "How do I maintain the Havells Siphon water purifier?",
    "What does the Bajaj mixer grinder warranty cover?",
]

st.set_page_config(page_title="E-commerce Support Assistant", page_icon="🛒", layout="centered")


def api(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
        st.error(f"API error: {detail}")
        return None


def init_state():
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("messages", [])  # dicts: role, content, sources, message_id
    st.session_state.setdefault("feedback_given", {})


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = f", page {s['page']}" if s.get("page") else ""
            title = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
            st.markdown(
                f"**[{s['ref']}] {title}**{page} · rerank {s['rerank_score']:.2f} · vector {s['score']:.2f}"
            )
            st.caption(s["excerpt"] + ("…" if len(s["excerpt"]) >= 400 else ""))


def render_feedback(msg):
    mid = msg["message_id"]
    given = st.session_state.feedback_given.get(mid)
    if given:
        st.caption(f"Feedback recorded: {'👍' if given == 'up' else '👎'}")
        return
    cols = st.columns([1, 1, 8])
    comment_key = f"comment_{mid}"
    for col, rating, label in ((cols[0], "up", "👍"), (cols[1], "down", "👎")):
        if col.button(label, key=f"{rating}_{mid}"):
            payload = {
                "session_id": st.session_state.session_id,
                "message_id": mid,
                "rating": rating,
                "comment": st.session_state.get(comment_key, ""),
            }
            if api("POST", "/feedback", json=payload):
                st.session_state.feedback_given[mid] = rating
                st.rerun()
    cols[2].text_input("Optional comment", key=comment_key, label_visibility="collapsed",
                       placeholder="Optional comment, then click 👍 or 👎")


def ask(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Searching the documents…"):
        data = api("POST", "/chat", json={"question": question, "session_id": st.session_state.session_id})
    if data:
        st.session_state.session_id = data["session_id"]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"],
                "message_id": data["message_id"],
                "refused": data["refused"],
                "standalone": data["standalone_question"],
                "retrieval": data.get("retrieval", {}),
            }
        )


init_state()

with st.sidebar:
    st.header("🛒 Support Assistant")
    st.caption("Answers only from the indexed documents, with citations.")
    health = api("GET", "/health")
    if health and health["ready"]:
        st.success("API ready")
    elif health:
        st.warning(f"API not ready: {health['error']}")
    docs = api("GET", "/documents") or []
    if docs:
        st.subheader("Knowledge base")
        for d in docs:
            mark = "✅" if d["available"] else "⚠️ missing"
            st.markdown(f"- {mark} [{d['title']}]({d['url']})")
    show_debug = st.toggle("Show retrieval details", value=False)
    if st.button("New conversation", use_container_width=True):
        if st.session_state.session_id:
            api("DELETE", f"/sessions/{st.session_state.session_id}")
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()
    summary = api("GET", "/feedback/summary")
    if summary:
        st.caption(f"Feedback so far: 👍 {summary['up']} · 👎 {summary['down']}")

st.title("E-commerce Support Assistant")

if not st.session_state.messages:
    st.write("Try one of these:")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}"):
            ask(ex)
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if show_debug:
                r = msg.get("retrieval", {})
                st.caption(
                    f"Search query: {msg['standalone']} · reranker: {r.get('reranker')} · "
                    f"{r.get('candidates', 0)} candidates → {r.get('kept', 0)} kept"
                )
            render_sources(msg.get("sources"))
            render_feedback(msg)

if prompt := st.chat_input("Ask about returns, refunds, shipping, warranties or the manual…"):
    ask(prompt)
    st.rerun()
