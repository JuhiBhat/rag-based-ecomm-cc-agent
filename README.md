# E-commerce Domain Support Assistant (RAG)

A customer-support chatbot for Indian e-commerce. It answers questions about **return, refund and exchange rules (Meesho), consumer law (Consumer Protection Act 2019, E-Commerce Rules 2020), a product manual (Havells)**. It answers **only from the provided documents**, cites its sources, handles follow-up questions, and refuses when the documents don't cover a question:

> I don't have enough information in the provided documents.

Stack: **LangChain · Qdrant · FastAPI · Streamlit · Docker**, with AI models from **OpenAI (cloud)** or **Ollama (free, on your own computer)**. You can switch between them with one setting.

> **Note on the sample logs and screenshots.** The assistant is built for the **OpenAI API**, which is the default configuration (`LLM_PROVIDER=openai`, `EMBEDDING_PROVIDER=openai`, using `gpt-4o-mini` and `text-embedding-3-small`; see `.env.example`). During testing my OpenAI account could not be used, so the sample conversations and screenshots were produced with **the same code running local models through Ollama** (`qwen3.5:9b` for chat, `bge-m3` for embeddings), selected by changing two settings in `.env`. No code changes are needed to run it with OpenAI: set both providers to `openai`, add `OPENAI_API_KEY`, and run `python ingest.py` (or restart Docker).

---

## 1. The idea in one paragraph

An AI model is like a brilliant new hire who has read the internet but has never seen *your* policies, and who will guess confidently rather than say "I don't know". RAG (retrieval-augmented generation) makes that hire safe to put in front of customers. You give them a **well-indexed library** of the right documents (Qdrant), a **rule to answer only from pages they've been handed** (the prompt and citations), and a **supervisor who stops them when the pages don't contain the answer** (reranking and the evidence check). Every component below serves one of those three jobs.


## 2. Architecture

### 2.1 System map

```
                 ┌──────────────────────── Docker Compose (your server) ────────────────────────┐
                 │                                                                               │
 User's   HTTP   │  ┌────────────┐  HTTP   ┌───────────────────────────────┐  HTTP  ┌──────────┐ │
 browser ───────▶│  │ ui         │ ──────▶ │ api (FastAPI + RAG core)      │ ─────▶ │ qdrant   │ │
 :8501           │  │ Streamlit  │ ◀────── │  memory · condense · retrieve │ ◀───── │ vector DB│ │
                 │  │ chat + 👍👎 │  JSON   │  rerank · gate · answer · cite│        │ :6333    │ │
                 │  └────────────┘         └───────┬──────────────┬────────┘        └────▲─────┘ │
                 │                                 │ writes       │ startup: ingest if   │       │
                 │                       logs/ ◀───┘              └── collection empty ──┘       │
                 │              interactions + feedback (JSONL)       (load → chunk → embed)     │
                 └───────────────────────────────────────┬─────────────────────▲─────────────────┘
                                                         │ AI calls            │ read-only mount
                                    ┌────────────────────▼───────────┐   ┌─────┴────────────────┐
                                    │ AI models (pick per .env)      │   │ data/documents/      │
                                    │  • OpenAI API (cloud)    or    │   │ 6 Indian PDFs/pages  │
                                    │  • Ollama on your computer     │   └──────────────────────┘
                                    └────────────────────────────────┘
```

- **Inside the box** are three containers on a private network. Only the `api` container talks to Qdrant and to the AI models, so your keys never reach the browser.
- **Outside the box** are your documents, mounted read-only, and the AI models. With Ollama, the models run on your own machine and nothing leaves it.

### 2.2 How one question travels across the containers

Example: the user earlier asked about Meesho returns and now types *"Does this apply to discounted items?"*

| # | From → To | What happens | Why it matters |
|---|---|---|---|
| 1 | Browser → `ui` | The user types the question. Streamlit holds the conversation's `session_id` | Ties follow-ups to the right conversation |
| 2 | `ui` → `api` (`POST /chat`) | Sends the question and `session_id`. FastAPI validates it and loads this user's last 6 exchanges from memory | One controlled entry point |
| 3 | `api` → AI model | **Clarify:** rewrites the follow-up into a full question, e.g. *"Does Meesho's return and exchange policy apply to discounted items?"* Skipped on the first message | The library can't search for "this" |
| 4 | `api` → AI model | **Embed:** turns the rewritten question into meaning-numbers (a vector) | Qdrant searches by meaning, not keywords |
| 5 | `api` → `qdrant` | **Search:** Qdrant returns the 20 closest passages with similarity scores, and weak ones (< 0.20) are dropped | Wide net, so the right passage is likely among them |
| 6 | `api` → AI model (or local cross-encoder) | **Rerank:** a judge scores how well each passage actually answers the question, and the best 4 scoring ≥ 0.40 are kept | Removes "sounds related but doesn't answer" passages |
| 7 | inside `api` | **Evidence check:** if nothing passes, the fixed refusal is returned and step 8 is skipped | The cheapest way to prevent invented answers |
| 8 | `api` → AI model | **Answer:** writes the reply using only the 4 passages, citing [1], [2] | A readable answer that stays tied to the source |
| 9 | `api` → `logs/` | Maps [n] to document, page and link, saves the exchange to memory, and appends it to `interactions.jsonl` with a `message_id` | Audit trail and context for the next follow-up |
| 10 | `api` → `ui` → browser | Returns JSON (answer, sources, scores, `refused`, `message_id`), which Streamlit renders | The user can verify every claim |
| 11 | Browser → `ui` → `api` (`POST /feedback`) | 👍/👎 plus a comment, saved to `feedback.jsonl` against the `message_id` | Shows exactly which answers need fixing |

A normal answer makes up to **4 AI calls** (steps 3, 4, 6 and 8). With OpenAI that's a few seconds and a small per-question cost. With Ollama it's free but slower, depending on your hardware.

### 2.3 What each component does

| Component | Plain-terms role | Problem it solves | Without it |
|---|---|---|---|
| Documents | The only facts the bot may use | The AI doesn't know your policies | Guesses or blanket refusals |
| Ingest (`ingest.py`) | Cuts documents into paragraph-sized passages and tags each with meaning-numbers | Precise, fast lookup instead of reading 200 pages per question | Slow, imprecise, expensive |
| Qdrant | The searchable library shelf | Finds passages by meaning ("send it back" ≈ "return") | Keyword search misses rephrasings |
| Streamlit (`ui`) | The chat screen | A simple interface for non-technical users | No usable front end |
| FastAPI (`api`) | Front desk that runs every step in order | One secure entry point, and any channel (web, app, WhatsApp) can plug in | Logic scattered and keys exposed |
| Session memory | Last 6 exchanges per user | Follow-ups like "what about eligibility?" | Every question treated in isolation |
| Clarify step | Rewrites follow-ups into complete questions | Search needs the full question | Wrong passages retrieved |
| Reranker | Shortlist interview after the CV screen | Similar is not the same as answering | Noisy context and weaker answers |
| Evidence check | Stops when the evidence is weak | Prevents improvised answers | Confident wrong answers |
| Answer step | Writes a clear, cited reply from 4 passages | Readable answers, tied to the source | Customers would have to read legal text |
| Citations | Source list under every answer | Anyone can verify in one click | No auditability |
| Feedback and logs | Ratings and a full record of every answer | Measure quality, find gaps, satisfy audits | You only learn of failures from complaints |
| Docker Compose | Starts everything with one command, identically anywhere | No "works on my laptop" problems | Manual, error-prone setup |
| `.env` | Keys and adjustable settings | Change behaviour without editing code | Leaked secrets, code edits for every tweak |

### 2.4 How hallucination is prevented
1. **Two-stage evidence gate.** Candidates below cosine 0.20 are dropped, then anything the reranker scores below 0.40. If nothing survives, the bot refuses **without calling the answer model**.
2. **Grounding prompt** (`app/prompts.py`). The model may use only the numbered passages, must cite each claim, must say which part is uncovered when an answer is partial, and must use the exact refusal sentence otherwise. Instructions found inside documents are ignored.
3. **Post-processing.** Refusal-like replies are normalised to the exact sentence with no sources. Citations come only from the passage numbers the model actually cited.
4. **Temperature 0**, so the model doesn't improvise.

---

## 3. Choosing AI models: OpenAI or Ollama

Two independent switches in `.env`:

```
LLM_PROVIDER=openai | ollama          # model that clarifies, reranks and answers
EMBEDDING_PROVIDER=openai | ollama    # model that turns text into meaning-numbers
```

| | OpenAI (cloud) | Ollama (local) |
|---|---|---|
| Cost | Pay per use, small per question | Free after download |
| Privacy | Questions and passages go to OpenAI | Nothing leaves your computer |
| Speed | Fast, a few seconds | Depends on your RAM/GPU, often slower on laptops |
| Quality | Strong and consistent | Good with the models below, weaker on very small models |
| Setup | API key | Install Ollama and pull models |

You can mix them, e.g. local embeddings (free, private indexing) with OpenAI answers.

**Each embedding model gets its own Qdrant collection** (`ecommerce_support_docs__ollama-bge-m3`, etc.). Vectors from different embedding models aren't comparable, so the first time you switch `EMBEDDING_PROVIDER` or the embedding model, run `python ingest.py` again. Switching back later reuses the earlier index.

### 3.1 Recommended Ollama models

Pick by your computer's memory (RAM, or unified memory on Apple Silicon):

| Your computer | Chat model (`OLLAMA_CHAT_MODEL`) | Embedding model (`OLLAMA_EMBEDDING_MODEL`) |
|---|---|---|
| 8 GB RAM | `qwen3.5:4b` (~3.4 GB) | `bge-m3` (~1.2 GB) |
| **16 GB RAM (most laptops, recommended)** | **`qwen3.5:9b`** (~6.6 GB) or `gemma4:12b-it-qat` (~7.2 GB) | **`bge-m3`** |
| 32 GB+ RAM or a 24 GB GPU | `gemma4:26b` (~18 GB, fast mixture-of-experts) | `bge-m3` |

Why these:
- **`bge-m3`** for embeddings: multilingual (Hindi and Hinglish questions still find English policy text), handles long passages (8K tokens), and needs no special prefixes. For an English-only setup on a very small machine, `nomic-embed-text` (274 MB) works too.
- **Qwen 3.5** and **Gemma 4** are current open model families with strong instruction-following and multilingual support. The 9B/12B sizes are the sweet spot for RAG on a laptop, while 4B works but is noticeably weaker at citing carefully and refusing correctly.
- These are reasoning ("thinking") models. `OLLAMA_REASONING=false` turns that off for speed, which RAG answers don't need. If a model you choose errors with "does not support thinking", set `OLLAMA_REASONING=` (empty).

Model tags change often, so check https://ollama.com/library before pulling.

**Reranking with local models:** the default `RERANKER=llm` uses your Ollama chat model to score all 20 candidates in one call. That works, but it's the slowest step on a laptop. For a fully local setup, consider `RERANKER=cross-encoder` (`pip install sentence-transformers`). It uses `BAAI/bge-reranker-v2-m3`, a small multilingual reranking model downloaded once, which is usually faster and more consistent than asking a small chat model to grade.

### 3.2 Set up Ollama
1. Install Ollama from https://ollama.com and open it (it runs in the background on port 11434).
2. Pull the models:
   ```bash
   ollama pull qwen3.5:9b
   ollama pull bge-m3
   ```
3. In `.env`:
   ```
   LLM_PROVIDER=ollama
   EMBEDDING_PROVIDER=ollama
   OLLAMA_CHAT_MODEL=qwen3.5:9b
   OLLAMA_EMBEDDING_MODEL=bge-m3
   ```
   No OpenAI key is needed when both providers are `ollama`.
4. Re-index with `python ingest.py`, then run as usual (section 5 or 6).

With Docker, keep Ollama running on your computer, not inside Docker: on a Mac, Docker can't use the Apple GPU. Compose already points the `api` container at `http://host.docker.internal:11434`.

---

## 4. Documents (all Indian, all publicly available)

Listed in `data/sources.json`, downloaded by `scripts/download_documents.py`:

| File | Document (public link) | Publisher | Covers |
|---|---|---|---|
| `india_consumer_protection_ecommerce_rules_2020.pdf` | [Consumer Protection (E-Commerce) Rules, 2020](https://taxguru.in/wp-content/uploads/2020/07/Consumer-Protection-E-Commerce-Rules-2020.pdf) | Ministry of Consumer Affairs, Govt. of India (G.S.R. 462(E)) | marketplace and seller duties, return/refund disclosure, grievance timelines |
| `CP Act 2019_1732700731.pdf` | [Consumer Protection Act, 2019](https://www.indiacode.nic.in/bitstream/123456789/18964/1/cpa.pdf) | India Code, Legislative Department, Govt. of India | consumer rights, defective goods, product liability, where to complain |
| `meesho.com_legal_returns, refunds and replacement_embed=true.pdf` | [Meesho Returns, Exchange and Refunds Policy](https://www.meesho.com/legal/returns,%20refunds%20and%20replacement?embed=true) | Meesho (official legal page) | customer returns, exchanges, refunds, non-returnable items, return costs |
| `Learn More About Shipping & Delivery Charges for Meesho Supplier.pdf` | [Meesho Shipping, Delivery & Returns Policy](https://supplier.meesho.com/shipping) | Meesho (supplier website) | dispatch timelines, return shipping fees, RTO, cancellations (seller side) |
| `Penalty Structure and Handling Incorrect Penalties _ Meesho.pdf` | [Meesho Supplier Learning Hub: penalties](https://supplier.meesho.com/learning-hub/lessons/how-to-handle-incorrect-penalties) | Meesho (supplier learning hub) | auto-cancellation, seller cancellation penalties, return shipping fee |
| `Siphon_WP_Manual_1.pdf` | [Havells Siphon Water Purifier User Manual](https://havells.com/media/wysiwyg/Siphon_WP_Manual_1.pdf) | Havells India Ltd. | installation, maintenance, troubleshooting, warranty |

- To add a document, put the PDF/TXT in `data/documents/`, add an entry to `sources.json` for a readable title and link, then re-run `python ingest.py`.
- Meesho's pages change over time. Refresh them with `python scripts/download_documents.py --force` and re-ingest.
- **Many sites block scripted downloads** (errors 403/406) or time out from some networks, so for this submission the documents were saved from a browser: PDFs downloaded directly, and the Meesho web pages saved with File → Print → Save as PDF. The files in `data/documents/` keep their browser filenames, and `data/sources.json` maps each filename to its title and public link (used in citations).

---

## 5. Run locally (no Docker)

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                   # Windows: copy .env.example .env
# edit .env: add OPENAI_API_KEY, or switch both providers to ollama (section 3.2)

python scripts/download_documents.py   # fetch the 6 documents into data/documents/
python ingest.py                       # chunk, embed and index into Qdrant
```

Text chat:
```bash
python chatbot.py --show-retrieval
```

Web chat (two terminals, both with `.venv` activated):
```bash
uvicorn app.api:app --port 8000        # API docs at http://localhost:8000/docs
streamlit run streamlit_app.py         # chat UI at http://localhost:8501
```

> Without Docker, Qdrant stores its data in a folder (`vectorstore/qdrant`) that only one program can open at a time. Finish `ingest.py` before starting the API, and stop the API before re-ingesting or running the sample-log script.

## 6. Run with Docker (UI + API + Qdrant)

```bash
cp .env.example .env                   # add your key, or set the providers to ollama
python scripts/download_documents.py
docker compose up --build
```
- Chat UI: http://localhost:8501 · API docs: http://localhost:8000/docs · Qdrant dashboard: http://localhost:6333/dashboard
- On first start, the API indexes the documents automatically if the collection for the current embedding model is empty.
- To re-index after changing documents: `docker compose run --rm -e FORCE_REINGEST=true api python scripts/ensure_index.py`
- To stop: `Ctrl+C` or `docker compose down`. Qdrant data persists in a Docker volume.

---

## 7. API

| Method | Path | Body / notes |
|---|---|---|
| GET | `/health` | readiness, active models (`llm`, `embeddings`, `reranker`, `collection`), startup error if any |
| GET | `/documents` | knowledge-base documents and whether each file is present |
| POST | `/chat` | `{"question": "...", "session_id": "optional"}`. Omit `session_id` to start a conversation, then send it back for follow-ups |
| DELETE | `/sessions/{id}` | clear conversation history |
| POST | `/feedback` | `{"session_id","message_id","rating":"up"\|"down","comment"}` |
| GET | `/feedback/summary` | 👍/👎 counts |

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"What is Meesho'"'"'s return and exchange policy?"}'
```
Response (shape):
```json
{
  "message_id": "…", "session_id": "…",
  "answer": "Based on the Meesho Returns, Exchange and Refunds Policy, … [1]",
  "refused": false,
  "standalone_question": "What is Meesho's return and exchange policy?",
  "sources": [{"ref": 1, "title": "Meesho Returns, Exchange and Refunds Policy",
               "source": "meesho_returns_refunds_exchange_policy.txt", "page": null,
               "url": "https://www.meesho.com/legal/…", "score": 0.61, "rerank_score": 0.9, "excerpt": "…"}],
  "retrieval": {"reranker": "llm", "retrieved": 20, "candidates": 14, "kept": 3, "top_rerank_scores": [0.9, 0.7, 0.6]}
}
```

## 8. Sample conversations / logs

Submitted evidence: `logs/sample_conversation.md` (full multi-turn run, with the rewritten follow-up queries, retrieval counts, rerank scores and citations) and `screenshots/sample conversation - screenshot.pdf` (Streamlit UI: an answer with Sources expanded, a 👍 rating recorded, a follow-up, and a refusal). `logs/feedback.jsonl` shows the recorded rating.

Reading notes for the sample run:
- Citation titles in that run show the raw filenames (e.g. "Cp Act 2019 1732700731") because the index was built before `data/sources.json` was updated. Re-running `python ingest.py` makes citations show the proper titles and links.
- The "Bajaj mixer grinder" questions refer to a product whose document is **not** in the knowledge base (the manufacturer removed it from their site), so the correct behaviour is the refusal shown.

```bash
python scripts/run_sample_conversation.py                         # local (stop the API first)
docker compose exec api python scripts/run_sample_conversation.py # Docker
```
This writes `logs/sample_conversation.md` and `.json`. It covers:
- Meesho return follow-ups ("What about eligibility?", "Does this apply to discounted items?")
- Meesho's seller-side rules
- the E-Commerce Rules and Consumer Protection Act
- the Havells product manual
- out-of-scope questions that must be refused

Every live question is also logged to `logs/interactions.jsonl`, and ratings go to `logs/feedback.jsonl`. For screenshots, use the chat UI with the Sources panel open.

## 9. Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
The tests run **offline with no API key and no Ollama**. A deterministic keyword embedding and a scripted fake LLM stand in for real models, and two synthetic fixture documents in `tests/fixtures/` are used (tests only, never the real knowledge base). They cover:
- ingestion and Qdrant indexing
- grounded, cited answers
- reranker ordering, filtering and fallback
- refusing without an LLM call
- history-aware follow-ups and refusal normalisation
- switching to Ollama models, with a separate collection per embedding model
- stripping "thinking" text from local models
- the `/chat` and `/feedback` endpoints

## 10. Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `EMBEDDING_PROVIDER` | `openai` / `openai` | `openai` or `ollama`, each set independently |
| `OPENAI_API_KEY` | (needed for openai) | OpenAI key |
| `OPENAI_CHAT_MODEL` / `OPENAI_EMBEDDING_MODEL` | `gpt-4o-mini` / `text-embedding-3-small` | OpenAI models |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | where Ollama runs (Docker overrides to `host.docker.internal`) |
| `OLLAMA_CHAT_MODEL` / `OLLAMA_EMBEDDING_MODEL` | `qwen3.5:9b` / `bge-m3` | local models |
| `OLLAMA_NUM_CTX` | 16384 | local context window (Ollama's default is too small for reranking) |
| `OLLAMA_REASONING` | `false` | thinking mode for reasoning models (`false` is faster; empty means model default) |
| `QDRANT_URL` | empty → embedded | Qdrant server URL (Docker sets `http://qdrant:6333`) |
| `QDRANT_PATH` / `QDRANT_COLLECTION` | `vectorstore/qdrant` / `ecommerce_support_docs` | storage folder / collection base name (suffixed per embedding model) |
| `CHUNKING_STRATEGY` | `recursive` | or `semantic` (`pip install langchain-experimental`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | passage size and overlap (characters) |
| `FETCH_K` | 20 | candidates pulled from Qdrant before reranking |
| `MIN_RELEVANCE` | 0.20 | loose cosine pre-filter (the reranker does the precision work) |
| `RERANKER` | `llm` | `llm`, `cross-encoder` or `none` |
| `RERANK_LLM_MODEL` | empty → chat model | model for the `llm` reranker (same provider as `LLM_PROVIDER`) |
| `CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` | model for the `cross-encoder` reranker (multilingual) |
| `TOP_K` | 4 | passages sent to the answer model |
| `MIN_RERANK_SCORE` | 0.40 | evidence threshold: raise for stricter refusals, lower if valid questions get refused |
| `HISTORY_TURNS` | 6 | exchanges remembered per conversation |

## 11. Project structure

```
app/
  config.py        settings from .env, refusal message, per-embedding collection name
  models.py        OpenAI / Ollama model factories (chat, rerank, embeddings)
  loaders.py       PDF/TXT loading + source catalog metadata
  chunking.py      recursive / semantic chunking
  vectorstore.py   Qdrant build/load (embedded or server)
  prompts.py       instructions for the clarify and answer steps
  memory.py        per-session conversation history
  reranker.py      LLM / cross-encoder / no-op rerankers
  rag.py           SupportAssistant: clarify → retrieve → rerank → gate → answer → cite
  feedback.py      interaction + feedback logs
  api.py           FastAPI service
ingest.py          build the library (CLI)
chatbot.py         text chat (CLI)
streamlit_app.py   web chat UI with sources + 👍/👎
scripts/           download_documents.py, ensure_index.py, run_sample_conversation.py
logs/              sample_conversation.md/.json (submitted), interaction + feedback logs
screenshots/       sample conversation screenshots (for submission)
data/sources.json  document catalog (titles, public URLs)
tests/             offline pipeline tests
Dockerfile, docker-compose.yml
```

## 12. Constraints honoured
No fine-tuning. No hard-coded answers: the only fixed string is the mandated refusal. Answers are always generated from retrieved passages, and retrieval directly gates whether the model is called at all.

## 13. Limitations and next steps
- Conversation memory lives inside the API process, so a restart clears it. Use Redis for persistence or several API replicas.
- Scanned (image-only) PDFs need OCR before ingestion.
- Small local models (≤4B) follow the citation and refusal rules less reliably. Test with the sample-log script before relying on them.
- Observed in the sample run: "Where can I file a complaint if a defective product is not refunded?" was refused although the Consumer Protection Act covers consumer commissions. The relevant section wasn't ranked into the top passages (the Act is long and its wording differs from the question). Raising `TOP_K`, lowering `MIN_RERANK_SCORE` slightly, or adding hybrid keyword search would likely fix this. The answer to "What rights does a consumer have?" is also incomplete (2 of the 6 rights) for the same reason: only the top passages reach the model.
- Possible improvements: hybrid dense + keyword retrieval (Qdrant sparse vectors), and an evaluation set (e.g. RAGAS) built from the feedback log.
