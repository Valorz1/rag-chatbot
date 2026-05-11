# The Archive — RAG Chatbot

A local-first retrieval-augmented chatbot with a quietly opinionated UI. It
answers from your own documents when it can, switches to Wikipedia when it
can't, streams tokens as they're generated, and keeps a conversation thread
that actually remembers what you were just talking about.

Everything runs on your machine — your model, your embeddings, your data.

![The Archive — retrieval console](https://img.shields.io/badge/local-first-d4a35a?style=flat-square) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![LlamaIndex](https://img.shields.io/badge/LlamaIndex-blue?style=flat-square) ![ChromaDB](https://img.shields.io/badge/ChromaDB-purple?style=flat-square) ![Ollama](https://img.shields.io/badge/Ollama-black?style=flat-square)

## Features

- **Streaming end-to-end** — tokens render live via Server-Sent Events.
- **Three personas** — *The Archivist*, *The Curator*, *The Analyst*. Switch from the topbar; each changes voice and the framing of fallback answers.
- **Topic awareness** — the topic of conversation is tracked, displayed in the rail, and (optionally) enforced. Pronouns in follow-ups get resolved before retrieval, so "where did he die" actually pulls chunks about the *he* you were just discussing.
- **Score-gated retrieval** — chunks below `SIMILARITY_CUTOFF` are dropped. If nothing survives, we go straight to Wikipedia instead of asking the LLM to hallucinate.
- **`NOT_FOUND` sniffing** — the prompt asks the LLM to reply with a sentinel when the context isn't enough; we detect it mid-stream and silently switch to the Wikipedia fallback, never leaking the token.
- **Real source citations** — every answer surfaces the retrieved chunks (file name + cosine score) in a click-to-expand rail panel.
- **Markdown rendering** — proper headings, lists, code blocks, blockquotes, links — sanitized via DOMPurify.
- **Copy + regenerate** per response. **Export** the whole conversation as Markdown.
- **Keyboard-first**: `Enter` send · `Shift+Enter` newline · `Esc` clear input · `Ctrl/⌘+K` new conversation.
- **Env-var overrides** for every knob in `src/config.py`.
- **No mocks, no cloud, no telemetry.**

## Setup

1. Install [Ollama](https://ollama.com) and pull a model:
   ```
   ollama pull llama3.2:3b
   ```
2. Install Python deps (Python 3.10+):
   ```
   pip install -r requirements.txt
   ```
3. Drop your documents into `data/` — `.txt`, `.md`, `.pdf`, `.docx` all work.
4. Run the server:
   ```
   uvicorn api:app --reload
   ```
5. Open <http://localhost:8000>.

First run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) into
the HuggingFace cache. The server auto-loads any existing index on startup;
click **Build** in the rail to (re)index from `data/`.

## How it works

```
question
   │
   ▼
┌──────────────────────────┐
│ Query rewriter (LLM)     │ ◀──── triggered only when the question is short
│  (resolves "he", "more") │       OR contains anaphoric pronouns; otherwise
└─────────────┬────────────┘       passes through untouched
              ▼
┌──────────────────────────┐
│ Chroma retrieval (top-K) │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Similarity postprocess   │ ◀──── drops chunks below SIMILARITY_CUTOFF
└─────────────┬────────────┘
              ▼
       ┌──────┴──────┐
   chunks?            no chunks
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────────────┐
│ LLM answers  │   │ Wikipedia fallback    │
│ over context │   │ (topic extracted by   │
│ + history    │   │  LLM from rewritten q)│
└──────┬───────┘   └─────────────┬────────┘
       │                          │
   NOT_FOUND? ─── yes ────────────┘
       │
       no
       │
       ▼
   stream tokens
```

The query rewriter is the part most people forget. Without it, a follow-up like
*"where did he die"* gets embedded as four context-free words and you end up
retrieving whatever chunk happens to mention death. The rewriter prepends the
topic + recent turns and asks the LLM to produce a standalone question
*before* anything touches the vector store.

A lightweight pronoun/length heuristic short-circuits the rewriter when the
question is obviously standalone — so switching topics ("tell me about
Napoleon") doesn't accidentally get rewritten into the *old* topic.

## Configuration

Every value in [`src/config.py`](src/config.py) is overridable via environment variable:

| Var | Default | Notes |
| --- | --- | --- |
| `OLLAMA_MODEL` | `llama3.2:3b` | Any Ollama-served model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_TEMPERATURE` | `0.1` | |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace ID |
| `EMBED_DEVICE` | `cpu` | `cuda`, `mps`, etc. |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | |
| `SIMILARITY_TOP_K` | `5` | Chunks retrieved per query |
| `SIMILARITY_CUTOFF` | `0.45` | Floor before falling back to Wikipedia |
| `MAX_HISTORY_TURNS` | `6` | Past turns shown to the LLM |
| `CHROMA_PERSIST_DIR` | `./chromadb_data` | |
| `WIKI_SUMMARY_CHARS` | `800` | Truncation for fallback summaries |

## API

| Method | Path | Body / notes |
| --- | --- | --- |
| `GET` | `/` | Serves the UI |
| `GET` | `/status` | `{status, index_loaded, model, documents, chunks}` |
| `GET` | `/personas` | List of available personas + the default |
| `POST` | `/load-index` | Load the persisted index from disk |
| `POST` | `/build-index` | Drop + rebuild the collection from `data/` |
| `POST` | `/chat` | `{question, history?, stream?, persona?, topic?, stay_on_topic?}` |

With `stream: true`, `/chat` returns SSE events:
- `{type: "delta", text: "..."}` — token deltas
- `{type: "done", sources: [...], used_wikipedia: bool}` — final metadata
- `{type: "error", message: "..."}` — surfaced over the stream so the UI can show it

## Project layout

```
api.py              FastAPI server, lifespan, SSE streaming
src/
  config.py         Env-overridable settings
  indexer.py        Build / load Chroma-backed index (clears old collection on rebuild)
  chatbot.py        Personas, query rewriter, retrieval, score gating, Wikipedia fallback, streaming
  wikipedia_tool.py Wikipedia API wrapper
static/index.html   The Archive — UI (Fraunces + JetBrains Mono, markdown, sources, log)
data/               Drop documents here
chromadb_data/      Persisted vectors (auto-created, gitignored)
```

## License

MIT. Do what you want with it.
