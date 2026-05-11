"""FastAPI server for the RAG chatbot."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src import config
from src.chatbot import DEFAULT_PERSONA, PERSONAS, RAGChatbot, build_history
from src.indexer import create_index, load_existing_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag.api")

STATIC_DIR = Path(__file__).parent / "static"


def _collection_stats() -> dict:
    """Return {chunks, documents} from the persisted Chroma collection if it exists."""
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        col = client.get_collection(config.CHROMA_COLLECTION_NAME)
        chunks = col.count()
        metas = col.get(include=["metadatas"]).get("metadatas") or []
        files = {m.get("file_name") for m in metas if m and m.get("file_name")}
        return {"chunks": chunks, "documents": len(files) or None}
    except Exception:
        return {"chunks": 0, "documents": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lock = asyncio.Lock()
    app.state.chatbot = None
    app.state.index_loaded = False

    try:
        index = load_existing_index()
        if index is not None:
            app.state.chatbot = RAGChatbot(index, use_wikipedia=True)
            app.state.index_loaded = True
            log.info("Loaded existing index on startup")
        else:
            log.info("No existing index found; build via POST /build-index")
    except Exception as e:
        log.warning("Failed to auto-load index: %s", e)

    yield


app = FastAPI(title="RAG Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[Message] = Field(default_factory=list)
    stream: bool = False
    persona: str = DEFAULT_PERSONA
    topic: str = ""
    stay_on_topic: bool = False


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/personas")
def personas():
    return {
        "default": DEFAULT_PERSONA,
        "personas": [
            {"id": k, "name": v["name"], "blurb": v["blurb"]} for k, v in PERSONAS.items()
        ],
    }


@app.get("/status")
def status():
    stats = _collection_stats() if app.state.index_loaded else {"chunks": 0, "documents": 0}
    return {
        "status": "running",
        "index_loaded": app.state.index_loaded,
        "model": config.OLLAMA_MODEL,
        **stats,
    }


@app.post("/load-index")
async def load_index_endpoint():
    async with app.state.lock:
        index = await asyncio.to_thread(load_existing_index)
        if index is None:
            return {"success": False, "message": "No existing index found"}
        app.state.chatbot = RAGChatbot(index, use_wikipedia=True)
        app.state.index_loaded = True
        return {"success": True, "message": "Index loaded", **_collection_stats()}


@app.post("/build-index")
async def build_index_endpoint():
    async with app.state.lock:
        try:
            index = await asyncio.to_thread(create_index)
        except Exception as e:
            log.exception("Index build failed")
            raise HTTPException(status_code=500, detail=f"Index build failed: {e}")

        if index is None:
            return {"success": False, "message": "Failed to build index (no documents?)"}
        app.state.chatbot = RAGChatbot(index, use_wikipedia=True)
        app.state.index_loaded = True
        return {"success": True, "message": "Index built", **_collection_stats()}


@app.post("/chat")
async def chat(req: ChatRequest):
    chatbot: RAGChatbot | None = app.state.chatbot
    if chatbot is None:
        raise HTTPException(status_code=409, detail="Index not loaded")

    history = build_history([m.model_dump() for m in req.history])
    kw = {"persona": req.persona, "topic": req.topic, "stay_on_topic": req.stay_on_topic}

    if not req.stream:
        answer = await asyncio.to_thread(chatbot.query, req.question, history, **kw)
        return answer.to_dict()

    def event_stream():
        sources: list = []
        used_wiki = False
        try:
            for delta, _sources, _used_wiki in chatbot.stream(req.question, history, **kw):
                sources = _sources
                used_wiki = _used_wiki
                yield f"data: {json.dumps({'type': 'delta', 'text': delta})}\n\n"
            payload = {
                "type": "done",
                "sources": [s.to_dict() for s in sources],
                "used_wikipedia": used_wiki,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            log.exception("Stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
