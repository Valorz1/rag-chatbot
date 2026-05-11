"""Document indexing: build / load a Chroma-backed vector index."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src import config

log = logging.getLogger(__name__)

_settings_configured = False


def setup_llm_and_embeddings() -> None:
    """Configure LlamaIndex global Settings. Idempotent."""
    global _settings_configured
    if _settings_configured:
        return

    Settings.llm = Ollama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=config.OLLAMA_TIMEOUT,
        temperature=config.OLLAMA_TEMPERATURE,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL,
        device=config.EMBED_DEVICE,
    )
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    _settings_configured = True


def _chroma_client() -> chromadb.PersistentClient:
    Path(config.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def _load_documents(data_path: str):
    path = Path(data_path)
    if not path.exists():
        log.warning("Data directory %s does not exist", data_path)
        return None
    files = [p for p in path.iterdir() if p.is_file()]
    if not files:
        log.warning("No files found in %s", data_path)
        return None
    log.info("Loading %d file(s) from %s", len(files), data_path)
    # exclude_hidden=False so paths containing a dotted component (e.g. running inside a
    # `.claude/worktrees/...` dir) don't get filtered out.
    docs = SimpleDirectoryReader(data_path, exclude_hidden=False).load_data()
    log.info("Parsed %d document chunks", len(docs))
    return docs


def create_index(data_path: str | None = None, *, reset: bool = True) -> VectorStoreIndex | None:
    """Build (or rebuild) the index from the data directory.

    When ``reset`` is true (default), the existing Chroma collection is dropped first
    so re-running does not duplicate documents.
    """
    data_path = data_path or config.DATA_DIR
    setup_llm_and_embeddings()

    documents = _load_documents(data_path)
    if not documents:
        return None

    client = _chroma_client()
    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION_NAME)
            log.info("Dropped existing collection %s", config.CHROMA_COLLECTION_NAME)
        except (ValueError, Exception):  # chroma raises NotFoundError or ValueError depending on version
            pass

    collection = client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    log.info("Embedding %d documents", len(documents))
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    log.info("Index built (%d vectors)", collection.count())
    return index


def load_existing_index() -> VectorStoreIndex | None:
    """Load a previously persisted index, or return None if none exists."""
    chroma_path = Path(config.CHROMA_PERSIST_DIR)
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        return None

    setup_llm_and_embeddings()

    client = _chroma_client()
    try:
        collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
    except Exception:
        return None

    if collection.count() == 0:
        return None

    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)
