"""Centralized configuration. Values can be overridden via environment variables."""

import os
from pathlib import Path

# Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chromadb_data")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))

# Retrieval
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
# Cosine-similarity floor for a chunk to count as "relevant".
# Below this, we treat the local index as not knowing and fall back to Wikipedia.
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.45"))

# Chat
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# Wikipedia
WIKI_USER_AGENT = os.getenv("WIKI_USER_AGENT", "RAGChatbot/1.0 (https://example.com)")
WIKI_SUMMARY_CHARS = int(os.getenv("WIKI_SUMMARY_CHARS", "800"))
