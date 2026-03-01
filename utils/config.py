"""Central configuration — all settings from environment with sensible defaults."""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = Path(os.environ.get("LEGAL_QA_DATA_DIR", _ROOT / "data"))
CHROMA_PATH = os.environ.get("LEGAL_QA_CHROMA_PATH", str(_ROOT / "chroma_db"))

# Models (Ollama)
EMBED_MODEL = os.environ.get("LEGAL_QA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LEGAL_QA_LLM_MODEL", "llama3.1")

# Retrieval
TOP_K = int(os.environ.get("LEGAL_QA_TOP_K", "10"))
RERANK_TOP_K = int(os.environ.get("LEGAL_QA_RERANK_TOP_K", "20"))
RERANK_FINAL_K = int(os.environ.get("LEGAL_QA_RERANK_FINAL_K", str(TOP_K)))
HYBRID_ALPHA = float(os.environ.get("LEGAL_QA_HYBRID_ALPHA", "0.7"))

# Chunking
CHUNK_MIN_CHARS = int(os.environ.get("LEGAL_QA_CHUNK_MIN_CHARS", "20"))
CHUNK_MAX_CHARS = int(os.environ.get("LEGAL_QA_CHUNK_MAX_CHARS", "2400"))
CHILD_MAX_CHARS = int(os.environ.get("LEGAL_QA_CHILD_MAX_CHARS", "400"))
CHUNK_OVERLAP = int(os.environ.get("LEGAL_QA_CHUNK_OVERLAP", "100"))

# Session & cache
MAX_HISTORY_TURNS = int(os.environ.get("LEGAL_QA_MAX_HISTORY_TURNS", "5"))
CACHE_TTL_SEC = int(os.environ.get("LEGAL_QA_CACHE_TTL_SEC", "300"))
LOG_REQUESTS_TO_CONSOLE = os.environ.get("LEGAL_QA_LOG_REQUESTS_TO_CONSOLE", "0").lower() in ("1", "true", "yes")
