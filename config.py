# ============================================================
#  config.py - Central configuration for the entire project
# ============================================================
# Edit values here to tune the system; no other file needs changing.

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "meta.pkl"
CHAT_FILE = BASE_DIR / "chat_history.json"
CHAT_DIR = DATA_DIR / "chats"
USER_INDEX_DIR = DATA_DIR / "indexes"
USERS_FILE = DATA_DIR / "users.json"

# Kept as a string for libraries and older code that expect path text.
DATA_PATH = str(INDEX_DIR) + "/"

# Sentence-transformer embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

# Default LLM (can be overridden from the sidebar)
DEFAULT_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
DEFAULT_MODEL = DEFAULT_OPENAI_MODEL if DEFAULT_LLM_PROVIDER == "openai" else DEFAULT_OLLAMA_MODEL

OLLAMA_MODELS = ["mistral", "llama3", "phi3", "gemma"]
OPENAI_MODELS = [DEFAULT_OPENAI_MODEL]

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Max chunks fed into the LLM per request
MAX_CONTEXT_CHUNKS = 20
