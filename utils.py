import json
import shutil

from auth import normalize_username
from config import CHAT_DIR, CHAT_FILE, INDEX_DIR, INDEX_FILE, META_FILE, USER_INDEX_DIR


def _chat_path(username: str | None = None):
    if not username:
        return CHAT_FILE
    safe_username = normalize_username(username)
    return CHAT_DIR / f"{safe_username}.json"


def user_index_dir(username: str | None = None):
    if not username:
        return INDEX_DIR
    return USER_INDEX_DIR / normalize_username(username)


def user_index_files(username: str | None = None):
    index_dir = user_index_dir(username)
    return index_dir / "index.faiss", index_dir / "meta.pkl"


def save_chat(messages: list, username: str | None = None) -> None:
    """Persist chat history to disk."""
    path = _chat_path(username)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[utils] Failed to save chat: {e}")


def load_chat(username: str | None = None) -> list:
    """Load chat history from disk. Returns empty list on any error."""
    path = _chat_path(username)
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        # Validate: must be a list of 2-element lists/tuples
        if isinstance(data, list) and all(
            isinstance(item, (list, tuple)) and len(item) == 2 for item in data
        ):
            return [list(item) for item in data]
        return []
    except Exception:
        return []


def clear_session(username: str | None = None, clear_index: bool = False) -> None:
    """Delete chat history, and optionally the FAISS index, from disk."""
    path = _chat_path(username)
    if path.exists():
        path.unlink()

    if clear_index:
        index_dir = user_index_dir(username)
        if index_dir.exists():
            shutil.rmtree(index_dir)


def index_exists(username: str | None = None) -> bool:
    """Return True when a persisted FAISS index and metadata are available."""
    if username:
        index_file, meta_file = user_index_files(username)
        return index_file.exists() and meta_file.exists()
    return INDEX_FILE.exists() and META_FILE.exists()
