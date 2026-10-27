from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import MODEL_NAME


class EmbeddingModelError(RuntimeError):
    """Raised when the sentence-transformer embedding model cannot load."""


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        try:
            return SentenceTransformer(MODEL_NAME)
        except Exception as exc:
            raise EmbeddingModelError(
                "Could not load the embedding model "
                f"'{MODEL_NAME}'. If this is the first run, connect to the internet "
                "once so sentence-transformers can download it, then restart the app."
            ) from exc
