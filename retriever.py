"""
retriever.py :-

Loads the FAISS index from disk and performs semantic search.

Usage:
    from retriever import Retriever
    r = Retriever()
    results = r.search("What is the proposed method?", top_k=5)
"""

import pickle
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi

from config import DATA_PATH
from embeddings import load_embedding_model
import faiss


class Retriever:
    def __init__(self, index_dir: str | Path | None = None):
        index_dir = Path(index_dir) if index_dir else Path(DATA_PATH)
        self.model = load_embedding_model()
        self.index = faiss.read_index(str(index_dir / "index.faiss"))

        with (index_dir / "meta.pkl").open("rb") as f:
            self.chunks = pickle.load(f)

        # BM25 for keyword search
        self.tokenized_chunks = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        # Pre-cache per-paper embeddings (avoid recomputing on every compare query)
        self._paper_chunks: dict[str, list] = {}
        self._paper_embeddings: dict[str, np.ndarray] = {}

        for c in self.chunks:
            self._paper_chunks.setdefault(c["source"], []).append(c)

        for source, src_chunks in self._paper_chunks.items():
            texts = [c["text"] for c in src_chunks]
            self._paper_embeddings[source] = self.model.encode(texts).astype("float32")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Rank results with reciprocal rank fusion across semantic and keyword search."""
        if not query.strip() or k <= 0 or not self.chunks:
            return []

        terms = re.findall(r"[\w]+", query.lower())
        expanded_terms = list(terms)
        lowered = query.lower()
        if any(term in lowered for term in ("where", "location", "address", "venue", "center", "centre")):
            expanded_terms.extend(["venue", "address", "location", "center", "centre"])
        if any(term in lowered for term in ("instruction", "rule", "allowed", "bring", "must", "should")):
            expanded_terms.extend(["instructions", "must", "should", "bring", "allowed", "prohibited"])

        candidate_k = min(len(self.chunks), max(k * 4, 20))
        q_emb = self.model.encode([query]).astype("float32")
        distances, semantic_indices = self.index.search(q_emb, candidate_k)
        bm25_scores = self.bm25.get_scores(expanded_terms)
        keyword_indices = np.argsort(bm25_scores)[-candidate_k:][::-1]

        # Reciprocal rank fusion combines semantic similarity and exact terms
        # without letting the scale of either score dominate. This helps
        # queries such as "center location" retrieve a PDF field named
        # "Exam Venue Address".
        fused_scores: dict[int, float] = {}
        for rank, idx in enumerate(semantic_indices[0], start=1):
            if idx >= 0:
                fused_scores[int(idx)] = fused_scores.get(int(idx), 0.0) + 1 / (60 + rank)
        for rank, idx in enumerate(keyword_indices, start=1):
            fused_scores[int(idx)] = fused_scores.get(int(idx), 0.0) + 1 / (60 + rank)

        if any(term in lowered for term in ("where", "location", "address", "venue", "center", "centre")):
            for idx in fused_scores:
                text = self.chunks[idx]["text"].lower()
                if "exam venue address" in text or "venue address" in text:
                    fused_scores[idx] += 0.03

        ranked_indices = sorted(
            fused_scores,
            key=lambda idx: (fused_scores[idx], float(bm25_scores[idx])),
            reverse=True,
        )

        seen = set()
        final = []
        for idx in ranked_indices:
            c = self.chunks[idx].copy()
            key = (c["source"], c["page"], " ".join(c["text"].lower().split()))
            if key not in seen:
                seen.add(key)
                c["score"] = fused_scores[idx]
                final.append(c)
            if len(final) >= k:
                break

        return final

    def search_per_paper(self, query: str, k: int = 3) -> dict[str, list[dict]]:
        """Top-k chunks per paper using pre-cached embeddings."""
        if not query.strip() or k <= 0:
            return {source: [] for source in self._paper_chunks}
        q_emb = self.model.encode([query]).astype("float32")
        results = {}

        for source, embs in self._paper_embeddings.items():
            scores = np.dot(embs, q_emb.T).flatten()
            top_idx = np.argsort(scores)[-k:][::-1]
            results[source] = [self._paper_chunks[source][i] for i in top_idx]

        return results
