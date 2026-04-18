import os
import pickle
import numpy as np

from backend.ingestion.chunker import Chunk
from backend.retrieval.embedder import embed_texts, embed_query, cosine_similarity

# where we save the index between restarts
INDEX_PATH = os.getenv("INDEX_PATH", "backend/index")


class VectorStore:
    """
    In-memory vector store backed by numpy arrays.
    No external vector DB - just a matrix of embeddings + a list of chunks.

    Search is brute-force cosine similarity over the full matrix.
    For a few thousand chunks this is totally fine, maybe up to ~100k too.
    If we needed to scale to millions of chunks we'd need something like HNSW,
    but for this use case it's overkill.

    Index is persisted to disk with pickle so we don't have to re-embed
    every time the server restarts.
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        # embedding matrix, shape (n_chunks, embed_dim)
        self.embeddings: np.ndarray = np.empty((0,), dtype=np.float32)
        self._loaded = False

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Embed a list of chunks and add them to the store.
        Skips chunks that are already indexed (same chunk_id).
        """
        existing_ids = {c.chunk_id for c in self.chunks}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            return

        texts = [c.text for c in new_chunks]
        new_embeddings = embed_texts(texts)

        self.chunks.extend(new_chunks)

        if self.embeddings.shape[0] == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Semantic search: embed the query, compute cosine similarity with all
        stored chunks, return top_k results as (chunk, score) pairs.
        """
        if len(self.chunks) == 0:
            return []

        q_vec = embed_query(query)
        scores = cosine_similarity(q_vec, self.embeddings)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def save(self) -> None:
        """Persist the index to disk."""
        os.makedirs(INDEX_PATH, exist_ok=True)
        with open(os.path.join(INDEX_PATH, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        np.save(os.path.join(INDEX_PATH, "embeddings.npy"), self.embeddings)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        chunks_path = os.path.join(INDEX_PATH, "chunks.pkl")
        emb_path = os.path.join(INDEX_PATH, "embeddings.npy")

        if not os.path.exists(chunks_path) or not os.path.exists(emb_path):
            return False

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.embeddings = np.load(emb_path)
        self._loaded = True
        return True

    def clear(self) -> None:
        """Wipe everything - useful for re-ingestion."""
        self.chunks = []
        self.embeddings = np.empty((0,), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.chunks)


# singleton - one store shared across the whole app
_store = VectorStore()


def get_store() -> VectorStore:
    return _store
