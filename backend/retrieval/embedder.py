import numpy as np
from fastembed import TextEmbedding

# using fastembed with ONNX backend - no PyTorch dependency
# BAAI/bge-small-en-v1.5: 384-dim, ~130MB, fast on CPU
# loading once at module level - slow first load but fast after that
_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts and return shape (n, 384) float32 array.

    fastembed handles batching internally, we just pass the full list.
    The model runs on CPU via ONNX - no GPU needed, works everywhere.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    embeddings = list(_model.embed(texts))
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (384,)."""
    result = list(_model.embed([query]))
    return np.array(result[0], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between query vector `a` and embedding matrix `b`.
    a: shape (dim,)
    b: shape (n, dim)
    returns: shape (n,)
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    b_normalized = b / b_norms
    return b_normalized @ a_norm
