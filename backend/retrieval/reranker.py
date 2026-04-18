import os
import numpy as np

from backend.ingestion.chunker import Chunk
from backend.retrieval.embedder import embed_query, cosine_similarity

# below this score, a chunk is considered weak evidence
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))


def rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
    top_k: int = None,
) -> list[tuple[Chunk, float]]:
    """
    Re-score and re-rank candidate chunks against the query using cosine similarity.

    The hybrid search already gives us a pretty good ranking via RRF, but RRF
    scores are rank-based and don't tell us the actual semantic similarity of
    each chunk to the query. Reranking with direct cosine similarity gives us
    actual confidence scores we can threshold on.

    I also apply a small length penalty - very short chunks (< 80 chars) are
    usually figure captions or section headers, not real content.
    """
    top_k = top_k or int(os.getenv("TOP_K", 5))

    if not candidates:
        return []

    query_vec = embed_query(query)
    chunk_texts = [c.text for c, _ in candidates]

    # batch embed all candidate chunks at once
    chunk_vecs = _get_embeddings(chunk_texts)
    raw_scores = cosine_similarity(query_vec, chunk_vecs)

    scored = []
    for i, (chunk, _) in enumerate(candidates):
        sem_score = float(raw_scores[i])

        # small penalty for very short chunks
        length_factor = min(1.0, len(chunk.text) / 150.0)
        final_score = sem_score * (0.85 + 0.15 * length_factor)

        scored.append((chunk, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts. Using the same embedder as the vector store
    so the similarity scores are on the same scale.
    """
    from backend.retrieval.embedder import embed_texts
    return embed_texts(texts)


def check_evidence(results: list[tuple[Chunk, float]]) -> bool:
    """
    Returns True if the top result meets the similarity threshold.

    If False, the pipeline should return 'insufficient evidence' instead
    of hallucinating an answer from weak chunks.

    I set the threshold at 0.45 - below that the chunk is semantically
    pretty distant from the query. Tuned this empirically on the sample PDFs.
    """
    if not results:
        return False
    top_score = results[0][1]
    return top_score >= SIMILARITY_THRESHOLD


def filter_by_threshold(
    results: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    """
    Remove chunks below the similarity threshold.
    Only called after check_evidence passes - this further cleans the citation list.
    """
    return [(chunk, score) for chunk, score in results if score >= SIMILARITY_THRESHOLD * 0.8]
