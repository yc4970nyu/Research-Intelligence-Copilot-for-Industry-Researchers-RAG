"""
Tests for embedder and vector store.

Run with:  python tests/test_embedder_and_store.py
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.retrieval.embedder import embed_texts, embed_query, cosine_similarity
from backend.retrieval.vector_store import VectorStore
from backend.ingestion.chunker import Chunk

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "sample_pdfs")


# ------------------------------------------------------------------ #
#  embedder tests
# ------------------------------------------------------------------ #

def test_embed_texts_shape():
    print("\n[CHECK] embed_texts returns correct shape...")
    texts = ["hello world", "transformer attention mechanism", "RAG retrieval"]
    vecs = embed_texts(texts)
    assert vecs.shape == (3, 384), f"Expected (3, 384), got {vecs.shape}"
    assert vecs.dtype == np.float32
    print(f"  PASS - shape={vecs.shape}, dtype={vecs.dtype}")


def test_embed_query_shape():
    print("\n[CHECK] embed_query returns 1D vector of dim 384...")
    q = embed_query("what is multi-head attention")
    assert q.shape == (384,), f"Expected (384,), got {q.shape}"
    assert q.dtype == np.float32
    print(f"  PASS - shape={q.shape}")


def test_embed_texts_empty():
    print("\n[CHECK] embed_texts handles empty input gracefully...")
    vecs = embed_texts([])
    assert vecs.shape[0] == 0
    print("  PASS - empty input returns empty array")


def test_cosine_similarity_range():
    print("\n[CHECK] Cosine similarity values are in [-1, 1]...")
    texts = [
        "transformer self-attention mechanism",
        "convolutional neural network image classification",
        "gradient descent optimization",
    ]
    embs = embed_texts(texts)
    q = embed_query("attention is all you need")
    scores = cosine_similarity(q, embs)
    assert scores.shape == (3,)
    assert all(-1.01 <= float(s) <= 1.01 for s in scores), f"Scores out of range: {scores}"
    print(f"  PASS - scores={[round(float(s), 4) for s in scores]}")


def test_cosine_similarity_ranking():
    print("\n[CHECK] Cosine similarity ranks semantically similar text higher...")
    texts = [
        "The transformer model uses self-attention to process text sequences.",
        "The stock market dropped significantly due to inflation concerns.",
        "Multi-head attention allows the model to attend to different positions.",
    ]
    embs = embed_texts(texts)
    q = embed_query("how does attention work in transformers")
    scores = cosine_similarity(q, embs)

    top_idx = int(np.argmax(scores))
    assert top_idx in [0, 2], f"Expected a transformer-related chunk on top, got index {top_idx}"
    print(f"  PASS - top result index={top_idx}, score={scores[top_idx]:.4f}")


# ------------------------------------------------------------------ #
#  vector store tests
# ------------------------------------------------------------------ #

def _make_chunks(n=5, doc_id="test-doc") -> list[Chunk]:
    import uuid
    sentences = [
        "Transformers use self-attention to model long-range dependencies.",
        "BERT is pre-trained on masked language modeling and next sentence prediction.",
        "GPT uses autoregressive language modeling with a decoder-only architecture.",
        "FlashAttention reduces memory usage by reordering attention computation.",
        "RAG combines retrieval with generation to ground answers in documents.",
    ]
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            filename=f"paper_{i}.pdf",
            page=i + 1,
            text=sentences[i % len(sentences)],
        )
        for i in range(n)
    ]


def test_store_add_and_size():
    print("\n[CHECK] VectorStore correctly tracks size after adding chunks...")
    store = VectorStore()
    assert store.size == 0
    chunks = _make_chunks(5)
    store.add_chunks(chunks)
    assert store.size == 5, f"Expected 5, got {store.size}"
    print(f"  PASS - store size={store.size}")


def test_store_no_duplicate_chunks():
    print("\n[CHECK] Adding same chunks twice doesn't create duplicates...")
    store = VectorStore()
    chunks = _make_chunks(3)
    store.add_chunks(chunks)
    store.add_chunks(chunks)  # add same chunks again
    assert store.size == 3, f"Expected 3, got {store.size} (duplicates added)"
    print("  PASS - no duplicates")


def test_store_search_returns_results():
    print("\n[CHECK] VectorStore search returns ranked results...")
    store = VectorStore()
    chunks = _make_chunks(5)
    store.add_chunks(chunks)

    results = store.search("how does self-attention work in transformers", top_k=3)
    assert len(results) > 0
    assert len(results) <= 3

    # results should be (Chunk, float) pairs
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)

    # scores should be descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by score"
    print(f"  PASS - got {len(results)} results, top score={scores[0]:.4f}")


def test_store_search_empty():
    print("\n[CHECK] Search on empty store returns empty list...")
    store = VectorStore()
    results = store.search("anything")
    assert results == []
    print("  PASS - empty store returns []")


def test_store_clear():
    print("\n[CHECK] VectorStore.clear() wipes all chunks and embeddings...")
    store = VectorStore()
    store.add_chunks(_make_chunks(4))
    assert store.size == 4
    store.clear()
    assert store.size == 0
    assert store.embeddings.shape[0] == 0
    print("  PASS - store cleared")


def test_store_save_and_load():
    print("\n[CHECK] VectorStore can save to disk and reload correctly...")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["INDEX_PATH"] = tmpdir

        store_a = VectorStore()
        chunks = _make_chunks(4)
        store_a.add_chunks(chunks)
        store_a.save()

        store_b = VectorStore()
        success = store_b.load()
        assert success, "load() should return True when index files exist"
        assert store_b.size == 4
        assert store_b.embeddings.shape == store_a.embeddings.shape

    # reset env
    os.environ.pop("INDEX_PATH", None)
    print("  PASS - save/load roundtrip works")


def test_store_search_on_real_pdf():
    print("\n[CHECK] Vector store search on real paper finds relevant chunks...")
    from backend.ingestion.pdf_extractor import extract_text_from_pdf
    from backend.ingestion.chunker import chunk_pages

    path = os.path.join(SAMPLE_DIR, "foundations", "attention_is_all_you_need.pdf")
    pages = extract_text_from_pdf(path)
    chunks = chunk_pages(pages, filename="attention_is_all_you_need.pdf", doc_id="attn-paper")

    store = VectorStore()
    store.add_chunks(chunks)

    results = store.search("multi-head attention query key value projection", top_k=3)
    assert len(results) > 0

    top_text = results[0][0].text.lower()
    assert "attention" in top_text, f"Top result doesn't mention attention: {top_text[:100]}"
    print(f"  PASS - top result score={results[0][1]:.4f}, page={results[0][0].page}")


# ------------------------------------------------------------------ #
#  runner
# ------------------------------------------------------------------ #

def run_all():
    tests = [
        test_embed_texts_shape,
        test_embed_query_shape,
        test_embed_texts_empty,
        test_cosine_similarity_range,
        test_cosine_similarity_ranking,
        test_store_add_and_size,
        test_store_no_duplicate_chunks,
        test_store_search_returns_results,
        test_store_search_empty,
        test_store_clear,
        test_store_save_and_load,
        test_store_search_on_real_pdf,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
