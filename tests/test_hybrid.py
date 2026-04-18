"""
Tests for hybrid retrieval (BM25 + semantic via RRF).

Run with:  python tests/test_hybrid.py
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.ingestion.chunker import Chunk
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.hybrid import hybrid_search

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "sample_pdfs")


def _make_store(texts: list[str], doc_id: str = "test") -> VectorStore:
    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            filename="test.pdf",
            page=i + 1,
            text=t,
        )
        for i, t in enumerate(texts)
    ]
    store = VectorStore()
    store.add_chunks(chunks)
    return store


def test_hybrid_empty_store():
    print("\n[CHECK] Hybrid search on empty store returns []...")
    store = VectorStore()
    results = hybrid_search("anything", store, top_k=5)
    assert results == []
    print("  PASS - empty store returns []")


def test_hybrid_returns_top_k():
    print("\n[CHECK] Hybrid search respects top_k limit...")
    texts = [f"research paper about topic {i} with neural networks" for i in range(20)]
    store = _make_store(texts)
    results = hybrid_search("neural network research", store, top_k=5)
    assert len(results) <= 5
    print(f"  PASS - got {len(results)} results")


def test_hybrid_scores_descending():
    print("\n[CHECK] RRF scores are sorted descending...")
    texts = [
        "Transformers use self-attention to model sequences.",
        "BERT uses bidirectional encoder representations.",
        "Dogs are friendly domestic animals.",
        "The stock market fluctuates based on investor sentiment.",
        "Attention mechanisms help models focus on relevant tokens.",
    ]
    store = _make_store(texts)
    results = hybrid_search("attention transformer model", store, top_k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Scores must be descending"
    print(f"  PASS - scores descending: {[round(s,5) for s in scores]}")


def test_hybrid_relevant_over_irrelevant():
    print("\n[CHECK] Hybrid search ranks relevant docs above irrelevant ones...")
    texts = [
        "Multi-head attention allows the model to attend to different subspaces.",
        "The Eiffel Tower is located in Paris, France.",
        "Scaled dot-product attention computes query key value interactions.",
        "Bananas are rich in potassium and natural sugars.",
        "Self-attention enables transformers to process sequences in parallel.",
    ]
    store = _make_store(texts)
    results = hybrid_search("attention mechanism in transformers", store, top_k=3)

    top_texts = [chunk.text for chunk, _ in results]
    relevant_keywords = ["attention", "transformer", "query", "self-attention"]

    # at least 2 of top 3 should contain an attention-related keyword
    hits = sum(
        1 for t in top_texts
        if any(kw in t.lower() for kw in relevant_keywords)
    )
    assert hits >= 2, f"Only {hits}/3 top results are relevant"
    print(f"  PASS - {hits}/3 top results contain relevant keywords")


def test_hybrid_cross_document():
    print("\n[CHECK] Hybrid search pulls from correct document for each query...")
    from backend.ingestion.pdf_extractor import extract_text_from_pdf
    from backend.ingestion.chunker import chunk_pages

    store = VectorStore()
    for fname, doc_id in [
        ("foundations/attention_is_all_you_need.pdf", "attn"),
        ("rag/rag_original.pdf", "rag"),
    ]:
        path = os.path.join(SAMPLE_DIR, fname)
        pages = extract_text_from_pdf(path)
        chunks = chunk_pages(pages, filename=fname, doc_id=doc_id)
        store.add_chunks(chunks)

    # attention-specific query should pull from attn paper
    r1 = hybrid_search("multi-head attention query key value", store, top_k=3)
    top_doc = r1[0][0].doc_id
    assert top_doc == "attn", f"Expected 'attn', got '{top_doc}'"

    # RAG-specific query should pull from rag paper
    r2 = hybrid_search("retrieval augmented generation open domain QA", store, top_k=3)
    top_doc2 = r2[0][0].doc_id
    assert top_doc2 == "rag", f"Expected 'rag', got '{top_doc2}'"

    print(f"  PASS - cross-document routing works correctly")


def run_all():
    tests = [
        test_hybrid_empty_store,
        test_hybrid_returns_top_k,
        test_hybrid_scores_descending,
        test_hybrid_relevant_over_irrelevant,
        test_hybrid_cross_document,
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
