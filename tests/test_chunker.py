"""
Tests for the chunker module.

Run with:  python tests/test_chunker.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.ingestion.pdf_extractor import extract_text_from_pdf
from backend.ingestion.chunker import chunk_pages

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "sample_pdfs")


def test_chunker_basic():
    print("\n[CHECK] Chunker produces chunks from a normal PDF...")
    path = os.path.join(SAMPLE_DIR, "foundations", "attention_is_all_you_need.pdf")
    pages = extract_text_from_pdf(path)
    chunks = chunk_pages(pages, filename="attention_is_all_you_need.pdf", doc_id="test-001")

    assert len(chunks) > 0, "Should produce at least one chunk"
    print(f"  PASS - got {len(chunks)} chunks")


def test_chunk_ids_unique():
    print("\n[CHECK] All chunk_ids are unique across multiple documents...")
    all_ids = []
    for fname, doc_id in [
        ("foundations/attention_is_all_you_need.pdf", "doc-1"),
        ("model-reports/mistral_7b.pdf", "doc-2"),
        ("rag/rag_original.pdf", "doc-3"),
    ]:
        path = os.path.join(SAMPLE_DIR, fname)
        pages = extract_text_from_pdf(path)
        chunks = chunk_pages(pages, filename=fname, doc_id=doc_id)
        all_ids.extend(c.chunk_id for c in chunks)

    assert len(all_ids) == len(set(all_ids)), "Chunk IDs must be globally unique"
    print(f"  PASS - {len(all_ids)} chunks, all IDs unique")


def test_no_empty_chunks():
    print("\n[CHECK] No chunk has empty or whitespace-only text...")
    path = os.path.join(SAMPLE_DIR, "model-reports", "llama2.pdf")
    pages = extract_text_from_pdf(path)
    chunks = chunk_pages(pages, filename="llama2.pdf", doc_id="doc-4")

    bad = [c for c in chunks if not c.text or not c.text.strip()]
    assert len(bad) == 0, f"Found {len(bad)} empty chunks"
    print(f"  PASS - {len(chunks)} chunks, none empty")


def test_chunk_size_reasonable():
    print("\n[CHECK] Chunk sizes stay within reasonable bounds...")
    path = os.path.join(SAMPLE_DIR, "foundations", "gpt3.pdf")
    pages = extract_text_from_pdf(path)
    chunks = chunk_pages(pages, filename="gpt3.pdf", doc_id="doc-5")

    max_len = max(len(c.text) for c in chunks)
    # allow some slack for the lookahead window - hard cap at 2x chunk_size
    assert max_len < 1200, f"Some chunk is too large: {max_len} chars"
    print(f"  PASS - max chunk size: {max_len} chars")


def test_reference_pages_skipped():
    print("\n[CHECK] Reference section pages are excluded from chunks...")
    # mistral_7b last page is a known reference page
    path = os.path.join(SAMPLE_DIR, "model-reports", "mistral_7b.pdf")
    pages = extract_text_from_pdf(path)
    chunks = chunk_pages(pages, filename="mistral_7b.pdf", doc_id="doc-6")

    # none of the chunks should contain the pattern [17] Woosuk... from the ref page
    ref_pattern = "[17] Woosuk"
    found = [c for c in chunks if ref_pattern in c.text]
    assert len(found) == 0, "Reference page content leaked into chunks"
    print("  PASS - reference pages excluded")


def test_chunk_metadata():
    print("\n[CHECK] Each chunk has correct metadata fields...")
    path = os.path.join(SAMPLE_DIR, "rag", "self_rag.pdf")
    pages = extract_text_from_pdf(path)
    chunks = chunk_pages(pages, filename="self_rag.pdf", doc_id="my-doc-id")

    for c in chunks:
        assert c.doc_id == "my-doc-id"
        assert c.filename == "self_rag.pdf"
        assert isinstance(c.page, int) and c.page >= 1
        assert isinstance(c.chunk_id, str) and len(c.chunk_id) > 0

    print(f"  PASS - all {len(chunks)} chunks have valid metadata")


def run_all():
    tests = [
        test_chunker_basic,
        test_chunk_ids_unique,
        test_no_empty_chunks,
        test_chunk_size_reasonable,
        test_reference_pages_skipped,
        test_chunk_metadata,
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
