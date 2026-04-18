import os
from fastapi import APIRouter, HTTPException

from backend.models import QueryRequest, QueryResponse, Chunk as ChunkModel
from backend.retrieval.intent import detect_intent, needs_search, rewrite_query
from backend.retrieval.vector_store import get_store
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.reranker import rerank, check_evidence, filter_by_threshold
from backend.generation.generator import generate
from backend.retrieval.intent import INTENT_REFUSAL

router = APIRouter()

TOP_K = int(os.getenv("TOP_K", 5))


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main query endpoint. Full pipeline:

      1. detect intent (chitchat / factual / synthesis / structured / refusal)
      2. if needs search: rewrite query, hybrid retrieve, rerank
      3. check evidence threshold
      4. generate answer with citations (or return canned response)
      5. return answer + citations + metadata
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # --- step 1: intent detection ---
    intent = detect_intent(question)

    # --- step 2: retrieval (skip for chitchat and refusal) ---
    rewritten = None
    final_chunks = []
    sufficient = True

    if needs_search(intent):
        store = get_store()

        if store.size == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents have been ingested yet. Please upload PDFs first."
            )

        rewritten = rewrite_query(question)
        search_query = rewritten or question

        # hybrid retrieval: BM25 + semantic, fused with RRF
        candidates = hybrid_search(search_query, store, top_k=TOP_K * 3)

        # rerank with direct cosine similarity
        reranked = rerank(search_query, candidates, top_k=TOP_K)

        # evidence check - if top chunk is below threshold, refuse to answer
        sufficient = check_evidence(reranked)

        if sufficient:
            final_chunks = filter_by_threshold(reranked)
        else:
            final_chunks = []

    # --- step 3: generation ---
    result = generate(
        query=question,
        intent=intent,
        chunks=final_chunks,
        sufficient_evidence=sufficient,
    )

    # --- step 4: build response ---
    citation_models = [
        ChunkModel(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            page=chunk.page,
            text=chunk.text,
            score=score,
        )
        for chunk, score in result.citations
    ]

    return QueryResponse(
        answer=result.answer,
        citations=citation_models,
        rewritten_query=rewritten,
        intent=intent,
        sufficient_evidence=result.sufficient_evidence,
    )
