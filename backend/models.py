from pydantic import BaseModel
from typing import Optional, List


# one chunk that comes out of the retrieval step
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    score: float  # final combined score after reranking


# request body for /query
class QueryRequest(BaseModel):
    question: str


# what /query returns back to the frontend
class QueryResponse(BaseModel):
    answer: str
    citations: List[Chunk]
    rewritten_query: Optional[str] = None
    intent: Optional[str] = None
    sufficient_evidence: bool


# what /ingest returns
class IngestResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks: int
