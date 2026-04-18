import os
import uuid
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from backend.ingestion.pdf_extractor import extract_text_from_pdf
from backend.ingestion.chunker import chunk_pages
from backend.retrieval.vector_store import get_store
from backend.models import IngestResponse

router = APIRouter()

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files for ingestion.

    Each file is:
      1. validated (PDF only, size limit)
      2. text-extracted page by page
      3. chunked with sliding window
      4. embedded and added to the vector store
      5. indexed in BM25 (happens at query time, not here)

    The vector store is persistent - re-ingesting the same file
    won't create duplicates because chunk_ids are content-addressed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    store = get_store()
    processed = []
    total_chunks = 0

    for upload in files:
        # only accept PDFs
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename} is not a PDF file"
            )

        content = await upload.read()

        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename} exceeds {MAX_FILE_SIZE_MB}MB limit"
            )

        # write to a temp file so pymupdf can open it
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            doc_id = str(uuid.uuid4())
            pages = extract_text_from_pdf(tmp_path)

            if not pages:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not extract text from {upload.filename}"
                )

            chunks = chunk_pages(
                pages,
                filename=upload.filename,
                doc_id=doc_id,
            )

            store.add_chunks(chunks)
            store.save()

            processed.append(upload.filename)
            total_chunks += len(chunks)

        finally:
            os.unlink(tmp_path)

    return IngestResponse(
        message=f"Successfully ingested {len(processed)} file(s)",
        files_processed=processed,
        total_chunks=total_chunks,
    )


@router.get("/index/stats")
def index_stats():
    """Quick check on how many chunks are currently indexed."""
    store = get_store()
    docs = {}
    for chunk in store.chunks:
        docs[chunk.filename] = docs.get(chunk.filename, 0) + 1
    return {
        "total_chunks": store.size,
        "documents": docs,
    }


@router.delete("/index")
def clear_index():
    """Wipe the entire index. Useful for re-ingesting from scratch."""
    store = get_store()
    store.clear()
    store.save()
    return {"message": "Index cleared"}
