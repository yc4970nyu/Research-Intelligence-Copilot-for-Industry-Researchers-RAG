import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from backend.retrieval.vector_store import get_store
from backend.routers import ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load existing index from disk on startup so we don't lose data between restarts
    store = get_store()
    loaded = store.load()
    if loaded:
        print(f"[startup] Loaded index: {store.size} chunks")
    else:
        print("[startup] No existing index found, starting fresh")
    yield
    # nothing special needed on shutdown


app = FastAPI(
    title="Research Intelligence Copilot",
    description="RAG pipeline for industry researchers - hybrid retrieval over PDF documents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, tags=["ingestion"])
app.include_router(query.router, tags=["query"])


@app.get("/health")
def health():
    store = get_store()
    return {"status": "ok", "indexed_chunks": store.size}


# serve frontend as static files - must come last so API routes take priority
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
