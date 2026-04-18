from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Research Intelligence Copilot",
    description="RAG pipeline for industry researchers",
    version="0.1.0",
)

# allow frontend to call the api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# placeholder - routers will be registered here later
@app.get("/health")
def health():
    return {"status": "ok"}
