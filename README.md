# Research Intelligence Copilot

A lightweight RAG system for industry researchers. Upload PDF papers, ask questions, get grounded answers with citations.

Built with FastAPI, a hand-rolled hybrid retrieval pipeline (no LangChain, no vector DB), and Claude for generation.

---

## What it does

The main use case is: you have a bunch of research papers or technical reports, and you want to ask questions across them without reading everything. The system retrieves relevant passages and generates answers that cite which page/document the information came from. If the evidence isn't strong enough, it refuses to answer instead of making things up.

Demo scenarios that work well:
- *"What dataset was used to evaluate this model?"* → factual lookup
- *"Which papers in this collection focus on retrieval-augmented generation?"* → cross-doc synthesis
- *"Compare Llama 2, Mistral 7B and Mixtral in a table"* → structured output
- *"hello"* → just a greeting, no retrieval triggered
- *"Which paper proves this is state-of-the-art on all benchmarks?"* → insufficient evidence, system refuses

---

## How to run

```bash
# 1. clone the repo
git clone https://github.com/yc4970nyu/Research-Intelligence-Copilot-for-Industry-Researchers-RAG
cd Research-Intelligence-Copilot-for-Industry-Researchers-RAG

# 2. install dependencies
pip install -r requirements.txt

# 3. set up env
cp .env.example .env
# fill in your ANTHROPIC_API_KEY in .env

# 4. start the server
uvicorn backend.main:app --reload --port 8000

# 5. open the UI
# go to http://localhost:8000 in your browser
```

The UI lets you upload PDFs from the left panel and chat on the right. You can also use the API directly:

```bash
# ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "files=@paper.pdf"

# ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?"}'

# check what is indexed
curl http://localhost:8000/index/stats
```

---

## System architecture

```
User
 │
 ├── POST /ingest ──────────────────────────────────────────────────┐
 │         │                                                         │
 │    [PDF file]                                                     │
 │         │                                                         │
 │    pdf_extractor.py                                              │
 │    - extract text page by page (PyMuPDF)                        │
 │    - handle two-column layout by sorting blocks by x-position   │
 │    - clean noise (page numbers, arXiv IDs, hyphen line breaks)  │
 │         │                                                         │
 │    chunker.py                                                    │
 │    - sliding window (512 chars, 64 overlap)                     │
 │    - break at sentence boundaries not arbitrary character count  │
 │    - skip reference pages                                        │
 │         │                                                         │
 │    vector_store.py                                               │
 │    - embed chunks (fastembed / BAAI/bge-small-en-v1.5)         │
 │    - store in numpy matrix                                       │
 │    - persist to disk (pickle + .npy)                            │
 │                                                                  │
 └── POST /query ───────────────────────────────────────────────────┘
           │
      [user question]
           │
      intent.py  ← detect intent
      ┌─────────────────────────────────────┐
      │ chitchat  → skip retrieval, chat    │
      │ refusal   → return canned message   │
      │ factual   ┐                         │
      │ synthesis ├→ continue to retrieval  │
      │ structured┘                         │
      └─────────────────────────────────────┘
           │
      intent.py  ← rewrite query
      - expand abbreviations (RAG, LLM, RLHF...)
      - remove filler phrases
           │
      hybrid.py
      ┌──────────────────────────────────────┐
      │  BM25 keyword search (from scratch)  │
      │  +                                   │
      │  Semantic search (cosine similarity) │
      │  =                                   │
      │  Reciprocal Rank Fusion (RRF)        │
      └──────────────────────────────────────┘
           │
      reranker.py
      - re-score with direct cosine similarity
      - check similarity threshold (0.65)
      - if below threshold → insufficient evidence
           │
      generator.py
      - pick prompt template by intent type
      - call Claude with numbered context + citation rules
      - post-hoc hallucination filter
           │
      { answer, citations, intent, sufficient_evidence }
```

---

## Design decisions

### PDF extraction

I used PyMuPDF and extract text using the `blocks` mode instead of just `get_text()`. The reason is that almost all ML papers use two-column format, and `get_text()` concatenates blocks in document order which gives completely broken output for two-column papers — you get left column text and right column text interleaved randomly.

With `blocks` mode I get the bounding box of each text block, so I can split by the page midpoint and process left column then right column separately. Not perfect for every layout, but works well for standard academic PDFs.

I also added cleaning for hyphenated line breaks (`atten-\ntion` → `attention`), standalone page number lines, and reference section detection.

### Chunking

512 characters with 64 character overlap. I chose characters instead of tokens because we don't have a tokenizer running locally, and 512 chars is well within the embedding model's limit anyway.

The overlap matters — without it, sentences that fall at chunk boundaries become meaningless fragments. 64 chars of overlap means if a key sentence gets cut, the next chunk still has enough context to be useful.

I also try to break at sentence boundaries (`.`, `!`, `?`, `\n\n`) with a small lookahead instead of cutting at a fixed character count. This makes chunks more coherent, which helps both retrieval and generation quality.

Reference pages are excluded entirely. A reference list entry like `[17] Woosuk Kwon...` adds zero value to retrieval.

### Why hybrid retrieval

Pure semantic search misses exact keyword matches. If someone asks about "FlashAttention-2 IO complexity", semantic search might return vaguely related attention content. BM25 would correctly prioritize exact term matches.

On the other hand, pure BM25 misses semantic similarity. If the query is *"how does the model handle very long sequences"* and the paper says *"we extend the context window using RoPE position interpolation"*, BM25 finds nothing because none of the query words match. Semantic search handles this fine.

So I use both. RRF combines the rankings:

```
rrf_score(doc) = 1/(60 + bm25_rank) + 1/(60 + semantic_rank)
```

The reason I use RRF instead of averaging raw scores is that BM25 scores and cosine similarity scores are on completely different scales — you can't add them directly. RRF only uses rank positions, so it's scale-invariant. It also happens to be very robust empirically, there's a whole paper on it (Cormack et al. 2009).

### Reranking

After RRF I do a second pass and re-score each candidate with direct cosine similarity. This gives us actual confidence values we can threshold on (RRF scores are relative ranks, not absolute similarity measures). Threshold is 0.65 — below that the retrieved content is too semantically distant from the query to generate a grounded answer.

### No external vector DB / search library

The BM25 implementation is written from scratch (standard Okapi BM25 formula, k1=1.5, b=0.75). The vector store is a numpy matrix with brute-force cosine similarity over all chunks. For a few thousand chunks this runs in milliseconds, no need for approximate nearest neighbor indices like HNSW.

### Generation and citations

The system prompt tells the model to cite inline using `[1]`, `[2]`, etc. matching the numbered context blocks, and to not use information outside the provided context. The prompt template varies by intent type — factual queries get a concise answer template, synthesis queries get a cross-document comparison template, structured queries get a markdown table/list template.

The post-hoc hallucination filter extracts content words from each answer sentence and checks how many appear in the retrieved context. If too many sentences seem unsupported it flags the response.

---

## Project structure

```
backend/
  main.py                    ← FastAPI app, lifespan, router registration
  models.py                  ← Pydantic request/response types
  ingestion/
    pdf_extractor.py         ← PyMuPDF, two-column handling, cleaning
    chunker.py               ← sliding window chunker
  retrieval/
    bm25.py                  ← BM25 from scratch
    embedder.py              ← fastembed (BAAI/bge-small-en-v1.5)
    vector_store.py          ← numpy vector store, cosine search, persistence
    hybrid.py                ← RRF fusion of BM25 + semantic
    reranker.py              ← second-pass scoring, evidence threshold
    intent.py                ← intent detection + query rewriting
  generation/
    claude_client.py         ← Anthropic API wrapper with availability check
    prompt_templates.py      ← system prompts per intent type
    generator.py             ← generation pipeline, hallucination filter

frontend/
  index.html / style.css / app.js

Data/sample_pdfs/
  foundations/               ← Attention, BERT, GPT-3, T5, InstructGPT, CoT
  long-context-efficiency/   ← FlashAttention, LongRoPE
  model-reports/             ← LLaMA 1/2/3, Mistral, Mixtral, Claude 3,
                                Gemini 1.5, Phi-3, Qwen2.5
  rag/                       ← RAG paper, RAG survey, Self-RAG

tests/                       ← one test file per module, ~35 checks total
```

---

## Libraries

| Library | Purpose |
|---------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | API framework |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF extraction |
| [fastembed](https://github.com/qdrant/fastembed) | Local ONNX embeddings, no PyTorch needed |
| [numpy](https://numpy.org/) | Vector math and cosine similarity |
| [httpx](https://www.python-httpx.org/) | Anthropic API calls |
| Anthropic Claude API | Generation, intent detection, query rewriting |

No LangChain, LlamaIndex, ChromaDB, FAISS, or any RAG/search framework.

---

## Security

- API keys loaded from `.env`, never in source code
- File uploads: PDF only, 50MB per file limit
- PII patterns in queries trigger refusal before retrieval
- Model is instructed to only use provided context, which limits prompt injection surface
- Uploaded files go to a temp directory and get deleted immediately after extraction

## Known limitations

- BM25 index is rebuilt from the full corpus on every query call. Fine for hundreds of chunks, but would need a persistent inverted index for larger collections.
- The hallucination filter is heuristic — catches obvious cases, not subtle ones.
- Two-column parsing works for standard academic PDFs but will break on scanned documents or unusual layouts.
