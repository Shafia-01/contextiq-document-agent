from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import os
from ingest import extract_documents
from vectorstore import InMemoryVectorStore
from qa import QAEngine, GroqModel, GeminiModel, download_pdf  

"""
FastAPI entrypoint for the ContextIQ backend.

Responsibilities:
- Accept file uploads and delegate to the ingestion pipeline.
- Orchestrate the QAEngine and expose a stable JSON contract to the frontend.
- Provide ArXiv integration endpoints for paper discovery and ingestion.

Non-trivial behaviours:
- LLM model selection is **per-request** (via the ``model`` form field) and is
  switched on the shared QAEngine instance. In a real multi-tenant deployment
  you would likely want per-session engines instead.
- The ``/ask`` endpoint returns structured answers (including document + page
  attribution and a confidence heuristic) as produced by ``QAEngine.ask``.
"""

app = FastAPI(title="DocuAgent API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve the uploads directory relative to this file so uploads work
# even if uvicorn is started from a different working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vectorstore = InMemoryVectorStore()

# Resolve the default LLM provider once at startup. This keeps Gemini as the
# default (per assessment requirements) but allows overriding via LLM_PROVIDER.
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
llm_instances = {"groq": GroqModel(), "gemini": GeminiModel()}
default_llm = llm_instances.get(DEFAULT_PROVIDER, llm_instances["gemini"])

qa_engine = QAEngine(vectorstore=vectorstore, llm=default_llm)


def save_uploaded_files(files: List[UploadFile]) -> List[str]:
    """
    Persist uploaded files to disk and return their paths.

    This keeps FastAPI's upload handling thin and makes the ingestion pipeline
    reusable from the CLI as well.
    """
    paths = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        paths.append(str(file_path))
    return paths


@app.get("/health")
def health():
    """Simple readiness probe used by the Streamlit frontend."""
    return {"status": "ok"}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents and ingest them into the vector store.

    Returns the number of chunks added so that a caller can reason about
    ingestion cost and completeness.
    """
    try:
        paths = save_uploaded_files(files)
        chunks = extract_documents(paths)
        qa_engine.add_documents(chunks)
        return {"message": f"Uploaded {len(files)} files", "chunks_added": len(chunks)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
async def ask_question(
    query: str = Form(...),
    model: str = Form("gemini"),
    top_k: int = Form(20)
):
    """
    Ask a question over the ingested document corpus.

    - ``model`` chooses the underlying LLM provider (Groq or Gemini).
    - ``top_k`` controls how many chunks are retrieved from the vector store.
    """
    try:
        if model not in llm_instances:
            return JSONResponse(
                status_code=400,
                content={"error": f"Model must be one of {list(llm_instances.keys())}"},
            )
        # Switch LLM implementation for this request; the vector store and
        # metadata remain unchanged.
        qa_engine.llm = llm_instances[model]
        answer = qa_engine.ask(query, top_k=top_k)
        return {"query": query, "answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/arxiv_search")
async def arxiv_search(
    query: str = Form(...),
    model: str = Form("groq"),
    max_papers: int = Form(3),
    action: str = Form("list"), 
    top_k: int = Form(10)
):
    """
    Light-weight wrapper around the ArXiv API.

    - ``action='list'``: return metadata only.
    - ``action='download'``: fetch PDFs and save them locally.
    - ``action='ingest'``: download and pass the PDFs through the same
      ingestion pipeline as user uploads.
    """
    try:
        if model not in llm_instances:
            return JSONResponse(
                status_code=400,
                content={"error": f"Model must be one of {list(llm_instances.keys())}"},
            )
        qa_engine.llm = llm_instances[model]

        papers = qa_engine.search_and_list_arxiv(query, max_papers=max_papers)

        if action == "list":
            return {"query": query, "papers": papers, "note": "Use action='download' or 'ingest' to proceed."}

        elif action == "download":
            saved_files = []
            for p in papers:
                if p.get("pdf_url"):
                    pdf_path = download_pdf(p["pdf_url"])
                    saved_files.append(pdf_path)
            return {"query": query, "downloaded_files": saved_files}

        elif action == "ingest":
            added_chunks = qa_engine.ingest_papers(papers)
            return {"query": query, "chunks_added": added_chunks, "note": "Papers ingested into QAEngine."}

        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid action. Must be one of: list, download, ingest"},
            )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})