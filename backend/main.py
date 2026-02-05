from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
from ingest import extract_documents
from vectorstore import InMemoryVectorStore
from qa import QAEngine, GroqModel, GeminiModel, download_pdf  

app = FastAPI(title="DocuAgent API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vectorstore = InMemoryVectorStore()
llm_instances = {"groq": GroqModel(), "gemini": GeminiModel()}

qa_engine = QAEngine(vectorstore=vectorstore, llm=llm_instances["groq"])

def save_uploaded_files(files: List[UploadFile]) -> List[str]:
    paths = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        paths.append(str(file_path))
    return paths

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
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
    model: str = Form("groq"),
    top_k: int = Form(10)
):
    try:
        if model not in llm_instances:
            return JSONResponse(status_code=400, content={"error": f"Model must be one of {list(llm_instances.keys())}"})
        qa_engine.llm = llm_instances[model]  # switch model dynamically
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
    try:
        if model not in llm_instances:
            return JSONResponse(status_code=400, content={"error": f"Model must be one of {list(llm_instances.keys())}"})
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
            return JSONResponse(status_code=400, content={"error": "Invalid action. Must be one of: list, download, ingest"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})