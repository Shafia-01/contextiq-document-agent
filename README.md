# ContextIQ
## An Enterprise-Ready Document Q&A AI Agent

**ContextIQ** is an end‑to‑end Document Q&A agent that ingests multiple research PDFs, extracts structured content (text, tables, basic metadata), and answers grounded questions over those documents using LLM APIs (Groq or Gemini). It is built as a small but realistic production slice: FastAPI backend, Streamlit UI, local vector store, and optional ArXiv integration for paper discovery.

### 1. Problem Statement & Objective

- **Problem**: Enterprise teams sit on large collections of PDFs and research reports that are hard to search and reason over. Copy‑pasting into a chat model leads to context loss, no source attribution, and high risk of hallucinations.
- **Objective**: Provide a **grounded, auditable Q&A agent** that:
  - Ingests multiple PDFs and related documents
  - Preserves **page‑level context** and document metadata
  - Answers questions, summarizes content, and surfaces key metrics
  - Always shows **where** the answer came from (document + page)

### 2. Architecture Overview

- **Frontend (`frontend/app.py`)**: Streamlit chat UI for uploading documents, running ArXiv searches, and interacting with the Q&A agent.
- **Backend (`backend/`)**:
  - `main.py` – FastAPI app exposing `/upload`, `/ask`, and `/arxiv_search`.
  - `ingest.py` – PDF‑first ingestion: text, page markers, images, tables, and chunking.
  - `qa.py` – QA engine, LLM abstraction (Groq/Gemini), ArXiv utilities, and retrieval + answer generation.
  - `vectorstore.py` – In‑memory cosine‑similarity vector store.
  - `llm_client.py` – Thin clients that read **API keys from environment variables** only.
- **Data (`data/`)**: Local storage for uploaded documents and extracted assets (tables/images).

End‑to‑end flow: **Upload → Ingest & Chunk → Embed → Retrieve → Generate Answer + Attribution → Display in UI.**

### 3. Enterprise‑Aligned Feature Map

- **Document ingestion**
  - PDF‑optimized pipeline using `fitz` and `pdfplumber`
  - Page‑level markers (`[PAGE n]`) preserved to recover **page numbers per chunk**
  - Extraction of text, basic metadata, tables (CSV) and images (for audit/debug)
- **Retrieval & QA**
  - SentenceTransformers (`all-mpnet-base-v2`) for local embeddings (Groq path)
  - Configurable chunk size and overlap (defaults: **1500 chars / 200‑char overlap**) to balance context and recall
  - Supports **per‑document** answers and **combined, cross‑document** analysis
- **Source attribution & grounding**
  - For each answer, backend returns:
    - Document name/title
    - Approximate **page number** for supporting chunks
    - Retrieval scores and a simple **confidence label** (high/medium/low)
  - Streamlit UI surfaces this alongside the natural‑language answer to reduce hallucination risk.
- **Security & operations**
  - API keys are **never hard‑coded**; they are read from env vars (`GROQ_API_KEY`, `GEMINI_API_KEY`).
  - In‑memory vector store kept deliberately simple for review; can be swapped for FAISS/Pinecone/Chroma.

### 4. Quick Start

- **Prerequisites**
  - Python 3.8+
  - At least one LLM provider key: **Groq** and/or **Gemini**

- **Setup**

```bash
git clone <your-repo-url>
cd ContextIQ

pip install -r requirements.txt

# .env (not committed) – example
export GROQ_API_KEY=your_groq_api_key_here
export GEMINI_API_KEY=your_gemini_api_key_here
```

- **Run backend**

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **Run frontend**

```bash
cd ../frontend
streamlit run app.py --server.port 8501
```

- **Access**
  - UI: `http://localhost:8501`
  - API docs: `http://localhost:8000/docs`

### 5. Example Run (Happy Path)

1. **Upload documents**
   - In the sidebar, upload several research PDFs (e.g. transformer papers).
   - Backend extracts text + tables, chunks into overlapping spans, and populates the vector store.
2. **Ask questions**
   - Example queries:
     - *“What is the main research question of each paper?”*
     - *“Across all papers, summarize how evaluation metrics are defined.”*
     - *“On which page do they report BLEU and ROUGE scores?”*
3. **Expected output (simplified)**
   - Natural‑language answer, e.g.:
     - *“Paper A studies vocabulary growth in large LMs; Paper B proposes a new metric for …”*
   - Plus **attribution section**, e.g.:
     - `Paper A (2509.00516v1.pdf) – pages 3–4 (high similarity)`
     - `Paper B (571e56b308aeaced7889e0bd.pdf) – page 7 (medium similarity)`
   - Confidence label: `confidence: medium (max similarity≈0.83, avg≈0.55)`.

You can also hit the API directly via `/upload`, `/ask`, and `/arxiv_search` using `curl` or Postman.

### 6. PDF Ingestion & Edge Cases

- **Text extraction**
  - Uses `fitz` to iterate pages; each page is logged and wrapped with a `[PAGE n]` marker so later chunks can be mapped back to pages.
  - `pdfplumber` is used opportunistically to extract tables per page to CSV.
- **Chunking strategy**
  - Greedy character‑based chunks of ~1500 chars with 200‑char overlap.
  - This size is chosen to:
    - Fit comfortably within LLM context windows
    - Preserve enough local context for section‑level questions
    - Avoid exploding the number of embeddings for large documents
- **Edge‑case handling (by design)**
  - **Very large PDFs**: chunking is capped via `max_chunks` (default 500) to prevent unbounded memory growth.
  - **Scanned/broken PDFs**: pages with very low text content are logged; behavior is to ingest what’s available and surface the limitation in logs (a realistic place to plug in OCR later).
  - **Empty pages**: explicitly detected and skipped in effective context while keeping page numbering consistent.

### 7. Evaluation & Sanity Checks

This project includes a **lightweight, explainable evaluation heuristic** rather than a heavy benchmark:

- During retrieval, each chunk receives a cosine similarity score to the query.
- The QA engine computes:
  - `max_score` and `avg_score` across retrieved chunks
  - A human‑friendly **confidence label**:
    - High: strong, consistent similarity
    - Medium: mixed evidence
    - Low: weak grounding – likely to contain hallucinations or off‑topic content
- Both the UI and API surface this label and the supporting documents/pages used.

This mirrors how an enterprise system might surface “answer quality” without hiding the underlying retrieval evidence.
