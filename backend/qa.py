"""
QA engine and model abstraction for ContextIQ.

This module wires together:
- A pluggable LLM interface (currently Groq + Gemini)
- A light-weight in-memory vector store
- Retrieval-augmented generation over ingested document chunks.

Non-trivial design choices:
- Embeddings for both providers are computed locally with
  ``all-mpnet-base-v2`` for speed and privacy; this keeps the retrieval layer
  independent from the choice of generation provider.
- The QAEngine returns **structured answers** including document-level
  attribution and a simple, explainable confidence heuristic derived from
  cosine similarity scores. This is deliberately transparent so that an
  enterprise reviewer can see how grounding is computed.
"""

from typing import List, Dict, Any, Tuple
from llm_client import get_llm_client
from ingest import extract_documents
from sentence_transformers import SentenceTransformer
import requests
import os
import hashlib
import arxiv

embed_model = SentenceTransformer("all-mpnet-base-v2")


class LLMInterface:
    """Minimal interface for LLM backends used by the QA engine."""

    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        """Return embedding vectors for each input text."""
        raise NotImplementedError

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a completion for ``prompt`` under an optional system prompt."""
        raise NotImplementedError

class GeminiModel(LLMInterface):
    """
    Gemini-backed implementation of the LLM interface.

    We deliberately keep **embeddings local** (SentenceTransformers) so that
    retrieval behaves the same regardless of whether the generator is Gemini
    or Groq. Gemini is only responsible for turning retrieved context into
    natural-language answers.

    Note: The default model id here is set to ``gemini-2.5-flash``, which
    matches the current Gemini Developer API naming. If your account exposes
    a different model (e.g. ``gemini-3.0-flash``), you can change this string
    or introduce an environment variable to override it.
    """

    def __init__(self, llm_model: str = "gemini-2.5-flash"):
        # Use the modern google-genai client wired via llm_client.
        self.client = get_llm_client("gemini")
        self.llm_model = llm_model

    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        # Shared local embedding model for all providers; this keeps retrieval
        # deterministic and avoids coupling to provider-specific embedding APIs.
        return [embed_model.encode(t).tolist() for t in text_list]

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        content = f"{system_prompt or 'You are a helpful assistant.'}\n\n{prompt}"
        resp = self.client.models.generate_content(
            model=self.llm_model,
            contents=content,
        )
        return resp.text

class GroqModel(LLMInterface):
    def __init__(self, llm_model="groq/compound"):
        self.client = get_llm_client("groq")
        self.llm_model = llm_model

    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        return [embed_model.encode(t).tolist() for t in text_list]

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()

def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for result in search.results():
        results.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })
    return results

def download_pdf(url: str, save_dir="data/assets") -> str:
    os.makedirs(save_dir, exist_ok=True)
    response = requests.get(url)
    file_hash = hashlib.sha1(url.encode()).hexdigest()[:10]
    file_path = os.path.join(save_dir, f"{file_hash}.pdf")
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path

class QAEngine:
    def __init__(self, vectorstore, llm: LLMInterface):
        self.vs = vectorstore
        self.llm = llm

    def add_documents(self, chunks: List[Dict]):
        """
        Add already-chunked documents to the vector store.

        Each chunk is embedded once and stored along with its metadata.
        Metadata is expected to contain:
        - ``document_name`` / ``title`` – human-readable identifier
        - ``source_name`` / ``source_path`` – original file reference
        - ``page`` – (optional) page number for PDFs, inferred at ingestion time.
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.llm.get_embeddings(texts)
        for c, vec in zip(chunks, embeddings):
            meta = c.get("meta", {})
            self.vs.add(vec, {"text": c["text"], "metadata": meta}, id=c["id"])
        print(f"[QA] Added {len(chunks)} chunks to vectorstore.")
        
        doc_names = {c.get("meta", {}).get("document_name", "unknown") for c in chunks}
        print(f"[QA] Document names stored: {list(doc_names)}")
    
    def _compute_confidence(self, scores: List[float]) -> Dict[str, Any]:
        """
        Convert raw similarity scores into a simple, explainable confidence label.

        - High: strong and consistent similarity across retrieved chunks
        - Medium: mixed evidence
        - Low: weak grounding; answers are more likely to drift or hallucinate.
        """
        if not scores:
            return {
                "label": "low",
                "max_score": 0.0,
                "avg_score": 0.0,
                "explanation": "No supporting chunks retrieved; answer is likely ungrounded.",
            }

        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        if max_score > 0.85 and avg_score > 0.65:
            label = "high"
        elif max_score > 0.6 and avg_score > 0.4:
            label = "medium"
        else:
            label = "low"

        return {
            "label": label,
            "max_score": max_score,
            "avg_score": avg_score,
            "explanation": (
                "Derived from cosine similarity between the question and retrieved chunks. "
                "Higher scores mean the answer is better grounded in the source documents."
            ),
        }

    def _group_retrieved_by_document(self, retrieved: List[Dict]) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
        """
        Group retrieved chunks by document for attribution and per-document QA.

        Returns:
        - mapping from document id/name → list of chunk dicts with text, score, and metadata
        - mapping from document id/name → stable source metadata (name, path, pages used)
        """
        grouped: Dict[str, List[Dict]] = {}
        sources: Dict[str, Dict] = {}

        for r in retrieved:
            payload = r["metadata"]
            text = payload.get("text", "")
            meta = payload.get("metadata", {})  # metadata stored by vectorstore.add

            doc_id = (
                meta.get("document_name")
                or meta.get("title")
                or meta.get("source_name")
                or "unknown_doc"
            )

            chunk_info = {
                "text": text,
                "score": r.get("score", 0.0),
                "page": meta.get("page"),
                "source_name": meta.get("source_name"),
                "document_name": meta.get("document_name") or meta.get("title") or meta.get("source_name"),
                "source_path": meta.get("source_path"),
            }

            print(f"[DEBUG] Retrieved chunk from: '{doc_id}' (page={chunk_info['page']}, score={chunk_info['score']:.3f})")
            grouped.setdefault(doc_id, []).append(chunk_info)

            # Maintain a compact, document-level source description.
            if doc_id not in sources:
                sources[doc_id] = {
                    "document_name": chunk_info["document_name"] or doc_id,
                    "source_name": chunk_info["source_name"],
                    "source_path": chunk_info["source_path"],
                    "pages": set(),  # we convert to list later for JSON
                }
            if chunk_info["page"] is not None:
                sources[doc_id]["pages"].add(chunk_info["page"])

        # Normalise page sets into sorted lists for JSON serialisation.
        for doc_id, src in sources.items():
            src["pages"] = sorted(src["pages"]) if src["pages"] else []

        return grouped, sources

    def ask(self, query: str, top_k: int = 10) -> Dict:
        """
        Answer a user query using retrieval-augmented generation.

        Returns a structured payload with:
        - mode: "combined" | "per_document"
        - answer / answers: natural-language answer(s) from the LLM
        - sources: document-level attribution (names, paths, pages)
        - confidence: lightweight similarity-based confidence heuristic.
        """
        print(f"[QA] User query: {query}")
        query_vec = self.llm.get_embeddings([query])[0]
        retrieved = self.vs.similarity_search(query_vec, top_k=top_k)

        if not retrieved:
            print("[QA] No chunks retrieved for query; returning fallback message.")
            return {
                "mode": "none",
                "answer": "I could not find any relevant content in the ingested documents for this question.",
                "sources": [],
                "confidence": {
                    "label": "low",
                    "max_score": 0.0,
                    "avg_score": 0.0,
                    "explanation": "No supporting chunks retrieved; answer is likely ungrounded.",
                },
            }

        grouped_chunks, source_meta = self._group_retrieved_by_document(retrieved)
        print(f"[QA] Retrieved content from {len(grouped_chunks)} documents: {list(grouped_chunks.keys())}")

        scores = [c["score"] for chunks in grouped_chunks.values() for c in chunks]
        confidence = self._compute_confidence(scores)
        
        # Log confidence to console (not shown in UI per user request)
        print(f"[QA] Confidence: {confidence.get('label', 'unknown')} "
              f"(max≈{confidence.get('max_score', 0.0):.3f}, "
              f"avg≈{confidence.get('avg_score', 0.0):.3f})")

        is_combined = any(
            keyword in query.lower() for keyword in ["these papers", "all papers", "combined", "together"]
        )

        if is_combined:
            # Multi-document combined analysis.
            all_text = "\n\n".join([c["text"] for chunks in grouped_chunks.values() for c in chunks])
            prompt = f"""You are a research assistant answering questions over multiple documents.
Use ONLY the provided context and do not invent facts that are not supported by it.

Context:
{all_text}

Question: {query}
Answer (be concise but specific):
"""
            answer = self.llm.generate_text(prompt)
            return {
                "mode": "combined",
                "answer": answer,
                "sources": list(source_meta.values()),
                "confidence": confidence,
            }

        # Per-document answers: one answer per document.
        answers: Dict[str, str] = {}
        for doc_id, chunks in grouped_chunks.items():
            doc_text = "\n\n".join(c["text"] for c in chunks)
            prompt = f"""You are a research assistant answering questions about a single document.
Use ONLY the provided context from this document; if the answer is not present, say so explicitly.

Context:
{doc_text}

Question: {query}
Answer (be concise but specific):
"""
            answers[doc_id] = self.llm.generate_text(prompt)

        return {
            "mode": "per_document",
            "answers": answers,
            "sources": list(source_meta.values()),
            "confidence": confidence,
        }

    def search_and_list_arxiv(self, query: str, max_papers: int = 5) -> List[Dict]:
        papers = search_arxiv(query, max_results=max_papers)
        print(f"[Arxiv] Found {len(papers)} papers for query: {query}")
        for i, p in enumerate(papers, 1):
            authors = ", ".join(p["authors"]) if p["authors"] else "Unknown"
            print(f"{i}. {p['title']} ({authors})")
            print(f"   PDF: {p['pdf_url']}\n   Summary: {p['summary'][:300]}...\n")
        return papers

    def ingest_papers(self, papers: List[Dict]):
        all_chunks = []
        for paper in papers:
            if not paper.get("pdf_url"):
                continue
            pdf_path = download_pdf(paper["pdf_url"])
            chunks = extract_documents([pdf_path])
            for c in chunks:
                c["meta"].update({
                    "source_name": paper["title"],
                    "document_name": paper["title"], 
                    "title": paper["title"]
                })
            all_chunks.extend(chunks)
        if all_chunks:
            self.add_documents(all_chunks)
        return len(all_chunks)

    def interactive_arxiv_qa(self, query: str, max_papers: int = 5, top_k: int = 10):
        papers = self.search_and_list_arxiv(query, max_papers=max_papers)

        download_choice = input("Do you want to download these papers as PDFs? (yes/no): ").strip().lower()
        downloaded_papers = []
        if download_choice == "yes":
            print("[Arxiv] Downloading papers...")
            for paper in papers:
                if not paper.get("pdf_url"):
                    continue
                pdf_path = download_pdf(paper["pdf_url"])
                paper["local_path"] = pdf_path
                downloaded_papers.append(paper)
            print(f"[Arxiv] Downloaded {len(downloaded_papers)} papers to 'data/assets/'.")
        else:
            print("[Arxiv] Skipped downloading.")

        ingest_choice = input("Do you want to ingest the downloaded papers into vectorstore? (yes/no): ").strip().lower()
        if ingest_choice == "yes" and downloaded_papers:
            print("[Arxiv] Ingesting downloaded papers...")
            self.ingest_papers(downloaded_papers)
            return self.ask(query, top_k=top_k)
        elif ingest_choice == "yes" and not downloaded_papers:
            return "No papers were downloaded, so ingestion is not possible."
        else:
            return "Skipped ingestion. You can still read/download the PDFs manually."