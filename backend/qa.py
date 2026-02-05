from typing import List, Dict
from llm_client import get_gemini_client, get_groq_client
from ingest import extract_documents
from sentence_transformers import SentenceTransformer
import requests
import os
import hashlib
import arxiv

embed_model = SentenceTransformer("all-mpnet-base-v2")

class LLMInterface:
    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError

class GeminiModel(LLMInterface):
    def __init__(self, embedding_model="models/text-embedding-004", llm_model="gemini-1.5-flash"):
        self.genai = get_gemini_client()
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        return [self.genai.embed_content(model=self.embedding_model, content=t)["embedding"] for t in text_list]

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        model = self.genai.GenerativeModel(self.llm_model)
        chat = model.start_chat(history=[])
        resp = chat.send_message(f"{system_prompt or 'You are a helpful assistant.'}\n\n{prompt}")
        return resp.text

class GroqModel(LLMInterface):
    def __init__(self, llm_model="groq/compound"):
        self.client = get_groq_client()
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
        texts = [c["text"] for c in chunks]
        embeddings = self.llm.get_embeddings(texts)
        for c, vec in zip(chunks, embeddings):
            meta = c.get("meta", {})
            self.vs.add(vec, {"text": c["text"], "metadata": meta}, id=c["id"])
        print(f"[QA] Added {len(chunks)} chunks to vectorstore.")
        
        doc_names = set()
        for c in chunks:
            doc_name = c.get("meta", {}).get("document_name", "unknown")
            doc_names.add(doc_name)
        print(f"[QA] Document names stored: {list(doc_names)}")
    
    def ask(self, query: str, top_k: int = 10) -> Dict:
        print(f"[QA] User query: {query}")
        query_vec = self.llm.get_embeddings([query])[0]
        retrieved = self.vs.similarity_search(query_vec, top_k=top_k)

        docs_dict = {}
        for r in retrieved:
            metadata = r["metadata"].get("metadata", {})  
            doc_id = (metadata.get("document_name") or 
                     metadata.get("title") or 
                     metadata.get("source_name") or 
                     "unknown_doc")
            
            print(f"[DEBUG] Retrieved chunk from: '{doc_id}'")  # Debug output
            docs_dict.setdefault(doc_id, []).append(r["metadata"]["text"])

        print(f"[QA] Retrieved content from {len(docs_dict)} documents: {list(docs_dict.keys())}")

        if any(keyword in query.lower() for keyword in ["these papers", "all papers", "combined", "together"]):
            all_text = "\n\n".join([text for texts in docs_dict.values() for text in texts])
            prompt = f"""Answer the following question using all provided context from multiple documents.

Context:
{all_text}

Question: {query}
Answer:
"""
            answer = self.llm.generate_text(prompt)
            return {"answer": answer, "sources": list(docs_dict.keys())}
        else:
            answers = {}
            for doc_id, texts in docs_dict.items():
                doc_text = "\n\n".join(texts)
                prompt = f"""Answer the following question using ONLY the provided context from this document.

Context:
{doc_text}

Question: {query}
Answer:
"""
                answer = self.llm.generate_text(prompt)
                answers[doc_id] = answer
            return {"answers": answers, "sources": list(docs_dict.keys())}

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