"""
Document ingestion utilities for ContextIQ.

This module focuses on **PDF-first** ingestion, but also supports DOCX and HTML.
The design goals are:
- Preserve enough structure to reason about documents (page markers, basic metadata)
- Extract auxiliary artefacts (tables, images) for auditability
- Produce retrieval-friendly text chunks with stable metadata for grounding.

Non-trivial decisions:
- We add explicit ``[PAGE n]`` markers into ``full_text`` so that downstream
  components can heuristically recover **page numbers per chunk** without
  re-opening the PDF.
- Chunking is character-based (1500 chars, 200 overlap by default). This is a
  good compromise between:
    * fitting into common LLM context windows
    * preserving local coherence at the section/paragraph level
    * keeping the embedding count manageable for large PDFs.
"""

import os
import re
import fitz
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
import hashlib
from tqdm import tqdm
from typing import List, Dict, Optional

ASSET_DIR = "data/assets"
os.makedirs(ASSET_DIR, exist_ok=True)

def get_document_name(path: str, doc_content: Dict) -> str:
    base_name = os.path.splitext(os.path.basename(path))[0]
    
    if path.lower().endswith('.pdf'):
        try:
            pdf_doc = fitz.open(path)
            metadata = pdf_doc.metadata
            if metadata and metadata.get('title') and metadata['title'].strip():
                pdf_title = metadata['title'].strip()
                pdf_doc.close()
                print(f"[DEBUG] Using PDF metadata title: {pdf_title}")
                return pdf_title
            pdf_doc.close()
        except Exception as e:
            print(f"[DEBUG] Could not extract PDF metadata: {e}")
    
    if doc_content.get('full_text'):
        lines = doc_content['full_text'].strip().split('\n')
        for line in lines[:10]:  # Check first 10 lines for better title detection
            clean_line = line.strip()
            # Skip page markers, empty lines, and very short/long lines
            if (clean_line and len(clean_line) > 3 and len(clean_line) < 200 and
                not clean_line.lower().startswith(('page ', 'chapter ', 'section ')) and
                not clean_line.startswith('[PAGE') and  # Skip [PAGE n] markers
                not re.match(r'^\[PAGE\s+\d+\]', clean_line, re.IGNORECASE)):  # Skip page markers
                print(f"[DEBUG] Using content-based title: {clean_line}")
                return clean_line
    
    print(f"[DEBUG] Using filename-based title: {base_name}")
    return base_name

def _save_image_from_page(doc, page, pdf_name):
    print(f"Extracting images from page {page.number+1}...")
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        ext = "png"
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img_name = f"{pdf_name}_page{page.number+1}_img{img_index}.{ext}"
        img_path = os.path.join(ASSET_DIR, img_name)
        pix.save(img_path)
        pix = None
        yield img_path

def extract_pdf(path: str) -> Dict:
    """
    Extract structured content from a PDF.

    Returns a dictionary containing:
    - per-page text and image paths
    - a ``full_text`` field with explicit ``[PAGE n]`` markers to allow
      downstream components to map character offsets back to page numbers
      without re-opening the file
    - a list of extracted tables with page references.

    Notes on edge cases:
    - For **very large PDFs**, we stream pages one by one instead of loading
      everything into memory at once.
    - Pages with almost no text are logged; these are often scanned or broken
      PDFs where OCR would be required. We keep the behaviour simple and
      transparent rather than silently dropping them.
    """
    print(f"[PDF] Opening {path}...")
    doc = fitz.open(path)
    pdf_name = os.path.splitext(os.path.basename(path))[0]
    print(f"[PDF] Document has {len(doc)} pages.")

    pages = []
    full_text = []
    all_tables = []

    for p in doc:
        page_number = p.number + 1
        print(f"Extracting text from page {page_number}...")
        text = p.get_text("text")

        # Very low-text pages are often scanned or empty; we still ingest them
        # but log this explicitly so a reviewer understands why answers may
        # lack grounded content for those pages.
        if not text or len(text.strip()) == 0:
            print(f"[WARN] Page {page_number} appears to be empty or low-text. "
                  f"This is common for scanned PDFs without OCR.")

        images = list(_save_image_from_page(doc, p, pdf_name))
        pages.append({"page": page_number, "text": text, "images": images})
        # Add a page marker so later we can infer page numbers from character offsets.
        full_text.append(f"[PAGE {page_number}]\n{text}")

    print(f"[PDF] Attempting table extraction with pdfplumber...")
    try:
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages):
                print(f"Extracting tables from page {i+1}...")
                tables = p.extract_tables()
                for t_idx, table in enumerate(tables):
                    table_path = os.path.join(
                        ASSET_DIR, f"{pdf_name}_page{i+1}_table{t_idx}.csv"
                    )
                    with open(table_path, "w", encoding="utf-8") as fh:
                        for row in table:
                            row_clean = [
                                "" if x is None else str(x).replace(",", " ") for x in row
                            ]
                            fh.write(",".join(row_clean) + "\n")
                    all_tables.append({"page": i+1, "csv": table_path})
    except Exception as e:
        print(f"[WARN] Table extraction failed: {e}")

    full_text_str = "\n\n".join(full_text)
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    doc_dict = {
        "id": doc_id,
        "path": path,
        "title": "",
        "pages": pages,
        "full_text": full_text_str,
        "tables": all_tables,
    }
    doc_dict["title"] = get_document_name(path, doc_dict)

    return doc_dict

def extract_docx(path: str) -> Dict:
    print(f"[DOCX] Opening {path}...")
    doc = Document(path)
    
    title = None
    if hasattr(doc.core_properties, 'title') and doc.core_properties.title:
        title = doc.core_properties.title.strip()
        print(f"[DEBUG] Found DOCX title property: {title}")
    
    text = [para.text for para in doc.paragraphs if para.text.strip()]
    full_text_str = "\n".join(text)
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    
    doc_dict = {
        "id": doc_id,
        "path": path,
        "title": title if title else "", 
        "pages": [],
        "full_text": full_text_str,
        "tables": [],
    }
    if not doc_dict["title"]:
        doc_dict["title"] = get_document_name(path, doc_dict)
    
    return doc_dict

def extract_html(path: str) -> Dict:
    print(f"[HTML] Opening {path}...")
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        title = None
        title_tag = soup.find('title')
        if title_tag and title_tag.text.strip():
            title = title_tag.text.strip()
            print(f"[DEBUG] Found HTML title tag: {title}")
        text = soup.get_text(" ", strip=True)
    
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    
    doc_dict = {
        "id": doc_id,
        "path": path,
        "title": title if title else "",
        "pages": [],
        "full_text": text,
        "tables": [],
    }
    
    if not doc_dict["title"]:
        doc_dict["title"] = get_document_name(path, doc_dict)
    
    return doc_dict

def extract_document(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".docx":
        return extract_docx(path)
    elif ext in [".html", ".htm"]:
        return extract_html(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def _infer_page_from_offset(full_text: str, char_index: int) -> Optional[int]:
    """
    Heuristically infer the page number for a chunk given its **start offset**
    in ``full_text``.

    We rely on the explicit ``[PAGE n]`` markers that ``extract_pdf`` adds
    when constructing ``full_text``. This avoids reopening the PDF or doing
    per-chunk page math in the retrieval layer.

    This is intentionally simple and explainable: if the markers are missing
    or malformed, we fall back to ``None`` rather than guessing.
    """
    marker = "[PAGE "
    # Look backwards from the chunk start to find the most recent page marker.
    pos = full_text.rfind(marker, 0, char_index)
    if pos == -1:
        return None

    end_bracket = full_text.find("]", pos)
    if end_bracket == -1:
        return None

    page_str = full_text[pos + len(marker) : end_bracket]
    try:
        return int(page_str.strip())
    except ValueError:
        return None

def chunk_text(text: str, chunk_chars: int = 1500, overlap: int = 200, max_chunks: int = 500):
    print(f"[Chunking] Splitting text into chunks (size={chunk_chars}, overlap={overlap})...")
    if not text or len(text.strip()) == 0:
        print("[Chunking] Warning: Empty text received.")
        return []

    text = text.replace("\r", "")
    chunks = []
    start = 0
    doc_len = len(text)
    chunk_id = 0

    # We deliberately use a simple sliding-window strategy here rather than
    # sentence segmentation. For research PDFs this tends to be more robust
    # across noisy layouts while still preserving enough local context.
    while start < doc_len and chunk_id < max_chunks:
        end = min(start + chunk_chars, doc_len)
        chunk_text = text[start:end]

        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_text,
            "meta": {"start": start, "end": end},
        })

        chunk_id += 1
        if end >= doc_len:
            break  # done

        start = max(0, end - overlap)

    print(f"[Chunking] Created {len(chunks)} chunks (capped at {max_chunks}).")
    return chunks

def extract_documents(file_paths: List[str]) -> List[Dict]:
    """
    High-level ingestion entrypoint used by the FastAPI backend.

    For each file path:
    - Extracts structured document content
    - Chunks ``full_text`` into overlapping spans
    - Attaches rich metadata (document id/name/path, and inferred page numbers)
      to every chunk so that the QA layer can:
        * ground answers in specific documents
        * display document **and page** back to the user.

    Any ingestion failures are logged but do not crash the whole pipeline,
    since in an enterprise setting it is common for one file in a batch to be
    malformed.
    """
    all_chunks = []
    for path in tqdm(file_paths, desc="Ingesting documents"):
        try:
            doc = extract_document(path)
            full_text = doc["full_text"]

            print(f"[DEBUG] Document title: '{doc['title']}'")
            print(f"[DEBUG] Full text length: {len(full_text)} characters")
            
            chunks = chunk_text(full_text, chunk_chars=1500, overlap=200)
            
            for c in chunks:
                start_offset = c["meta"].get("start", 0)
                page = _infer_page_from_offset(full_text, start_offset)

                c["meta"].update({
                    "source_doc": doc["id"],
                    "source_name": os.path.basename(path),
                    "document_name": doc["title"],
                    "title": doc.get("title", ""),
                    "source_path": path,
                    # This page number is approximate for non-PDF formats and
                    # may be None; for PDFs it is inferred from [PAGE n] markers.
                    "page": page,
                })
            
            all_chunks.extend(chunks)
            print(f"[DEBUG] Added {len(chunks)} chunks from '{doc['title']}'")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python backend/ingest.py file1.pdf file2.docx ...")
        sys.exit(1)

    files = sys.argv[1:]
    chunks = extract_documents(files)
    print(f"\n[Done] Ingested {len(files)} documents → {len(chunks)} chunks total.")
    
    doc_names = set()
    for chunk in chunks:
        doc_names.add(chunk["meta"]["document_name"])
    
    print("\nDocument names extracted:")
    for name in sorted(doc_names):
        print(f" - '{name}'")
    
    print("\nFiles processed:")
    for f in files:
        print(f" - {f}")