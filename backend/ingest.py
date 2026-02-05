import os
import fitz
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
import hashlib
from tqdm import tqdm
from typing import List, Dict

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
        for line in lines[:5]:  # Check first 5 lines
            clean_line = line.strip()
            if clean_line and len(clean_line) > 3 and len(clean_line) < 100:
                if not clean_line.lower().startswith(('page ', 'chapter ', 'section ')):
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
    print(f"[PDF] Opening {path}...")
    doc = fitz.open(path)
    pdf_name = os.path.splitext(os.path.basename(path))[0]
    print(f"[PDF] Document has {len(doc)} pages.")

    pages = []
    full_text = []
    all_tables = []

    for p in doc:
        print(f"Extracting text from page {p.number+1}...")
        text = p.get_text("text")
        images = list(_save_image_from_page(doc, p, pdf_name))
        pages.append({"page": p.number+1, "text": text, "images": images})
        full_text.append(f"[PAGE {p.number+1}]\n{text}")

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
    all_chunks = []
    for path in tqdm(file_paths, desc="Ingesting documents"):
        try:
            doc = extract_document(path)
            print(f"[DEBUG] Document title: '{doc['title']}'")
            print(f"[DEBUG] Full text length: {len(doc['full_text'])} characters")
            
            chunks = chunk_text(doc["full_text"], chunk_chars=1500, overlap=200)
            
            for c in chunks:
                c["meta"].update({
                    "source_doc": doc["id"],
                    "source_name": os.path.basename(path), 
                    "document_name": doc["title"], 
                    "title": doc.get("title", ""),
                    "source_path": path 
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