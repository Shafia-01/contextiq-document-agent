import streamlit as st
import requests
import time
import re

st.set_page_config(
    page_title="ContextIQ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "arxiv_papers" not in st.session_state:
    st.session_state.arxiv_papers = []


def upload_documents(files):
    try:
        file_data = []
        for file in files:
            file_data.append(("files", (file.name, file.getvalue(), file.type)))
        
        with st.spinner("Uploading documents..."):
            response = requests.post(f"{API_BASE_URL}/upload", files=file_data)
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            error_msg = response.json().get("error", "Upload failed")
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the API server. Please make sure your FastAPI server is running."
    except Exception as e:
        return False, f"Upload error: {str(e)}"


def ask_question(query: str, model: str = "groq", target_document: str = None):
    try:
        data = {
            "query": query,
            "model": model,
            "top_k": 20  # retrieve more chunks to improve recall
        }
        # Add target_document filter if specified
        if target_document:
            data["target_document"] = target_document
        
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_BASE_URL}/ask", data=data)
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            error_msg = response.json().get("error", "Question failed")
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the API server. Please make sure your FastAPI server is running."
    except Exception as e:
        return False, f"Error: {str(e)}"


def arxiv_search(query: str, model: str = "groq", action: str = "list", max_papers: int = 3):
    try:
        data = {
            "query": query,
            "model": model,
            "max_papers": max_papers,
            "action": action,
            "top_k": 20
        }
        
        with st.spinner(f"ArXiv {action}ing..."):
            response = requests.post(f"{API_BASE_URL}/arxiv_search", data=data)
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            error_msg = response.json().get("error", "ArXiv search failed")
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the API server. Please make sure your FastAPI server is running."
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def display_arxiv_papers(papers):
    for i, paper in enumerate(papers, 1):
        with st.expander(f"📄 Paper {i}: {paper['title'][:100]}..."):
            st.write(f"**Authors:** {', '.join(paper['authors'])}")
            st.write(f"**Summary:** {paper['summary'][:500]}...")
            st.write(f"**PDF URL:** {paper['pdf_url']}")


def _paper_label(document_name: str | None, source_name: str | None) -> str:
    """
    Map a document to a stable, human-friendly label.

    - If the document corresponds to one of the uploaded files in this
      Streamlit session, we label it as "Paper N: <title or filename>" where
      N is the 1-based upload order.
    - Otherwise we fall back to the best available name.
    - Filters out page markers like "[PAGE 1]" from document names.
    """
    import re
    
    base_name = document_name or source_name or "Unknown document"
    
    # Remove page markers like "[PAGE 1]" that might have been incorrectly
    # extracted as part of the document title
    base_name = re.sub(r'\[PAGE\s+\d+\]\s*', '', base_name, flags=re.IGNORECASE).strip()
    
    # If after cleaning we have nothing meaningful, fall back to source_name
    if not base_name or base_name == "[PAGE":
        base_name = source_name or "Unknown document"

    uploaded = st.session_state.get("uploaded_files", [])
    for idx, f in enumerate(uploaded):
        if source_name and f.get("name") == source_name:
            return f"Paper {idx + 1}: {base_name}"

    return base_name


def _format_answer_for_display(answer_payload: dict) -> str:
    """
    Turn the structured answer payload from the backend into a readable
    markdown string for the chat UI.

    The backend always returns:
    - mode: "combined" | "per_document" | "none"
    - answer / answers
    - sources: list of {document_name, source_name, source_path, pages}
    - confidence: {label, max_score, avg_score, explanation}
    """
    if not isinstance(answer_payload, dict):
        # Fallback – if backend ever returns a plain string.
        return str(answer_payload)

    mode = answer_payload.get("mode")
    confidence = answer_payload.get("confidence", {})
    sources = answer_payload.get("sources", [])

    lines = []

    # Confidence is logged in backend console, not shown in UI per user request

    # Build a lookup from document name to source metadata so we can attach
    # "Paper N" labels consistently in both answers and sources.
    source_index = {}
    for src in sources:
        key = src.get("document_name") or src.get("source_name")
        if key:
            source_index[key] = src

    if mode == "combined":
        lines.append("**Answer (combined across documents):**")
        lines.append(answer_payload.get("answer", "No answer available."))
    elif mode == "per_document":
        lines.append("**Answers by document:**")
        answers = answer_payload.get("answers", {})
        for doc_id, text in answers.items():
            src = source_index.get(doc_id)
            label = _paper_label(
                (src or {}).get("document_name") if src else doc_id,
                (src or {}).get("source_name") if src else None,
            )
            lines.append(f"- **{label}**")
            lines.append(f"  {text}")
    else:
        lines.append(answer_payload.get("answer", "No answer available."))

    # Source attribution: where did the answer come from?
    if sources:
        lines.append("")
        lines.append("**Sources used (documents & pages):**")
        for src in sources:
            name = _paper_label(src.get("document_name"), src.get("source_name"))
            pages = src.get("pages") or []
            page_str = f"pages {', '.join(str(p) for p in pages)}" if pages else "page information unavailable"
            lines.append(f"- {name} – {page_str}")

        lines.append("")
        lines.append(
            "_Showing document names and approximate pages helps you verify the answer "
            "against the original PDFs and reduces hallucination risk._"
        )

    return "\n".join(lines)


def _extract_target_paper_from_query(query: str) -> tuple[str, str]:
    """
    Detect if the query mentions 'paper 1', 'paper 2', etc., and return
    both the rewritten query and the target filename for filtering.
    
    Returns:
        (rewritten_query, target_filename) where target_filename is None
        if no specific paper is mentioned.
    """
    uploaded = st.session_state.get("uploaded_files", [])
    if not uploaded:
        return query, None
    
    target_filename = None
    
    def replacer(match: re.Match) -> str:
        nonlocal target_filename
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(uploaded):
            filename = uploaded[idx].get("name", f"paper {match.group(1)}")
            target_filename = filename  # Capture for filtering
            return filename
        return match.group(0)
    
    rewritten = re.sub(r"paper\s+(\d+)", replacer, query, flags=re.IGNORECASE)
    return rewritten, target_filename


def main():
    st.markdown("""
        <style>
            /* Apply Cambria globally */
            html, body, p, div.stMarkdown, div.stText, .stChatMessage, label, input, textarea, select {
                font-family: Cambria, serif !important;
                text-align: center !important;
            }

            /* Titles */
            h1, h2, h3, h4, h5, h6 {
                font-family: Cambria, serif !important;
                font-weight: bold;
                color: #0d47a1;
                text-align: center !important;
            }

            /* Center the whole main block */
            div.block-container {
                text-align: center;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background-color: #FFDAF5;
                border-right: 2px solid #d0d7de;
                font-family: Cambria, serif !important;
            }
            
            /* Buttons */
            .stButton button {
                font-family: Cambria, serif !important;
                background-color: #0d47a1;
                color: white;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: bold;
                border: none;
                transition: 0.3s;
                display: block;
                margin: 0 auto;
            }
            .stButton button:hover {
                background-color: #1565c0;
                transform: translateY(-2px);
            }

            /* Chat bubbles */
            .stChatMessage {
                border-radius: 12px;
                padding: 0.8rem;
                margin-bottom: 0.6rem;
                text-align: center;
            }
            .stChatMessage[data-testid="stChatMessage-user"] {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
            }
            .stChatMessage[data-testid="stChatMessage-assistant"] {
                background-color: #f1f8e9;
                border: 1px solid #aed581;
            }

            /* Expanders */
            details summary {
                font-family: inherit !important;
                text-align: center;
            }

            /* Inputs */
            input, textarea, select {
                border-radius: 6px !important;
                border: 1px solid #d0d7de !important;
                padding: 0.4rem !important;
                text-align: center !important;
            }
            /* Remove typing/caret indicator from select (e.g. Select Model dropdown) */
            select {
                caret-color: transparent !important;
                outline: none !important;
            }
            select:focus {
                outline: none !important;
            }
            
            /* Tighter spacing between LIST / DOWNLOAD / INGEST buttons */
            [data-testid="column"] + [data-testid="column"] {
                padding-left: 0.25rem !important;
            }
            section[data-testid="stSidebar"] [data-testid="column"] {
                padding-left: 0.25rem !important;
                padding-right: 0.25rem !important;
            }
            /* Sidebar divider styling */
            section[data-testid="stSidebar"] hr {
                border-top: 3px solid #5E0347 !important;  /* Thickness + color */
                margin: 1rem 0 !important;
            }
            /* Optional: make horizontal rules inside main content bold too */
            hr {
                border-top: 3px solid #5E0347 !important;
                margin: 1rem 0 !important;
            }
                
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(
        '<div style="line-height:1.2;">'
        '<h1 style="font-size:5rem; text-align:center; font-family:Cambria, serif; color:#5E0347; margin-bottom:-5;">🤖 ContextIQ</h1>'
        '<p style="font-size:1.5rem; text-align:center; font-family:Cambria, serif; color:#EB2993; margin-top:0.1rem;">Ask smarter. Get grounded answers.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    if not check_api_health():
        st.error(f"**⚠️ Cannot connect to the API server at {API_BASE_URL}**")
        st.info("**Please make sure your FastAPI server is running and the URL is correct.**")
        st.code("uvicorn main:app --reload", language="bash")
        return
    
    st.success("**✅ Connected to API server**")

    with st.sidebar:
        st.header("⚙️ Configuration")

        model = st.selectbox(
            "**Select Model**",
            options=["gemini", "groq"],
            index=0,
            help="**Choose the AI model for answering questions**"
        )
        
        st.divider()

        st.header("🔍 ArXiv Search")
        
        with st.form("arxiv_form"):
            arxiv_query = st.text_input(
                "**ArXiv Search Query**",
                placeholder="e.g., robotics"
            )
            
            max_papers = st.slider(
                "**Max Papers**",
                min_value=1,
                max_value=10,
                value=3,
                help="**Maximum number of papers to find**"
            )
            
            action_col1, action_col2, action_col3 = st.columns(3, gap="small")
            
            with action_col1:
                list_papers = st.form_submit_button("**📋 LIST**", type="secondary")
            with action_col2:
                download_papers = st.form_submit_button("**⬇️ DOWNLOAD**", type="secondary")
            with action_col3:
                ingest_papers = st.form_submit_button("**📚 INGEST**", type="secondary")

        if list_papers and arxiv_query:
            success, result = arxiv_search(arxiv_query, model, "list", max_papers)
            if success:
                st.session_state.arxiv_papers = result.get("papers", [])
                st.success(f"Found {len(st.session_state.arxiv_papers)} papers")
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"🔍 Found {len(st.session_state.arxiv_papers)} ArXiv papers for: '{arxiv_query}'",
                    "timestamp": time.time()
                })
                st.rerun()
            else:
                st.error(f"ArXiv search failed: {result}")
        
        if download_papers and arxiv_query:
            success, result = arxiv_search(arxiv_query, model, "download", max_papers)
            if success:
                downloaded_files = result.get("downloaded_files", [])
                st.success(f"Downloaded {len(downloaded_files)} papers")
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"⬇️ Downloaded {len(downloaded_files)} ArXiv papers to local storage",
                    "timestamp": time.time()
                })
                st.rerun()
            else:
                st.error(f"Download failed: {result}")
        
        if ingest_papers and arxiv_query:
            success, result = arxiv_search(arxiv_query, model, "ingest", max_papers)
            if success:
                chunks_added = result.get("chunks_added", 0)
                st.success(f"Ingested papers - added {chunks_added} chunks")
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"📚 Ingested ArXiv papers and added {chunks_added} chunks to knowledge base",
                    "timestamp": time.time()
                })
                st.rerun()
            else:
                st.error(f"Ingestion failed: {result}")
        
        if st.session_state.arxiv_papers:
            st.subheader("📄 Found ArXiv Papers")
            display_arxiv_papers(st.session_state.arxiv_papers)
        
        st.divider()

        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "**Choose documents**",
            type=['pdf', 'doc', 'docx', 'txt', 'md'],
            accept_multiple_files=True,
            help="**Upload PDF, Word documents, or text files**"
        )
        
        if uploaded_files:
            if st.button("Upload Documents", type="primary"):
                success, result = upload_documents(uploaded_files)
                
                if success:
                    st.success(f"✅ Successfully uploaded {len(uploaded_files)} file(s)")
                    st.info(f"Added {result.get('chunks_added', 0)} chunks to knowledge base")
                    
                    for file in uploaded_files:
                        if file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                            st.session_state.uploaded_files.append({
                                'name': file.name,
                                'size': file.size,
                                'type': file.type
                            })

                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"📁 Successfully uploaded {len(uploaded_files)} document(s) and added {result.get('chunks_added', 0)} chunks to knowledge base.",
                        "timestamp": time.time()
                    })
                    
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {result}")

        if st.session_state.uploaded_files:
            st.subheader("📄 Uploaded Documents")
            for i, file in enumerate(st.session_state.uploaded_files):
                with st.expander(f"{file['name']}", expanded=False):
                    st.write(f"**Size:** {file['size']:,} bytes")
                    st.write(f"**Type:** {file['type']}")
                    
                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state.uploaded_files.pop(i)
                        st.rerun()
        
        st.divider()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []

        if st.session_state.arxiv_papers and st.button("🗑️ Clear ArXiv Papers"):
            st.session_state.arxiv_papers = []
            st.rerun()

        st.subheader("📊 Status")
        st.write(f"**Documents:** {len(st.session_state.uploaded_files)}")
        st.write(f"**ArXiv Papers:** {len(st.session_state.arxiv_papers)}")
        st.write(f"**Model:** {model.upper()}")
    
    total_sources = len(st.session_state.uploaded_files) + len(st.session_state.arxiv_papers)
    if total_sources > 0:
        st.info(f"**📚 {total_sources} knowledge sources loaded**", icon="📖")
    else:
        st.warning("**📚 No knowledge sources loaded**", icon="⚠️")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
                    
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    # Assistant messages may now contain structured answers that
                    # include confidence and source attribution. Render them as
                    # markdown so recruiters can see document + page grounding.
                    st.markdown(message["content"])
                    
            elif message["role"] == "system":
                with st.chat_message("assistant", avatar="📁"):
                    st.success(message["content"])
            
            elif message["role"] == "error":
                with st.chat_message("assistant", avatar="❌"):
                    st.error(message["content"])
    
    query = st.chat_input("Ask a question about your documents or ingested ArXiv papers...")
    
    if query:
        has_sources = st.session_state.uploaded_files or st.session_state.arxiv_papers
        
        if not has_sources:
            st.error("**❌ Please upload documents or search/ingest ArXiv papers first to ask questions**")
            return
        
        rewritten_query, target_filename = _extract_target_paper_from_query(query)

        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "timestamp": time.time()
        })
        
        success, result = ask_question(rewritten_query, model, target_document=target_filename)
        
        if success:
            # The backend returns a structured answer payload under "answer".
            formatted = _format_answer_for_display(result.get("answer", {}))
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted,
                "timestamp": time.time()
            })
        else:
            st.session_state.messages.append({
                "role": "error",
                "content": f"Error: {result}",
                "timestamp": time.time()
            })
        st.rerun()
    
    if not st.session_state.messages:
        st.markdown("""
        ### 🚀 Getting Started:
        **Upload documents** in the sidebar or **search ArXiv papers** to build your knowledge base, then **ask questions** in the chat below to get intelligent AI-powered answers.
        """)  

if __name__ == "__main__":
    main()