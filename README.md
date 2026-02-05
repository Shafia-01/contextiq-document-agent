# IntelliDoc - Intelligent Document Q&A Agent

🤖 This is a powerful AI-powered document question-answering system that allows you to upload documents and ask intelligent questions about their content. Built with FastAPI backend and Streamlit frontend, it supports multiple AI models and integrates with ArXiv for research paper ingestion.

## ✨ Features

### 📁 Document Processing
- **Multi-format Support**: PDF, DOCX, HTML, TXT, MD files
- **Smart Extraction**: Text, tables, images, and metadata extraction
- **Intelligent Chunking**: Optimized text segmentation with overlap
- **Document Naming**: Automatic title extraction from content and metadata

### 🤖 AI Models
- **Groq Integration**: Fast inference with local embeddings
- **Google Gemini**: Advanced reasoning capabilities
- **Flexible Switching**: Change models on-the-fly
- **Local Embeddings**: SentenceTransformers for vector similarity

### 🔍 ArXiv Integration
- **Paper Search**: Find relevant research papers
- **Batch Operations**: List, download, and ingest papers
- **Research Assistant**: Combine uploaded docs with research papers

### 💬 Interactive Interface
- **Streamlit Frontend**: User-friendly web interface
- **Real-time Chat**: Interactive Q&A with your documents
- **Multi-source Answers**: Per-document or combined responses
- **Session Management**: Persistent chat history and file tracking

## 🏗️ Project Structure

```
IntelliDoc/
├── backend/                # FastAPI backend services
│   ├── main.py             # API endpoints and server setup
│   ├── ingest.py           # Document processing pipeline
│   ├── qa.py               # Q&A engine with LLM integration
│   ├── llm_client.py       # AI model client factory
│   └── vectorstore.py      # In-memory vector storage
├── frontend/
│   └── app.py              # Streamlit web interface
├── data/                   # Document storage and processing
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for AI models (Groq and/or Gemini)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Docu Agent"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   # Required: At least one AI model API key
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Start the backend server**
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Start the frontend** (in a new terminal)
   ```bash
   cd frontend
   streamlit run app.py --server.port 8501
   ```

6. **Access the application**
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## 📖 Usage Guide

### Document Upload
1. **Upload Files**: Use the sidebar to upload PDF, DOCX, or text files
2. **Automatic Processing**: Documents are chunked and added to the knowledge base
3. **Smart Naming**: Document titles are extracted automatically

### ArXiv Research
1. **Search Papers**: Enter keywords to find relevant research papers
2. **Preview Results**: List papers to see titles, authors, and summaries
3. **Download**: Save papers locally for offline reading
4. **Ingest**: Add papers to the knowledge base for Q&A

### Asking Questions
1. **Simple Questions**: Ask about specific documents
2. **Combined Analysis**: Use keywords like "all papers" or "combined" for multi-document answers
3. **Model Selection**: Choose between Groq (faster) or Gemini (more advanced)

## 🔧 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /upload` - Upload multiple documents
- `POST /ask` - Ask questions about documents
- `POST /arxiv_search` - Search and manage ArXiv papers

### Example API Usage
```bash
# Upload documents
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -F "query=What is the main topic of this document?" \
  -F "model=groq" \
  -F "top_k=10"

# Search ArXiv papers
curl -X POST "http://localhost:8000/arxiv_search" \
  -F "query=machine learning transformers" \
  -F "action=list" \
  -F "max_papers=5"
```

## 🛠️ Configuration

### AI Models
- **Groq**: Fast inference, local embeddings with SentenceTransformers
- **Gemini**: Advanced reasoning, cloud-based embeddings
- **Custom Models**: Extend `llm_client.py` to add more providers

### Document Processing
- **Chunk Size**: 1500 characters with 200-character overlap
- **Max Chunks**: 500 per document (safety limit)
- **Supported Formats**: PDF, DOCX, HTML, TXT, MD

### Vector Storage
- **In-Memory**: Fast local storage using NumPy
- **Production**: Easily swap for FAISS, Pinecone, or Chroma

## 🔍 Advanced Features

### Multi-Document Analysis
- **Per-Document Answers**: Get answers specific to each document
- **Combined Analysis**: Merge insights from multiple sources
- **Source Attribution**: See which documents contributed to answers

### Research Integration
- **ArXiv Search**: Find papers by keywords, authors, or topics
- **Batch Processing**: Download and ingest multiple papers at once
- **Metadata Extraction**: Automatic paper title and author detection

### Smart Document Handling
- **Title Extraction**: Intelligent document naming from content
- **Table Processing**: CSV export of PDF tables
- **Image Extraction**: Save images from PDFs for reference

## 🐛 Troubleshooting

### Common Issues
1. **API Connection Error**: Ensure the FastAPI server is running on port 8000
2. **Missing API Keys**: Check your `.env` file has valid API keys
3. **Document Processing**: Large PDFs may take time to process
4. **Memory Usage**: In-memory storage grows with document count

### Performance Tips
- Use Groq for faster responses
- Limit document chunk count for large files
- Clear chat history periodically to free memory

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


***Ask anything. Know everything.***