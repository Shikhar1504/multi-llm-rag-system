# Production RAG Assistant

Production RAG Assistant is a modular Retrieval-Augmented Generation system built with FastAPI, LangChain-style abstractions, pluggable LLM providers, and a vector database layer for document search and grounded question answering. It supports PDF ingestion, adaptive retrieval, streaming responses, and persistent chat history. The system is designed to work with multiple providers, with Mistral, Gemini, and HuggingFace as primary options and OpenAI as an optional fallback.

## Key Features

- PDF document ingestion and upload handling
- Recursive chunking with configurable chunk size and overlap
- Embedding generation with provider-based selection
- Vector search with Chroma as the local default and Pinecone support for managed deployments
- Adaptive retrieval pipeline:
  - query rewriting
  - multi-query retrieval
  - optional reranking
- Streaming answer generation
- Source attribution in responses
- Multi-provider LLM support:
  - Mistral
  - Gemini
  - HuggingFace
  - optional OpenAI
- Chat history persistence with SQLite
- Duplicate document protection during ingestion
- Clean API layer with upload, query, history, and health endpoints
- Optional React + Vite frontend for chat interaction

## Architecture Overview

The backend follows a modular structure with clear separation of concerns:

- `app/api`  
  HTTP routes and dependency wiring. This is the thin API layer that exposes upload, query, history, and health endpoints.

- `app/services`  
  Business logic for document ingestion, chat orchestration, and history persistence.

- `app/rag`  
  RAG pipeline logic, including ingestion, embeddings, retrieval, reranking, vector store adapters, and answer assembly.

- `app/core`  
  Application configuration, settings, LLM helper functions, logging, and FastAPI app creation.

### Data Flow

1. Upload a PDF document.
2. The document is chunked into smaller segments.
3. Each chunk is embedded.
4. Chunks are stored in the vector database.
5. A user query is rewritten if needed.
6. The retriever performs direct or multi-query search.
7. Relevant chunks may be reranked.
8. The answer is generated from retrieved context.
9. Sources are returned with the response.
10. User and assistant messages are persisted in SQLite history.

## Tech Stack

- Backend: FastAPI
- RAG / orchestration: LangChain-style components
- LLM providers: Mistral, Gemini, HuggingFace, optional OpenAI
- Embeddings: HuggingFace by default, OpenAI optional, Gemini optional
- Vector database: Chroma locally, Pinecone optional
- Database: SQLite
- Frontend: React 19 + Vite
- Document parsing: PDF loader + recursive text splitter

## Setup Instructions

### 1. Clone the repository

```powershell
git clone https://github.com/Shikhar1504/multi-llm-rag-system.git
cd multi-llm-rag-system
```

### 2. Create a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install backend dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```powershell
Set-Location frontend
npm install
Set-Location ..
```

### 5. Create environment files

```powershell
Copy-Item .env.example .env
Copy-Item frontend\.env.example frontend\.env
```

### 6. Add provider keys

Fill in the required API keys in `.env` based on the provider you select.

## Environment Variables

### Application

- `APP_NAME` - Display name for the API application.
- `APP_VERSION` - Application version string.
- `DEBUG` - Enables FastAPI debug mode.
- `API_PREFIX` - Base prefix for all API routes.
- `HOST` - Backend host address.
- `PORT` - Backend port.
- `RELOAD` - Enables autoreload in development.
- `CORS_ORIGINS` - Comma-separated list of allowed frontend origins.

### Storage

- `VECTOR_STORE_BACKEND` - Vector store backend: `chroma` or `pinecone`.
- `CHROMA_DIR` - Local Chroma persistence directory.
- `HISTORY_DB_PATH` - SQLite database path for chat history.
- `REGISTRY_DB_PATH` - SQLite database path for document registry.
- `PINECONE_API_KEY` - Pinecone API key.
- `PINECONE_INDEX_NAME` - Pinecone index name.
- `PINECONE_NAMESPACE` - Pinecone namespace.

### LLM Providers

- `LLM_PROVIDER` - Selected chat provider: `gemini`, `mistral`, `huggingface`, or `openai`.
- `LLM_MODEL` - Model name or repository id for the selected provider.
- `TEMPERATURE` - Generation temperature.
- `GEMINI_API_KEY` - Gemini API key.
- `MISTRAL_API_KEY` - Mistral API key.
- `HUGGINGFACE_API_KEY` - HuggingFace token or API key.
- `OPENAI_API_KEY` - OpenAI API key.

### Embeddings

- `EMBEDDINGS_PROVIDER` - Embedding provider: `local`, `gemini`, or `openai`.
- `EMBEDDINGS_MODEL` - Embedding model for local or Gemini embedding backends.
- `OPENAI_EMBEDDINGS_MODEL` - OpenAI embedding model name.

### Retrieval Controls

- `CHUNK_SIZE` - Chunk size for recursive splitting.
- `CHUNK_OVERLAP` - Overlap between chunks.
- `RETRIEVER_TOP_K` - Number of top results to return from retrieval.
- `RETRIEVER_FETCH_K` - Number of documents fetched before MMR selection.
- `MULTI_QUERY_COUNT` - Number of generated query variants for multi-query retrieval.
- `RERANK_TOP_K` - Number of chunks kept after reranking.
- `RERANKER_MODEL_NAME` - Cross-encoder reranker model name.
- `SIMPLE_QUERY_MAX_WORDS` - Heuristic threshold for treating a query as simple.
- `SIMPLE_QUERY_MAX_CHARS` - Character threshold for treating a query as simple.
- `MAX_CONTEXT_CHARS` - Maximum context size passed to the LLM.
- `ENABLE_MULTI_QUERY` - Enables or disables multi-query retrieval.
- `ENABLE_RERANKING` - Enables or disables reranking.
- `ENABLE_QUERY_REWRITE` - Enables or disables query rewriting.

## Running the Project

### Start the backend

From the repository root:

```powershell
python main.py
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Start the frontend

From the `frontend` folder:

```powershell
npm run dev
```

The frontend will typically run at:

```text
http://localhost:5173
```

The frontend proxies `/api` requests to the backend.

## API Endpoints

### `POST /api/upload`

Upload and index a PDF document.

#### Request

- Content type: `multipart/form-data`
- Field: `file`

#### Example

```powershell
curl -X POST "http://127.0.0.1:8000/api/upload" ^
  -F "file=@sample.pdf"
```

#### Response example

```json
{
  "document_id": "a1b2c3d4e5f6g7h8",
  "file_name": "sample.pdf",
  "status": "indexed",
  "chunk_count": 24,
  "skipped": false
}
```

---

### `POST /api/query`

Ask a question against the indexed documents.

#### Request body

```json
{
  "question": "What is this document about?",
  "session_id": "default",
  "stream": true,
  "top_k": 4,
  "multi_query_count": 3,
  "rewrite_query": true,
  "rerank": true
}
```

#### Example

```powershell
curl -X POST "http://127.0.0.1:8000/api/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is this document about?\",\"session_id\":\"default\",\"stream\":false}"
```

#### Response example

```json
{
  "answer": "The document discusses ...",
  "rewritten_query": "document summary",
  "sources": [
    {
      "source": "sample.pdf",
      "page": 2,
      "chunk_id": "0",
      "score": null
    }
  ],
  "context_found": true,
  "session_id": "default",
  "created_at": "2026-06-21T12:00:00Z"
}
```

When `stream` is enabled, the backend returns newline-delimited JSON events.

---

### `GET /api/history`

Fetch chat history for a session.

#### Query parameters

- `session_id` - Session identifier

#### Example

```powershell
curl "http://127.0.0.1:8000/api/history?session_id=default"
```

#### Response example

```json
{
  "session_id": "default",
  "messages": [
    {
      "role": "user",
      "content": "What is this document about?",
      "created_at": "2026-06-21T12:00:00Z",
      "sources": []
    },
    {
      "role": "assistant",
      "content": "The document discusses ...",
      "created_at": "2026-06-21T12:00:10Z",
      "sources": [
        {
          "source": "sample.pdf",
          "page": 2,
          "chunk_id": "0",
          "score": null
        }
      ]
    }
  ]
}
```

---

### `GET /api/health`

Health check endpoint.

#### Example

```powershell
curl "http://127.0.0.1:8000/api/health"
```

#### Response

```json
{
  "status": "ok"
}
```

## Example Usage

### 1. Upload a PDF

Send the PDF to `POST /api/upload`. The backend stores the file temporarily, chunks it, embeds it, and writes it into the selected vector store.

### 2. Ask a query

Send a question to `POST /api/query`. The system may rewrite the query, generate multiple query variants, rerank retrieved chunks, and then answer using only the retrieved context.

### 3. Streaming response format

When `stream: true`, the backend returns newline-delimited JSON events.

Example stream:

```text
{"type":"sources","rewritten_query":"document summary","context_found":true,"retrieval_mode":"expanded+reranked","context_truncated":false,"sources":[...]}
{"type":"token","content":"The"}
{"type":"token","content":" document"}
{"type":"token","content":" discusses"}
{"type":"done","answer":"The document discusses ..."}
```

If streaming fails, a fallback error event is emitted and the response is finalized safely.

## Project Structure

```text
.
├── main.py
├── app
│   ├── api
│   │   ├── deps.py
│   │   └── routes.py
│   ├── core
│   │   ├── app.py
│   │   ├── config.py
│   │   ├── llm.py
│   │   └── logging.py
│   ├── models
│   │   └── schemas.py
│   ├── rag
│   │   ├── embeddings.py
│   │   ├── ingestion.py
│   │   ├── pipeline.py
│   │   └── vector_stores
│   │       ├── base.py
│   │       ├── chroma.py
│   │       ├── factory.py
│   │       └── pinecone.py
│   └── services
│       ├── chat_service.py
│       ├── document_service.py
│       └── history_service.py
├── frontend
│   ├── index.html
│   ├── package.json
│   ├── src
│   └── vite.config.ts
├── tests
│   └── test_pipeline.py
├── requirements.txt
├── README.md
├── .env.example
└── data
```

### Brief explanation

- `main.py` - FastAPI entry point.
- `app/api` - request/response endpoints and dependency wiring.
- `app/core` - configuration, provider setup, and application bootstrap.
- `app/models` - Pydantic schemas.
- `app/rag` - ingestion, embeddings, retrieval, reranking, and vector store adapters.
- `app/services` - ingestion, chat, and history business logic.
- `frontend` - React-based UI for chat, upload, and history.
- `tests` - backend unit tests.

## Configuration Options

The system is driven by `.env` values, especially:

- `CHUNK_SIZE` and `CHUNK_OVERLAP` to control document splitting
- `RETRIEVER_TOP_K` and `RETRIEVER_FETCH_K` to control retrieval breadth
- `MULTI_QUERY_COUNT` to control multi-query retrieval
- `ENABLE_QUERY_REWRITE` to toggle query rewriting
- `ENABLE_MULTI_QUERY` to toggle multi-query retrieval
- `ENABLE_RERANKING` to toggle reranking
- `LLM_PROVIDER` to switch between `mistral`, `gemini`, `huggingface`, and `openai`
- `EMBEDDINGS_PROVIDER` to switch between `local`, `gemini`, and `openai`
- `VECTOR_STORE_BACKEND` to switch between `chroma` and `pinecone`

## Testing

Run backend tests:

```powershell
python -m unittest discover -s tests -v
```

Build the frontend:

```powershell
Set-Location frontend
npm run build
```

## Future Improvements

- Add authentication and authorization
- Add document-level access control
- Move file storage to object storage
- Add background jobs for large document ingestion
- Add observability, tracing, and metrics
- Add deployment automation and containerization
- Add rate limiting and abuse protection
