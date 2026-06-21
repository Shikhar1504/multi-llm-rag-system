# Production RAG Assistant

A modular Retrieval-Augmented Generation application built with FastAPI, LangChain, Chroma or Pinecone, and a React + Vite frontend.

## Overview

This project upgrades a basic RAG prototype into a production-oriented structure with clear separation of concerns:

- FastAPI backend for upload, query, and history APIs
- Pluggable vector store layer with Chroma as the local fallback and Pinecone as the scalable option
- Query rewriting, multi-query retrieval, optional reranking, and metadata-aware chunking
- Streaming chat frontend with PDF upload and conversation history
- Basic persistence for indexed documents and chat history

## Project Structure

- `main.py` - backend entry point
- `app/core` - configuration, app factory, logging, and LLM helpers
- `app/api` - HTTP routes and dependency wiring
- `app/models` - request and response schemas
- `app/rag` - ingestion, embeddings, pipeline, and vector store adapters
- `app/services` - document ingestion, chat orchestration, and history persistence
- `frontend` - React + Vite chat UI
- `tests` - backend unit tests

## Features

- Recursive text splitting with configurable chunk size and overlap
- Metadata support for source, page number, and chunk index
- Configurable top-k retrieval
- Multi-query retrieval for better recall
- Optional reranking layer
- Query rewriting before retrieval
- Source attribution in answers
- No-context fallback to reduce hallucination
- Streaming responses from the backend to the frontend

## Tech Stack

- Python 3.12+
- FastAPI
- LangChain
- ChromaDB
- Pinecone optional
- React 19
- Vite

## Setup

### 1. Backend environment

Create a virtual environment and install the backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the example environment file and fill in any provider keys you want to use:

```powershell
Copy-Item .env.example .env
```

### 2. Frontend environment

Install the frontend dependencies:

```powershell
Set-Location frontend
npm install
```

If needed, create a frontend environment file:

```powershell
Copy-Item .env.example .env
```

## Run Locally

### Backend

From the repository root:

```powershell
python main.py
```

The FastAPI app runs at `http://127.0.0.1:8000` by default.

### Frontend

From the `frontend` folder:

```powershell
npm run dev
```

The Vite app runs at `http://localhost:5173` by default and proxies `/api` requests to the backend.

## API Endpoints

- `POST /api/upload` - upload and index a PDF document
- `POST /api/query` - ask a question against the indexed documents
- `GET /api/history?session_id=...` - fetch chat history for a session
- `GET /api/health` - health check

### Upload

Send a multipart form upload with a `file` field containing a PDF.

### Query

Example request body:

```json
{
  "question": "What is the document about?",
  "session_id": "default",
  "stream": true,
  "top_k": 4,
  "multi_query_count": 3,
  "rewrite_query": true,
  "rerank": true
}
```

When `stream` is enabled, the response uses newline-delimited JSON events.

## Environment Variables

Key backend settings are defined in `.env.example`.

Important values:

- `VECTOR_STORE_BACKEND` - `chroma` or `pinecone`
- `EMBEDDINGS_PROVIDER` - `local` or `openai`
- `LLM_PROVIDER` - `mistral` or `openai`
- `CHUNK_SIZE` and `CHUNK_OVERLAP` - text splitting controls
- `RETRIEVER_TOP_K` and `RETRIEVER_FETCH_K` - retrieval tuning
- `ENABLE_MULTI_QUERY`, `ENABLE_RERANKING`, `ENABLE_QUERY_REWRITE` - RAG behavior toggles

## Testing

Run backend tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Build the frontend:

```powershell
Set-Location frontend
npm run build
```

## Notes

- Chroma is the default local vector store and persists under `data/chroma`.
- Pinecone support is available through the vector-store abstraction if you want a managed deployment later.
- Chat history is stored in a lightweight SQLite database under `data/history.sqlite3`.

## Next Improvements

- Dockerize backend and frontend
- Add authentication and document-level access control
- Move uploads to object storage
- Add background indexing jobs for large file sets
- Add observability with structured logs and metrics
