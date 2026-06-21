from __future__ import annotations

from dataclasses import dataclass
import unittest
import json

from langchain_core.documents import Document

from app.core.config import Settings
from app.rag.pipeline import RAGPipeline
from app.models.schemas import QueryRequest, SourceCitation
from app.services.chat_service import ChatService


@dataclass
class FakeVectorStore:
    documents: list[Document]

    def add_documents(self, documents: list[Document]) -> None:
        self.documents.extend(documents)

    def search(self, query: str, top_k: int, fetch_k: int | None = None) -> list[Document]:
        return self.documents[:top_k]


class FakeLLM:
    def __init__(self) -> None:
        self.rewrite_calls = 0
        self.multi_query_calls = 0
        self.answer_calls = 0

    def invoke(self, messages):
        prompt = messages[-1].content
        if "alternate search queries" in prompt.lower():
            self.multi_query_calls += 1
            return type("Response", (), {"content": "query one\nquery two\nquery three"})()
        if "rewrite the user question" in prompt.lower() or "rewrite search queries" in prompt.lower():
            self.rewrite_calls += 1
            return type("Response", (), {"content": "rewritten query"})()
        self.answer_calls += 1
        return type("Response", (), {"content": "answer from context"})()

    def stream(self, messages):
        yield type("Chunk", (), {"content": "answer"})()
        yield type("Chunk", (), {"content": " from"})()
        yield type("Chunk", (), {"content": " stream"})()


class FakeHistoryService:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []

    def add_message(self, session_id: str, role: str, content: str, sources=None) -> None:
        self.messages.append((session_id, role, content))


class StreamingPipelineStub:
    def stream_answer(self, query: str, *, top_k: int | None = None, fetch_k: int | None = None):
        context = type(
            "Context",
            (),
            {
                "rewritten_query": query,
                "documents": [Document(page_content="content", metadata={"source": "doc.pdf", "page": 1})],
                "sources": [SourceCitation(source="doc.pdf", page=1, chunk_id="1")],
                "retrieval_mode": "direct",
                "context_truncated": False,
            },
        )()

        def generator():
            yield "partial"
            raise RuntimeError("stream interrupted")

        return context, generator(), True


class PipelineTests(unittest.TestCase):
    def test_simple_query_skips_expansion(self) -> None:
        settings = Settings(
            enable_query_rewrite=True,
            enable_multi_query=True,
            enable_reranking=False,
            retriever_top_k=2,
            retriever_fetch_k=2,
        )
        vector_store = FakeVectorStore(
            documents=[
                Document(page_content="Paris is the capital of France.", metadata={"source": "doc.pdf", "page": 1}),
                Document(page_content="Berlin is the capital of Germany.", metadata={"source": "doc.pdf", "page": 2}),
            ]
        )

        llm = FakeLLM()
        pipeline = RAGPipeline(settings=settings, vector_store=vector_store, llm=llm)
        answer, context, context_found = pipeline.answer("What is RAG?")

        self.assertTrue(context_found)
        self.assertTrue(answer)
        self.assertEqual(context.sources[0].source, "doc.pdf")
        self.assertEqual(context.rewritten_query, "What is RAG?")
        self.assertEqual(llm.rewrite_calls, 0)
        self.assertEqual(llm.multi_query_calls, 0)
        self.assertEqual(llm.answer_calls, 1)

    def test_complex_query_uses_expansion(self) -> None:
        settings = Settings(
            enable_query_rewrite=True,
            enable_multi_query=True,
            enable_reranking=False,
            retriever_top_k=2,
            retriever_fetch_k=2,
        )
        llm = FakeLLM()
        pipeline = RAGPipeline(settings=settings, vector_store=FakeVectorStore([Document(page_content="x", metadata={"source": "doc.pdf", "page": 1})]), llm=llm)

        answer, context, context_found = pipeline.answer("Compare the major ideas discussed in section one versus section two of the document")

        self.assertTrue(context_found)
        self.assertTrue(answer)
        self.assertNotEqual(context.rewritten_query, "Compare the major ideas discussed in section one versus section two of the document")
        self.assertTrue(context.used_rewrite)
        self.assertTrue(context.used_multi_query)
        self.assertEqual(llm.rewrite_calls, 1)
        self.assertEqual(llm.multi_query_calls, 1)

    def test_context_budget_truncates_large_payloads(self) -> None:
        settings = Settings(
            enable_query_rewrite=False,
            enable_multi_query=False,
            enable_reranking=False,
            max_context_chars=180,
        )
        vector_store = FakeVectorStore(
            documents=[
                Document(page_content="A" * 200, metadata={"source": "doc.pdf", "page": 1}),
                Document(page_content="B" * 200, metadata={"source": "doc.pdf", "page": 2}),
            ]
        )
        pipeline = RAGPipeline(settings=settings, vector_store=vector_store, llm=FakeLLM())

        context = pipeline.retrieve("Summarize the document")

        self.assertLessEqual(len(context.context_text), 180)
        self.assertTrue(context.context_truncated)

    def test_pipeline_handles_empty_store(self) -> None:
        settings = Settings(enable_query_rewrite=False, enable_multi_query=False, enable_reranking=False)
        pipeline = RAGPipeline(settings=settings, vector_store=FakeVectorStore([]), llm=FakeLLM())

        answer, context, context_found = pipeline.answer("Unknown question")

        self.assertFalse(context_found)
        self.assertIn("could not find", answer.lower())
        self.assertEqual(context.documents, [])

    def test_stream_failure_still_finalizes_history(self) -> None:
        history = FakeHistoryService()
        service = ChatService(settings=Settings(), pipeline=StreamingPipelineStub(), history=history)

        stream = service.stream_answer(QueryRequest(question="What is RAG?", session_id="session-1", stream=True))
        payloads = [json.loads(item) for item in stream]

        self.assertEqual(payloads[0]["type"], "sources")
        self.assertEqual(payloads[-1]["type"], "done")
        self.assertTrue(any(message[1] == "assistant" for message in history.messages))


if __name__ == "__main__":
    unittest.main()
