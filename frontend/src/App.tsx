import { useEffect, useMemo, useState } from "react";
import {
  fetchHistory,
  streamQuery,
  uploadDocument,
  type ChatMessage,
  type HistoryItem,
  type SourceCitation,
  type UploadResponse,
} from "./lib/api";
import { MessageBubble } from "./components/MessageBubble";
import { HistoryPanel } from "./components/HistoryPanel";
import { UploadCard } from "./components/UploadCard";

function createSessionId() {
  return crypto.randomUUID();
}

function toMessage(item: HistoryItem, index: number): ChatMessage {
  return {
    id: `${item.created_at}-${index}`,
    role: item.role,
    content: item.content,
    sources: item.sources,
  };
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => createSessionId());
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadState, setUploadState] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceCitation[]>([]);
  const [statusText, setStatusText] = useState("Ready");

  useEffect(() => {
    let cancelled = false;

    async function loadHistory() {
      try {
        const items = await fetchHistory(sessionId);
        if (cancelled) return;
        setHistory(items);
        setMessages(items.map(toMessage));
      } catch {
        if (!cancelled) {
          setHistory([]);
          setMessages([]);
        }
      }
    }

    void loadHistory();

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const hasMessages = useMemo(() => messages.length > 0, [messages.length]);

  async function handleUpload(file: File) {
    setUploading(true);
    setError(null);
    setStatusText("Uploading document");

    try {
      const response = await uploadDocument(file);
      setUploadState(response);
      setStatusText(
        response.skipped ? "Document was already indexed" : "Document indexed",
      );
    } catch (uploadError) {
      setError(
        uploadError instanceof Error ? uploadError.message : "Upload failed",
      );
      setStatusText("Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk(nextQuestion: string) {
    const trimmed = nextQuestion.trim();
    if (!trimmed || loading) return;

    setError(null);
    setLoading(true);
    setStatusText("Retrieving context");
    setSources([]);

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
      sources: [],
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setQuestion("");

    try {
      await streamQuery(
        {
          question: trimmed,
          session_id: sessionId,
          stream: true,
          top_k: 4,
          multi_query_count: 3,
          rewrite_query: true,
          rerank: true,
        },
        {
          onSources: (nextSources) => {
            setSources(nextSources);
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, sources: nextSources }
                  : message,
              ),
            );
          },
          onToken: (chunk) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: `${message.content}${chunk}`,
                      streaming: true,
                    }
                  : message,
              ),
            );
          },
          onDone: (answer) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: answer, streaming: false }
                  : message,
              ),
            );
            setStatusText("Answer ready");
          },
        },
      );

      const refreshedHistory = await fetchHistory(sessionId);
      setHistory(refreshedHistory);
    } catch (askError) {
      setError(askError instanceof Error ? askError.message : "Query failed");
      setStatusText("Query failed");
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: "Something went wrong while generating the answer.",
                streaming: false,
              }
            : message,
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  function startNewChat() {
    setSessionId(createSessionId());
    setMessages([]);
    setHistory([]);
    setSources([]);
    setUploadState(null);
    setError(null);
    setQuestion("");
    setStatusText("Ready");
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <h1>RAG Studio</h1>
            <p>Production chat over PDFs</p>
          </div>
        </div>

        <UploadCard
          onUpload={handleUpload}
          status={uploadState}
          loading={uploading}
        />

        <section className="panel panel-meta">
          <div className="panel-header">
            <h2>Session</h2>
            <span>{sessionId.slice(0, 8)}</span>
          </div>
          <button
            className="secondary-button"
            onClick={startNewChat}
            type="button"
          >
            New chat
          </button>
          <p className="muted">{statusText}</p>
          <p className="muted">
            Indexed history is stored on the backend, so the conversation
            survives refreshes.
          </p>
        </section>

        <HistoryPanel items={history} onSelect={handleAsk} />
      </aside>

      <main className="workspace">
        <header className="hero">
          <div>
            <span className="eyebrow">FastAPI + LangChain + Vector DB</span>
            <h2>
              Ask questions over uploaded documents with streaming answers and
              source attribution.
            </h2>
          </div>
          <div className="hero-stats">
            <div>
              <strong>
                {
                  messages.filter((message) => message.role === "assistant")
                    .length
                }
              </strong>
              <span>answers</span>
            </div>
            <div>
              <strong>{sources.length}</strong>
              <span>current sources</span>
            </div>
          </div>
        </header>

        <section className="chat-panel">
          <div className="chat-stream">
            {hasMessages ? (
              messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))
            ) : (
              <div className="empty-state">
                <h3>Upload a document, then ask a question.</h3>
                <p>
                  The backend rewrites your query, runs multi-query retrieval,
                  optionally reranks the results, and streams the answer back.
                </p>
              </div>
            )}
          </div>

          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              void handleAsk(question);
            }}
          >
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask something specific about the uploaded document..."
              rows={3}
              disabled={loading}
            />
            <div className="composer-actions">
              <span className={`live-chip ${loading ? "active" : ""}`}>
                {loading ? "Streaming" : "Idle"}
              </span>
              <button
                className="primary-button"
                type="submit"
                disabled={loading || !question.trim()}
              >
                {loading ? "Sending…" : "Send question"}
              </button>
            </div>
          </form>

          {error ? <p className="error-banner">{error}</p> : null}
        </section>
      </main>
    </div>
  );
}
