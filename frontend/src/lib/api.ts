export type SourceCitation = {
  source: string;
  page?: number | null;
  chunk_id?: string | null;
  score?: number | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceCitation[];
  streaming?: boolean;
};

export type QueryRequest = {
  question: string;
  session_id: string;
  stream?: boolean;
  top_k?: number;
  multi_query_count?: number;
  rewrite_query?: boolean;
  rerank?: boolean;
};

export type HistoryItem = {
  role: "user" | "assistant";
  content: string;
  created_at: string;
  sources: SourceCitation[];
};

export type UploadResponse = {
  document_id: string;
  file_name: string;
  status: string;
  chunk_count: number;
  skipped: boolean;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<T>;
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  return parseJsonResponse<UploadResponse>(response);
}

export async function fetchHistory(sessionId: string): Promise<HistoryItem[]> {
  const response = await fetch(
    `${API_BASE}/history?session_id=${encodeURIComponent(sessionId)}`,
  );
  const payload = await parseJsonResponse<{ messages: HistoryItem[] }>(
    response,
  );
  return payload.messages;
}

export async function streamQuery(
  request: QueryRequest,
  handlers: {
    onSources: (
      sources: SourceCitation[],
      rewrittenQuery?: string,
      contextFound?: boolean,
    ) => void;
    onToken: (chunk: string) => void;
    onDone: (answer: string) => void;
  },
): Promise<void> {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...request, stream: true }),
  });

  if (!response.ok || !response.body) {
    throw new Error(await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      const event = JSON.parse(trimmed) as
        | {
            type: "sources";
            sources: SourceCitation[];
            rewritten_query?: string;
            context_found?: boolean;
          }
        | { type: "token"; content: string }
        | { type: "done"; answer: string };

      if (event.type === "sources") {
        handlers.onSources(
          event.sources,
          event.rewritten_query,
          event.context_found,
        );
      }

      if (event.type === "token") {
        handlers.onToken(event.content);
      }

      if (event.type === "done") {
        handlers.onDone(event.answer);
      }
    }
  }
}
