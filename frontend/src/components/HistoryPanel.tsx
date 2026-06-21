import type { HistoryItem } from "../lib/api";

type Props = {
  items: HistoryItem[];
  onSelect: (question: string) => void;
};

export function HistoryPanel({ items, onSelect }: Props) {
  const turns = items.filter((item) => item.role === "user");

  return (
    <section className="panel panel-history">
      <div className="panel-header">
        <h2>Conversation History</h2>
        <span>{turns.length} turns</span>
      </div>

      <div className="history-list">
        {turns.length === 0 ? (
          <p className="muted">
            No messages yet. Ask a question to populate the history.
          </p>
        ) : (
          turns.map((item, index) => (
            <button
              className="history-item"
              key={`${item.created_at}-${index}`}
              onClick={() => onSelect(item.content)}
            >
              <span className="history-index">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="history-text">{item.content}</span>
            </button>
          ))
        )}
      </div>
    </section>
  );
}
