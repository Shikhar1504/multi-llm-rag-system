import type { ChatMessage } from "../lib/api";

type Props = {
  message: ChatMessage;
};

export function MessageBubble({ message }: Props) {
  return (
    <article className={`message message-${message.role}`}>
      <div className="message-badge">
        {message.role === "user" ? "You" : "Assistant"}
      </div>
      <div className="message-content">
        <p>{message.content || (message.streaming ? "Thinking…" : "")}</p>
        {message.sources && message.sources.length > 0 ? (
          <div className="source-list">
            {message.sources.map((source, index) => (
              <span className="source-pill" key={`${source.source}-${index}`}>
                {source.source}
                {typeof source.page === "number" ? ` • p.${source.page}` : ""}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}
