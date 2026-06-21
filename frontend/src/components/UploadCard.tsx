import type { UploadResponse } from "../lib/api";

type Props = {
  onUpload: (file: File) => Promise<void>;
  status?: UploadResponse | null;
  loading: boolean;
};

export function UploadCard({ onUpload, status, loading }: Props) {
  return (
    <section className="panel panel-upload">
      <div className="panel-header">
        <h2>Upload PDF</h2>
        <span>Index documents</span>
      </div>

      <label className="file-dropzone">
        <input
          type="file"
          accept="application/pdf"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              void onUpload(file);
            }
          }}
          disabled={loading}
        />
        <strong>{loading ? "Indexing…" : "Choose a PDF"}</strong>
        <span>
          Recursive splitting, metadata, and vector indexing happen on the
          backend.
        </span>
      </label>

      {status ? (
        <div className="upload-status">
          <span className={`status-badge status-${status.status}`}>
            {status.status}
          </span>
          <div>
            <p>{status.file_name}</p>
            <small>
              {status.chunk_count} chunks{status.skipped ? " • cached" : ""}
            </small>
          </div>
        </div>
      ) : null}
    </section>
  );
}
