"use client";

/**
 * One shape for every failed read: what failed, what to do, and a way back.
 *
 * A failed read must never leave sample content or an earlier answer on
 * screen, because both read as this corpus's data. The route drops the data
 * and renders this in its place.
 */
interface Props {
  title: string;
  message: string;
  hint?: string;
  onRetry: () => void;
}

export default function FetchError({ title, message, hint, onRetry }: Props) {
  return (
    <div
      role="alert"
      className="space-y-2 rounded-lg border px-4 py-3 text-sm"
      style={{ borderColor: "var(--danger)", background: "var(--card)" }}
    >
      <p className="font-semibold" style={{ color: "var(--danger)" }}>
        {title}
      </p>
      <p style={{ color: "var(--foreground)" }}>{message}</p>
      {hint && (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          {hint}
        </p>
      )}
      <button
        type="button"
        onClick={onRetry}
        className="rounded border px-3 py-1.5 text-xs font-medium transition-opacity hover:opacity-70"
        style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
      >
        Try again
      </button>
    </div>
  );
}
