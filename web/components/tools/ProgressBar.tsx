"use client";

interface Props {
  current: number;
  total: number;
  label?: string;
  sublabel?: string;
}

export default function ProgressBar({ current, total, label, sublabel }: Props) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;

  return (
    <div className="space-y-1">
      {(label || sublabel) && (
        <div className="flex items-center justify-between text-xs">
          {label && <span style={{ color: "var(--foreground)" }}>{label}</span>}
          {sublabel && <span style={{ color: "var(--muted)" }}>{sublabel}</span>}
        </div>
      )}
      <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: "var(--border)" }}>
        <div
          className="h-full rounded-full transition-all duration-300 ease-out"
          style={{ width: `${pct}%`, background: "var(--accent)" }}
        />
      </div>
      <div className="text-xs text-right" style={{ color: "var(--muted)" }}>
        {current} / {total} ({pct}%)
      </div>
    </div>
  );
}
