"use client";

interface Props {
  text: string;
  align?: "center" | "left";
}

export default function InfoTip({ text, align = "center" }: Props) {
  const positionClass =
    align === "left"
      ? "absolute left-0 top-full mt-1.5"
      : "absolute left-1/2 top-full mt-1.5 -translate-x-1/2";

  return (
    <span className="relative inline-flex items-center ml-1 group">
      <span
        className="inline-flex items-center justify-center rounded-full cursor-help"
        style={{
          width: 14,
          height: 14,
          fontSize: 9,
          fontWeight: 700,
          fontStyle: "italic",
          color: "var(--muted)",
          border: "1px solid var(--border)",
          lineHeight: 1,
        }}
        aria-label="Info"
      >
        i
      </span>
      <span
        role="tooltip"
        className={`${positionClass} z-50 px-3 py-2 rounded-md text-xs font-normal normal-case tracking-normal pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150`}
        style={{
          width: 280,
          color: "var(--foreground)",
          background: "var(--card)",
          border: "1px solid var(--border)",
          boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
          lineHeight: 1.5,
          whiteSpace: "normal",
          textAlign: "left",
        }}
      >
        {text}
      </span>
    </span>
  );
}
