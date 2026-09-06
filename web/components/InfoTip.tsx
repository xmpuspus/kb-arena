"use client";

import { useId, useState } from "react";

/**
 * The short note beside a label, reachable by keyboard.
 *
 * The first version opened on hover only, from a span the tab order never
 * reached. A reader on a keyboard saw the letter and never the text. The
 * trigger is a button now, it opens on focus as well as on hover, and Escape
 * closes it.
 */
interface Props {
  text: string;
  /** What the note is about. It becomes the button's accessible name. */
  label?: string;
  align?: "center" | "left";
}

export default function InfoTip({ text, label, align = "center" }: Props) {
  const [open, setOpen] = useState(false);
  const id = useId();

  const positionClass =
    align === "left"
      ? "absolute left-0 top-full mt-1.5"
      : "absolute left-1/2 top-full mt-1.5 -translate-x-1/2";

  return (
    <span
      className="relative inline-flex items-center ml-1"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        className="info-tip inline-flex items-center justify-center rounded-full"
        style={{
          width: 16,
          height: 16,
          fontSize: 9,
          fontWeight: 700,
          fontStyle: "italic",
          color: "var(--muted)",
          border: "1px solid var(--border-strong)",
          lineHeight: 1,
        }}
        aria-label={label ? `About ${label}` : "More information"}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onClick={() => setOpen((was) => !was)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(event) => {
          if (event.key === "Escape") setOpen(false);
        }}
      >
        i
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className={`${positionClass} z-50 px-3 py-2 rounded-md text-xs font-normal normal-case tracking-normal pointer-events-none`}
          style={{
            // A fixed 280px panel near the right edge widened the document at
            // 375px. The clamp keeps it inside the viewport instead.
            width: "min(280px, calc(100vw - 2rem))",
            color: "var(--foreground)",
            background: "var(--card)",
            border: "1px solid var(--border-strong)",
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
            lineHeight: 1.5,
            whiteSpace: "normal",
            textAlign: "left",
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
}
