"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/demo", label: "Demo" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/graph", label: "Graph" },
  { href: "/tools", label: "Tools" },
];

export default function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <nav
      className="sticky top-0 z-50 border-b px-6 py-3"
      style={{ background: "var(--card)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link href="/" className="font-bold text-lg tracking-tight" style={{ color: "var(--foreground)" }}>
            KB Arena
          </Link>
          <div className="hidden sm:flex items-center gap-1">
            {links.map((l) => {
              const active = pathname === l.href;
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  className="text-sm px-3 py-1.5 transition-colors"
                  style={{
                    color: active ? "var(--accent)" : "var(--muted)",
                    borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
                  }}
                >
                  {l.label}
                </Link>
              );
            })}
          </div>
        </div>
        <button
          className="sm:hidden p-2 rounded"
          style={{ color: "var(--muted)" }}
          onClick={() => setOpen(!open)}
          aria-label="Toggle navigation"
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
            {open ? (
              <path d="M5 5l10 10M15 5L5 15" />
            ) : (
              <path d="M3 5h14M3 10h14M3 15h14" />
            )}
          </svg>
        </button>
      </div>
      {open && (
        <div className="sm:hidden pt-3 pb-1 flex flex-col gap-1">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className="text-sm px-3 py-2 rounded transition-colors"
                style={{
                  color: active ? "var(--accent)" : "var(--muted)",
                  background: active ? "var(--background)" : "transparent",
                }}
              >
                {l.label}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}
