"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { clearApiToken, getApiToken, setApiToken } from "@/lib/auth";

const links = [
  { href: "/", label: "Home" },
  { href: "/demo", label: "Demo" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/retriever-lab", label: "Retriever Lab" },
  { href: "/graph", label: "Graph" },
  { href: "/arena", label: "Arena" },
  { href: "/tools", label: "Tools" },
];

export default function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [tokenOpen, setTokenOpen] = useState(false);
  const [token, setToken] = useState("");
  const [hasToken, setHasToken] = useState(false);
  const tokenTriggerRef = useRef<HTMLButtonElement>(null);
  const tokenDialogRef = useRef<HTMLDivElement>(null);
  const tokenInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setHasToken(Boolean(getApiToken()));
  }, []);

  useEffect(() => {
    if (!tokenOpen) return;
    const tokenTrigger = tokenTriggerRef.current;
    tokenInputRef.current?.focus();
    return () => tokenTrigger?.focus();
  }, [tokenOpen]);

  function openTokenDialog() {
    setToken(getApiToken());
    setTokenOpen(true);
  }

  function saveToken() {
    setApiToken(token);
    setHasToken(Boolean(token.trim()));
    setTokenOpen(false);
  }

  function removeToken() {
    clearApiToken();
    setToken("");
    setHasToken(false);
    setTokenOpen(false);
  }

  return (
    <>
      <nav
        className="sticky top-0 z-50 border-b px-6 py-3"
        style={{ background: "var(--card)", borderColor: "var(--border)" }}
      >
        <div className="flex items-center justify-between">
          <div className="flex min-w-0 items-center gap-8">
            <Link href="/" className="inline-flex min-h-11 shrink-0 items-center font-bold text-lg tracking-tight" style={{ color: "var(--foreground)" }}>
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
          <div className="flex items-center gap-1">
            <button
              ref={tokenTriggerRef}
              type="button"
              className="relative flex min-h-11 min-w-11 items-center justify-center rounded border transition-colors"
              style={{
                color: hasToken ? "var(--success)" : "var(--muted)",
                borderColor: "var(--border)",
              }}
              onClick={openTokenDialog}
              aria-label="API access"
              title="API access"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="7.5" cy="15.5" r="3.5" />
                <path d="m10 13 8-8 2 2-2 2 1.5 1.5-2 2L16 11l-3.5 3.5" />
              </svg>
              {hasToken && (
                <span
                  className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full"
                  style={{ background: "var(--success)" }}
                />
              )}
            </button>
            <button
              type="button"
              className="sm:hidden min-h-11 min-w-11 p-2 rounded"
              style={{ color: "var(--muted)" }}
              onClick={() => setOpen(!open)}
              aria-label="Toggle navigation"
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                {open ? (
                  <path d="M5 5l10 10M15 5L5 15" />
                ) : (
                  <path d="M3 5h14M3 10h14M3 15h14" />
                )}
              </svg>
            </button>
          </div>
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
                  className="flex min-h-11 items-center text-sm px-3 py-2 rounded transition-colors"
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

      {tokenOpen && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 px-4"
          role="presentation"
          onMouseDown={(event) => {
            if (event.currentTarget === event.target) setTokenOpen(false);
          }}
        >
          <div
            ref={tokenDialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="api-access-title"
            className="w-full max-w-sm rounded-lg border p-5 shadow-xl"
            style={{ background: "var(--card)", borderColor: "var(--border)" }}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                setTokenOpen(false);
                return;
              }
              if (event.key !== "Tab") return;

              const focusable = Array.from(
                tokenDialogRef.current?.querySelectorAll<HTMLElement>(
                  'button:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])',
                ) ?? [],
              );
              if (focusable.length === 0) return;

              const first = focusable[0];
              const last = focusable[focusable.length - 1];
              if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
              } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
              }
            }}
          >
            <div className="flex items-center justify-between gap-4">
              <h2 id="api-access-title" className="text-base font-semibold">
                API access
              </h2>
              <button
                type="button"
                className="flex min-h-11 min-w-11 items-center justify-center rounded"
                style={{ color: "var(--muted)" }}
                onClick={() => setTokenOpen(false)}
                aria-label="Close"
                title="Close"
              >
                <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <path d="M5 5l10 10M15 5 5 15" />
                </svg>
              </button>
            </div>
            <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
              Stored in this browser tab until it closes.
            </p>
            <label htmlFor="api-token" className="mt-4 block text-sm font-medium">
              Bearer token
            </label>
            <input
              ref={tokenInputRef}
              id="api-token"
              type="password"
              autoComplete="off"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") saveToken();
              }}
              className="mt-2 w-full rounded border px-3 py-2 text-sm outline-none"
              style={{
                background: "var(--background)",
                borderColor: "var(--border)",
                color: "var(--foreground)",
              }}
            />
            <div className="mt-5 flex items-center justify-between gap-3">
              <button
                type="button"
                onClick={removeToken}
                disabled={!hasToken}
                className="px-3 py-2 text-sm font-medium disabled:opacity-40"
                style={{ color: "var(--danger)" }}
              >
                Clear
              </button>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setTokenOpen(false)}
                  className="rounded border px-3 py-2 text-sm font-medium"
                  style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={saveToken}
                  className="rounded px-3 py-2 text-sm font-medium text-white"
                  style={{ background: "var(--accent)" }}
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
