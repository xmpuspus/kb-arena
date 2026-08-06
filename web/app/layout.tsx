import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import ClientErrorBoundary from "@/components/ClientErrorBoundary";

export const metadata: Metadata = {
  title: "KB Arena | Retrieval Architecture Decision Lab",
  description:
    "Compare retrieval architectures on your documentation with reproducible quality, latency, cost, and reliability evidence.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen" style={{ background: "var(--background)", color: "var(--foreground)" }}>
        <Nav />
        <main>
          <ClientErrorBoundary>{children}</ClientErrorBoundary>
        </main>
        <footer className="border-t mt-16 py-8 text-center text-sm" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
          <a
            href="https://github.com/xmpuspus/kb-arena"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex min-h-11 items-center px-2 hover:opacity-70 transition-opacity"
            style={{ color: "var(--accent)" }}
          >
            GitHub
          </a>
          <span className="mx-2">|</span>
          <span>KB Arena | Retrieval Architecture Decision Lab</span>
        </footer>
      </body>
    </html>
  );
}
