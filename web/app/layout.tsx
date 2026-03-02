import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "KB Arena — Knowledge Graph vs Vector RAG Benchmark",
  description:
    "Benchmark 5 retrieval strategies on real documentation. Knowledge graphs vs vector RAG, side-by-side.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen" style={{ background: "var(--background)", color: "var(--foreground)" }}>
        <Nav />
        <main>{children}</main>
        <footer className="border-t mt-16 py-8 text-center text-sm" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
          <a
            href="https://github.com/xavier/kb-arena"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-70 transition-opacity"
            style={{ color: "var(--accent)" }}
          >
            GitHub
          </a>
          <span className="mx-2">·</span>
          <span>KB Arena — Knowledge Graph Benchmark</span>
        </footer>
      </body>
    </html>
  );
}
