"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ThemeToggle from "./ThemeToggle";

const links = [
  { href: "/", label: "Home" },
  { href: "/demo", label: "Demo" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/graph", label: "Graph" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav
      className="sticky top-0 z-50 border-b px-6 py-3 flex items-center justify-between"
      style={{ background: "var(--background)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center gap-8">
        <Link href="/" className="font-bold text-lg tracking-tight" style={{ color: "var(--foreground)" }}>
          KB Arena
        </Link>
        <div className="hidden sm:flex items-center gap-6">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="text-sm transition-colors"
              style={{
                color: pathname === l.href ? "var(--accent)" : "var(--muted)",
              }}
            >
              {l.label}
            </Link>
          ))}
        </div>
      </div>
      <ThemeToggle />
    </nav>
  );
}
