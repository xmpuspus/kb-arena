"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

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
      className="sticky top-0 z-50 border-b px-6 py-3 flex items-center gap-8"
      style={{ background: "var(--card)", borderColor: "var(--border)" }}
    >
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
    </nav>
  );
}
