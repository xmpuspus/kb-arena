"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [dark, setDark] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    const isDark = stored ? stored === "dark" : true;
    setDark(isDark);
    document.documentElement.classList.toggle("light", !isDark);
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    localStorage.setItem("theme", next ? "dark" : "light");
    document.documentElement.classList.toggle("light", !next);
  }

  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className="p-2 rounded text-sm transition-colors hover:opacity-70"
      style={{ color: "var(--muted)" }}
    >
      {dark ? "☀ Light" : "☾ Dark"}
    </button>
  );
}
