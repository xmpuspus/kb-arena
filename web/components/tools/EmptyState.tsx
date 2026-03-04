"use client";

import { useState } from "react";

interface Props {
  title: string;
  description: string;
  command?: string;
}

export default function EmptyState({ title, description, command }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!command) return;
    await navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      <h3 className="text-lg font-semibold mb-2" style={{ color: "var(--foreground)" }}>
        {title}
      </h3>
      <p className="text-sm mb-4 max-w-md" style={{ color: "var(--muted)" }}>
        {description}
      </p>
      {command && (
        <div className="relative">
          <pre
            className="mono text-xs px-4 py-2 rounded-lg border cursor-pointer hover:opacity-80 transition-opacity"
            style={{ background: "var(--background)", borderColor: "var(--border)", color: "var(--foreground)" }}
            onClick={handleCopy}
          >
            {command}
          </pre>
          {copied && (
            <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs px-2 py-0.5 rounded" style={{ background: "var(--accent)", color: "#fff" }}>
              Copied!
            </span>
          )}
        </div>
      )}
    </div>
  );
}
