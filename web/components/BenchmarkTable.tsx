"use client";

import { useState } from "react";
import { STRATEGY_LABELS, type Strategy } from "@/lib/api";

interface Row {
  strategy: Strategy;
  tiers: number[];
  latencyMs: number;
  costUsd: number;
}

interface Props {
  rows: Row[];
}

type SortKey = "strategy" | "avg" | "latencyMs" | "costUsd" | `tier${number}`;

function avg(tiers: number[]) {
  return tiers.reduce((a, b) => a + b, 0) / tiers.length;
}

function accuracyColor(val: number) {
  if (val >= 80) return "var(--success)";
  if (val >= 50) return "var(--warning)";
  return "var(--danger)";
}

export default function BenchmarkTable({ rows }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("avg");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  const sorted = [...rows].sort((a, b) => {
    let av = 0;
    let bv = 0;
    if (sortKey === "strategy") {
      av = 0;
      bv = STRATEGY_LABELS[a.strategy].localeCompare(STRATEGY_LABELS[b.strategy]);
      return sortDir === "asc" ? (bv > 0 ? 1 : -1) : (bv > 0 ? -1 : 1);
    } else if (sortKey === "avg") {
      av = avg(a.tiers);
      bv = avg(b.tiers);
    } else if (sortKey === "latencyMs") {
      av = a.latencyMs;
      bv = b.latencyMs;
    } else if (sortKey === "costUsd") {
      av = a.costUsd;
      bv = b.costUsd;
    } else {
      const idx = parseInt(sortKey.replace("tier", "")) - 1;
      av = a.tiers[idx] ?? 0;
      bv = b.tiers[idx] ?? 0;
    }
    return sortDir === "asc" ? av - bv : bv - av;
  });

  const tierCount = rows[0]?.tiers.length ?? 5;

  function Th({ k, label }: { k: SortKey; label: string }) {
    const active = sortKey === k;
    return (
      <th
        onClick={() => handleSort(k)}
        className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer select-none whitespace-nowrap"
        style={{ color: active ? "var(--accent)" : "var(--muted)" }}
      >
        {label} {active ? (sortDir === "asc" ? "↑" : "↓") : ""}
      </th>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr style={{ background: "var(--card)", borderBottom: "1px solid var(--border)" }}>
            <Th k="strategy" label="Strategy" />
            {Array.from({ length: tierCount }, (_, i) => (
              <Th key={i} k={`tier${i + 1}`} label={`Tier ${i + 1}`} />
            ))}
            <Th k="avg" label="Avg %" />
            <Th k="latencyMs" label="Latency" />
            <Th k="costUsd" label="Cost/Q" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const a = avg(row.tiers);
            const costPerCorrect = a > 0 ? (row.costUsd / (a / 100)).toFixed(4) : "—";
            return (
              <tr
                key={row.strategy}
                className="border-b last:border-0 hover:opacity-80 transition-opacity"
                style={{ borderColor: "var(--border)" }}
              >
                <td className="px-3 py-2 font-medium" style={{ color: "var(--foreground)" }}>
                  {STRATEGY_LABELS[row.strategy]}
                </td>
                {row.tiers.map((t, i) => (
                  <td key={i} className="px-3 py-2 mono text-center">
                    <span
                      className="px-1.5 py-0.5 rounded text-xs font-semibold"
                      style={{
                        color: accuracyColor(t),
                        background: `${accuracyColor(t)}22`,
                      }}
                    >
                      {t}%
                    </span>
                  </td>
                ))}
                <td className="px-3 py-2 mono text-center font-semibold" style={{ color: accuracyColor(a) }}>
                  {a.toFixed(1)}%
                </td>
                <td className="px-3 py-2 mono text-center" style={{ color: "var(--muted)" }}>
                  {row.latencyMs.toFixed(0)} ms
                </td>
                <td className="px-3 py-2 mono text-center" style={{ color: "var(--muted)" }}>
                  ${row.costUsd.toFixed(4)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="px-3 py-1.5 text-xs" style={{ color: "var(--muted)", background: "var(--card)" }}>
        [PENDING BENCHMARK] — placeholder data. Run <code className="mono">kb-arena benchmark</code> to populate.
      </p>
    </div>
  );
}
