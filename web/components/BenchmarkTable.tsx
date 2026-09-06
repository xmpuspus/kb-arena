"use client";

import { useState } from "react";
import {
  STRATEGY_LABELS,
  STRATEGY_DESCRIPTIONS,
  TIER_INFO,
  type Strategy,
} from "@/lib/api";
import InfoTip from "@/components/InfoTip";
import MetricNote, { type MetricKey } from "@/components/MetricNote";

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
type SortDirection = "asc" | "desc";

interface SortableHeaderProps {
  column: SortKey;
  label: string;
  sortKey: SortKey;
  sortDirection: SortDirection;
  onSort: (key: SortKey) => void;
  /** The metric this column reports, when the column reports one. */
  note?: MetricKey;
}

function SortableHeader({
  column,
  label,
  sortKey,
  sortDirection,
  onSort,
  note,
}: SortableHeaderProps) {
  const active = sortKey === column;
  return (
    <th
      className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider select-none whitespace-nowrap"
      style={{ color: active ? "var(--accent)" : "var(--muted)" }}
      aria-sort={active ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}
    >
      <span className="inline-flex items-center gap-1">
        <button
          type="button"
          onClick={() => onSort(column)}
          className="cursor-pointer uppercase tracking-wider"
          style={{ color: "inherit" }}
        >
          {label} {active ? (sortDirection === "asc" ? "^" : "v") : ""}
        </button>
        {note && <MetricNote metric={note} align="left" />}
      </span>
    </th>
  );
}

interface TierHeaderProps extends Omit<SortableHeaderProps, "column" | "label"> {
  tierNumber: number;
}

function TierHeader({
  tierNumber,
  sortKey,
  sortDirection,
  onSort,
}: TierHeaderProps) {
  const column = `tier${tierNumber}` as SortKey;
  const active = sortKey === column;
  const info = TIER_INFO[tierNumber];
  return (
    <th
      className="px-3 py-2 text-center select-none"
      style={{ color: active ? "var(--accent)" : "var(--muted)" }}
      aria-sort={active ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(column)}
        className="text-[10px] font-medium uppercase tracking-wider cursor-pointer"
        style={{ color: "var(--muted)", opacity: 0.7 }}
      >
        Tier {tierNumber} {active ? (sortDirection === "asc" ? "^" : "v") : ""}
      </button>
      <div className="text-xs font-semibold flex items-center justify-center gap-0.5">
        {info?.label ?? `T${tierNumber}`}
        {info && <InfoTip text={info.description} />}
      </div>
    </th>
  );
}

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
  const [sortDir, setSortDir] = useState<SortDirection>("desc");

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

  return (
    <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr style={{ background: "var(--background)", borderBottom: "2px solid var(--border)" }}>
            <SortableHeader column="strategy" label="Strategy" sortKey={sortKey} sortDirection={sortDir} onSort={handleSort} />
            {Array.from({ length: tierCount }, (_, i) => (
              <TierHeader key={i} tierNumber={i + 1} sortKey={sortKey} sortDirection={sortDir} onSort={handleSort} />
            ))}
            <SortableHeader column="avg" label="Avg %" sortKey={sortKey} sortDirection={sortDir} onSort={handleSort} note="accuracy" />
            <SortableHeader column="latencyMs" label="Latency" sortKey={sortKey} sortDirection={sortDir} onSort={handleSort} note="latency" />
            <SortableHeader column="costUsd" label="Cost/Q" sortKey={sortKey} sortDirection={sortDir} onSort={handleSort} note="cost_per_query" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const a = avg(row.tiers);
            return (
              <tr
                key={row.strategy}
                className="border-b last:border-0 hover:opacity-80 transition-opacity"
                style={{ borderColor: "var(--border)" }}
              >
                <td className="px-3 py-2 font-medium" style={{ color: "var(--foreground)" }}>
                  <span className="inline-flex items-center gap-0.5">
                    {STRATEGY_LABELS[row.strategy]}
                    <InfoTip text={STRATEGY_DESCRIPTIONS[row.strategy]} align="left" />
                  </span>
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
      <p className="px-3 py-1.5 text-xs" style={{ color: "var(--muted)", background: "var(--background)" }}>
        Results from your benchmark runs. Run <code className="mono">kb-arena benchmark</code> to regenerate with different corpora or strategies.
      </p>
    </div>
  );
}
