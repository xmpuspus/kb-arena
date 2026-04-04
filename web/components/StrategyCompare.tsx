"use client";

import { useState } from "react";
import {
  STRATEGY_LABELS,
  STRATEGY_COLORS,
  STRATEGY_DESCRIPTIONS,
  TIER_INFO,
  STRATEGIES,
  type Strategy,
} from "@/lib/api";

interface Row {
  strategy: Strategy;
  tiers: number[];
  latencyMs: number;
  costUsd: number;
}

interface Props {
  rows: Row[];
}

function avg(tiers: number[]) {
  return tiers.reduce((a, b) => a + b, 0) / tiers.length;
}

function accuracyColor(val: number) {
  if (val >= 80) return "var(--success)";
  if (val >= 50) return "var(--warning)";
  return "var(--danger)";
}

function Delta({ a, b }: { a: number; b: number }) {
  const diff = a - b;
  if (Math.abs(diff) < 0.05) return <span style={{ color: "var(--muted)" }}>—</span>;
  const sign = diff > 0 ? "+" : "";
  const color = diff > 0 ? "var(--success)" : "var(--danger)";
  return <span style={{ color }}>{sign}{diff.toFixed(1)}</span>;
}

function TierBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 rounded-full" style={{ background: "var(--border)" }}>
        <div
          className="h-2 rounded-full transition-all"
          style={{ width: `${value}%`, background: accuracyColor(value) }}
        />
      </div>
      <span
        className="text-xs font-semibold mono w-9 text-right"
        style={{ color: accuracyColor(value) }}
      >
        {value}%
      </span>
    </div>
  );
}

export default function StrategyCompare({ rows }: Props) {
  const available = rows.map((r) => r.strategy);

  const defaultA = available[0] ?? "qna_pairs";
  const defaultB = available[1] ?? "knowledge_graph";

  const [stratA, setStratA] = useState<Strategy>(defaultA);
  const [stratB, setStratB] = useState<Strategy>(defaultB);

  const rowA = rows.find((r) => r.strategy === stratA);
  const rowB = rows.find((r) => r.strategy === stratB);

  const tierCount = rows[0]?.tiers.length ?? 5;

  const cardStyle = {
    background: "var(--card)",
    border: "1px solid var(--border)",
    borderRadius: 8,
  };

  return (
    <div className="space-y-4">
      {/* Strategy pickers */}
      <div className="grid grid-cols-2 gap-4">
        {([
          { val: stratA, set: setStratA, other: stratB, label: "Strategy A" },
          { val: stratB, set: setStratB, other: stratA, label: "Strategy B" },
        ] as const).map(({ val, set, other, label }) => (
          <div key={label} className="space-y-1">
            <label className="text-xs font-medium" style={{ color: "var(--muted)" }}>{label}</label>
            <select
              value={val}
              onChange={(e) => set(e.target.value as Strategy)}
              className="w-full px-3 py-1.5 rounded-lg border text-sm"
              style={{
                background: "var(--card)",
                borderColor: "var(--border)",
                color: "var(--foreground)",
              }}
            >
              {STRATEGIES.filter((s) => available.includes(s)).map((s) => (
                <option key={s} value={s} disabled={s === other}>
                  {STRATEGY_LABELS[s]}
                </option>
              ))}
            </select>
            {val && (
              <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
                {STRATEGY_DESCRIPTIONS[val]}
              </p>
            )}
          </div>
        ))}
      </div>

      {rowA && rowB ? (
        <>
          {/* Tier accuracy breakdown */}
          <div style={cardStyle} className="p-4">
            <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--foreground)" }}>
              Accuracy by tier
            </h3>
            <div className="space-y-3">
              {Array.from({ length: tierCount }, (_, i) => {
                const tier = i + 1;
                const info = TIER_INFO[tier];
                return (
                  <div key={tier} className="grid grid-cols-[1fr_80px_1fr] gap-3 items-center">
                    <TierBar value={rowA.tiers[i] ?? 0} />
                    <div className="text-center">
                      <div className="text-xs font-semibold" style={{ color: "var(--foreground)" }}>
                        {info?.label ?? `T${tier}`}
                      </div>
                      <div className="text-[10px]" style={{ color: "var(--muted)" }}>Tier {tier}</div>
                    </div>
                    <TierBar value={rowB.tiers[i] ?? 0} />
                  </div>
                );
              })}
            </div>

            {/* Column headers below bars */}
            <div className="grid grid-cols-2 gap-4 mt-4 pt-3" style={{ borderTop: "1px solid var(--border)" }}>
              {[{ row: rowA, strat: stratA }, { row: rowB, strat: stratB }].map(({ row, strat }) => (
                <div key={strat} className="flex items-center gap-2">
                  <span
                    className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: STRATEGY_COLORS[strat] }}
                  />
                  <span className="text-xs font-medium" style={{ color: "var(--foreground)" }}>
                    {STRATEGY_LABELS[strat]}
                  </span>
                  <span className="text-xs mono font-semibold" style={{ color: accuracyColor(avg(row.tiers)) }}>
                    {avg(row.tiers).toFixed(1)}% avg
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Summary stats */}
          <div className="grid grid-cols-3 gap-4">
            {[
              {
                label: "Avg accuracy",
                a: avg(rowA.tiers).toFixed(1) + "%",
                b: avg(rowB.tiers).toFixed(1) + "%",
                deltaA: avg(rowA.tiers),
                deltaB: avg(rowB.tiers),
                higherIsBetter: true,
              },
              {
                label: "Latency",
                a: rowA.latencyMs.toFixed(0) + " ms",
                b: rowB.latencyMs.toFixed(0) + " ms",
                deltaA: -rowA.latencyMs,
                deltaB: -rowB.latencyMs,
                higherIsBetter: false,
              },
              {
                label: "Cost per query",
                a: "$" + rowA.costUsd.toFixed(4),
                b: "$" + rowB.costUsd.toFixed(4),
                deltaA: -rowA.costUsd,
                deltaB: -rowB.costUsd,
                higherIsBetter: false,
              },
            ].map(({ label, a, b, deltaA, deltaB }) => (
              <div key={label} style={cardStyle} className="p-4 space-y-2">
                <div className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
                  {label}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { strat: stratA, val: a, delta: deltaA, other: deltaB },
                    { strat: stratB, val: b, delta: deltaB, other: deltaA },
                  ].map(({ strat, val, delta, other }) => (
                    <div key={strat} className="space-y-0.5">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                          style={{ background: STRATEGY_COLORS[strat] }}
                        />
                        <span className="text-[10px]" style={{ color: "var(--muted)" }}>
                          {STRATEGY_LABELS[strat]}
                        </span>
                      </div>
                      <div className="text-sm font-semibold mono" style={{ color: "var(--foreground)" }}>
                        {val}
                      </div>
                      <div className="text-xs mono">
                        <Delta a={delta} b={other} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div
          className="rounded-lg border p-8 text-center text-sm"
          style={{ borderColor: "var(--border)", color: "var(--muted)" }}
        >
          Select two strategies to compare.
        </div>
      )}
    </div>
  );
}
