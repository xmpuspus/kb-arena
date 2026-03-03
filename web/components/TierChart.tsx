"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { STRATEGY_LABELS, STRATEGY_COLORS, TIER_INFO, type Strategy } from "@/lib/api";

interface Row {
  strategy: Strategy;
  tiers: number[];
}

interface Props {
  rows: Row[];
}

export default function TierChart({ rows }: Props) {
  const tierCount = rows[0]?.tiers.length ?? 5;

  const data = Array.from({ length: tierCount }, (_, i) => {
    const label = TIER_INFO[i + 1]?.label ?? `Tier ${i + 1}`;
    const point: Record<string, string | number> = { tier: `T${i + 1} ${label}` };
    for (const row of rows) {
      point[row.strategy] = row.tiers[i] ?? 0;
    }
    return point;
  });

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="tier" tick={{ fill: "var(--muted)", fontSize: 12 }} />
        <YAxis
          domain={[0, 100]}
          tickFormatter={(v) => `${v}%`}
          tick={{ fill: "var(--muted)", fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            color: "var(--foreground)",
          }}
          formatter={(val, key) => [
            `${val ?? 0}%`,
            STRATEGY_LABELS[key as Strategy] ?? key,
          ]}
        />
        <Legend
          content={() => (
            <div style={{ display: "flex", justifyContent: "center", gap: 16, color: "var(--muted)", fontSize: 12 }}>
              {rows.map((row) => (
                <span key={row.strategy} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: STRATEGY_COLORS[row.strategy] }} />
                  {STRATEGY_LABELS[row.strategy] ?? row.strategy}
                </span>
              ))}
            </div>
          )}
        />
        {rows.map((row) => (
          <Bar
            key={row.strategy}
            dataKey={row.strategy}
            fill={STRATEGY_COLORS[row.strategy]}
            radius={[2, 2, 0, 0]}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
