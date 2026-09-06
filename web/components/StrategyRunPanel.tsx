"use client";

import { useEffect, useMemo, useState } from "react";
import {
  DEFAULT_STRATEGY_CATALOG,
  STRATEGY_GROUP_LABELS,
  STRATEGY_GROUP_ORDER,
  fetchStrategyCatalog,
  strategyGroup,
  type CorpusInfo,
  type Strategy,
  type StrategyCatalogRecord,
} from "@/lib/api";

interface Props {
  corpora: CorpusInfo[];
}

const SPLIT_OPTIONS = [
  { value: "", label: "Any split" },
  { value: "development", label: "Development" },
  { value: "validation", label: "Validation" },
  { value: "holdout", label: "Holdout" },
  { value: "all", label: "All splits" },
];

const selectStyle = {
  background: "var(--card)",
  borderColor: "var(--border-strong)",
  color: "var(--foreground)",
};

function defaultSelection(catalog: StrategyCatalogRecord[]): Set<Strategy> {
  return new Set(catalog.filter((r) => r.default_benchmark).map((r) => r.name));
}

import { isSafeId, UNSAFE_COMMAND } from "@/lib/decide";

const NOTHING_PICKED = "Pick at least one strategy. An empty pick is not a run.";
// `kb_arena/strategies/base.py` refuses a top-k outside this range, so a
// command carrying a larger one is a command that cannot run.
const MAX_RETRIEVAL_CANDIDATES = 1000;
const OUT_OF_RANGE =
  `Top-k and ceiling-k run from 1 to ${MAX_RETRIEVAL_CANDIDATES}. Retrieval refuses anything outside that, so this page will not build a command from it.`;
const NO_CORPUS =
  "This deployment reported no corpus, so there is no name to put in a command.";

export default function StrategyRunPanel({ corpora }: Props) {
  const [catalog, setCatalog] = useState<StrategyCatalogRecord[]>(DEFAULT_STRATEGY_CATALOG);
  const [corpus, setCorpus] = useState(corpora[0]?.value ?? "aws-compute");
  const [selected, setSelected] = useState<Set<Strategy>>(() =>
    defaultSelection(DEFAULT_STRATEGY_CATALOG)
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [topK, setTopK] = useState(5);
  const [tier, setTier] = useState(0);
  const [split, setSplit] = useState("");
  const [seed, setSeed] = useState("");
  const [ceilingK, setCeilingK] = useState("");
  const [referenceFree, setReferenceFree] = useState(false);
  const [copied, setCopied] = useState<"benchmark" | "lab" | null>(null);

  useEffect(() => {
    fetchStrategyCatalog().then(setCatalog);
  }, []);

  // The corpus list arrives after mount. Keep the current pick if it is
  // still valid, otherwise fall back to the first real corpus.
  useEffect(() => {
    if (corpora.length > 0 && !corpora.some((c) => c.value === corpus)) {
      setCorpus(corpora[0].value);
    }
    // Runs only when the corpus list changes, not on every keystroke below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [corpora]);

  const groups = useMemo(() => {
    const byGroup: Record<string, StrategyCatalogRecord[]> = {
      baseline: [],
      advanced: [],
      experimental: [],
    };
    for (const record of catalog) byGroup[strategyGroup(record)].push(record);
    return byGroup;
  }, [catalog]);

  const selectedNames = useMemo(
    () => catalog.filter((r) => selected.has(r.name)).map((r) => r.name),
    [catalog, selected]
  );

  function toggle(name: Strategy) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  // An empty pick used to read "all", so a reader who unchecked every box got a
  // command that runs the nine default strategies while the page said 0 of 19.
  // The panel refuses instead, because the command has to match what it shows.
  const nothingPicked = selectedNames.length === 0;
  const ceilingValue = ceilingK.trim() ? Number(ceilingK.trim()) : null;
  const kInRange = (value: number | null) =>
    value === null ||
    (Number.isInteger(value) && value >= 1 && value <= MAX_RETRIEVAL_CANDIDATES);
  const kValuesFit = kInRange(topK) && kInRange(ceilingValue);
  const strategyArg = selectedNames.join(",");

  const benchmarkCommand = useMemo(() => {
    if (corpora.length === 0) return NO_CORPUS;
    if (!strategyArg) return NOTHING_PICKED;
    if (!kValuesFit) return OUT_OF_RANGE;
    // The corpus and every strategy name reach a line a reader pastes into a
    // terminal, so this panel refuses what the API refuses. decide.ts holds the
    // same check, and compare.py holds the pattern both read from.
    if (![corpus, ...selectedNames].every(isSafeId)) return UNSAFE_COMMAND;
    const parts = ["kb-arena", "benchmark", "--corpus", corpus, "--strategy", strategyArg];
    if (topK !== 5) parts.push("--top-k", String(topK));
    if (tier !== 0) parts.push("--tier", String(tier));
    if (split) parts.push("--split", split);
    if (seed.trim()) parts.push("--seed", seed.trim());
    if (referenceFree) parts.push("--reference-free");
    return parts.join(" ");
  }, [corpora, corpus, kValuesFit, selectedNames, strategyArg, topK, tier, split, seed, referenceFree]);

  const retrieverLabCommand = useMemo(() => {
    if (corpora.length === 0) return NO_CORPUS;
    if (!strategyArg) return NOTHING_PICKED;
    if (!kValuesFit) return OUT_OF_RANGE;
    if (![corpus, ...selectedNames].every(isSafeId)) return UNSAFE_COMMAND;
    const parts = ["kb-arena", "retriever-lab", "--corpus", corpus, "--strategies", strategyArg];
    if (topK !== 5) parts.push("--top-k", String(topK));
    if (split) parts.push("--split", split);
    if (ceilingK.trim()) parts.push("--ceiling-k", ceilingK.trim());
    return parts.join(" ");
  }, [corpora, corpus, kValuesFit, selectedNames, strategyArg, topK, split, ceilingK]);

  async function copyCommand(command: string, which: "benchmark" | "lab") {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(which);
      setTimeout(() => setCopied((prev) => (prev === which ? null : prev)), 2000);
    } catch {
      // Clipboard access can be refused. The command stays selectable text.
    }
  }

  return (
    <section
      className="rounded-lg border p-4 space-y-4"
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
    >
      <div>
        <h2 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
          Plan a run
        </h2>
        <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
          Pick a corpus and strategies, then copy the command to run in your terminal.
        </p>
      </div>

      <div className="flex items-center gap-2">
        <label htmlFor="run-corpus" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
          Corpus
        </label>
        <select
          id="run-corpus"
          value={corpus}
          onChange={(e) => setCorpus(e.target.value)}
          className="px-3 py-1.5 rounded-lg border text-sm"
          style={selectStyle}
        >
          {corpora.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-3">
        {STRATEGY_GROUP_ORDER.map((groupKey) => (
          <fieldset
            key={groupKey}
            className="rounded-lg border p-3"
            style={{ borderColor: "var(--border)" }}
          >
            <legend className="text-xs font-semibold px-1" style={{ color: "var(--foreground)" }}>
              {STRATEGY_GROUP_LABELS[groupKey]}
            </legend>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1.5">
              {groups[groupKey].map((record) => {
                const id = `strategy-${record.name}`;
                return (
                  <div key={record.name} className="flex items-start gap-2 py-0.5">
                    <input
                      type="checkbox"
                      id={id}
                      checked={selected.has(record.name)}
                      onChange={() => toggle(record.name)}
                      className="mt-0.5"
                    />
                    <label htmlFor={id} className="text-sm leading-snug" style={{ color: "var(--foreground)" }}>
                      {record.label}
                      {record.default_benchmark && (
                        <span
                          className="ml-1.5 text-[10px] font-medium uppercase tracking-wide"
                          style={{ color: "var(--accent)" }}
                        >
                          Recommended
                        </span>
                      )}
                    </label>
                  </div>
                );
              })}
            </div>
          </fieldset>
        ))}
      </div>

      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={nothingPicked || corpora.length === 0}
            onClick={() => copyCommand(benchmarkCommand, "benchmark")}
            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80 disabled:opacity-50"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            {copied === "benchmark" ? "Command copied" : "Copy run command"}
          </button>
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            {selectedNames.length} of {catalog.length} strategies selected
          </span>
        </div>
        <code
          className="block text-xs mono break-all rounded border px-3 py-2"
          style={{ borderColor: "var(--border)", background: "var(--background)", color: "var(--foreground)" }}
        >
          {benchmarkCommand}
        </code>
      </div>

      <div>
        <button
          type="button"
          aria-expanded={advancedOpen}
          aria-controls="advanced-run-controls"
          onClick={() => setAdvancedOpen((v) => !v)}
          className="text-xs font-medium underline"
          style={{ color: "var(--accent)" }}
        >
          {advancedOpen ? "Hide advanced options" : "Show advanced options"}
        </button>

        <div
          id="advanced-run-controls"
          hidden={!advancedOpen}
          className="mt-3 space-y-4 border-t pt-3"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="space-y-1">
              <label htmlFor="run-top-k" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                Top-k
              </label>
              <input
                id="run-top-k"
                type="number"
                min={1}
                max={MAX_RETRIEVAL_CANDIDATES}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value) || 5)}
                className="w-full px-3 py-1.5 rounded-lg border text-sm"
                style={selectStyle}
              />
            </div>

            <div className="space-y-1">
              <label htmlFor="run-tier" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                Tier
              </label>
              <select
                id="run-tier"
                value={tier}
                onChange={(e) => setTier(Number(e.target.value))}
                className="w-full px-3 py-1.5 rounded-lg border text-sm"
                style={selectStyle}
              >
                <option value={0}>All tiers</option>
                {[1, 2, 3, 4, 5].map((t) => (
                  <option key={t} value={t}>
                    Tier {t}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label htmlFor="run-split" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                Split
              </label>
              <select
                id="run-split"
                value={split}
                onChange={(e) => setSplit(e.target.value)}
                className="w-full px-3 py-1.5 rounded-lg border text-sm"
                style={selectStyle}
              >
                {SPLIT_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label htmlFor="run-seed" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                Seed
              </label>
              <input
                id="run-seed"
                type="number"
                placeholder="unset"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                className="w-full px-3 py-1.5 rounded-lg border text-sm"
                style={selectStyle}
              />
            </div>

            <div className="space-y-1">
              <label htmlFor="run-ceiling-k" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                Ceiling-k
              </label>
              <input
                id="run-ceiling-k"
                type="number"
                min={1}
                max={MAX_RETRIEVAL_CANDIDATES}
                placeholder="top-k x 4"
                value={ceilingK}
                onChange={(e) => setCeilingK(e.target.value)}
                className="w-full px-3 py-1.5 rounded-lg border text-sm"
                style={selectStyle}
              />
            </div>

            <div className="space-y-1">
              <span className="text-xs font-medium block" style={{ color: "var(--muted)" }}>
                Judging
              </span>
              <div className="flex items-center gap-2 h-[34px]">
                <input
                  id="run-reference-free"
                  type="checkbox"
                  checked={referenceFree}
                  onChange={(e) => setReferenceFree(e.target.checked)}
                />
                <label htmlFor="run-reference-free" className="text-sm" style={{ color: "var(--foreground)" }}>
                  Reference-free
                </label>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                kb-arena retriever-lab (cheaper, IR metrics only, uses ceiling-k)
              </span>
              <button
                type="button"
                disabled={nothingPicked || corpora.length === 0}
                onClick={() => copyCommand(retrieverLabCommand, "lab")}
                className="text-xs px-2 py-1 rounded border disabled:opacity-50"
                style={{ borderColor: "var(--border-strong)", color: "var(--foreground)" }}
              >
                {copied === "lab" ? "Copied" : "Copy command"}
              </button>
            </div>
            <code
              className="block text-xs mono break-all rounded border px-3 py-2"
              style={{ borderColor: "var(--border)", background: "var(--background)", color: "var(--foreground)" }}
            >
              {retrieverLabCommand}
            </code>
          </div>
        </div>
      </div>
    </section>
  );
}
