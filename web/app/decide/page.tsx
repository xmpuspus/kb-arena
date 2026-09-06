"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  DEFAULT_STRATEGY_CATALOG,
  STRATEGY_DESCRIPTIONS,
  fetchStrategyCatalog,
  type CorpusInfo,
  type Strategy,
} from "@/lib/api";
import {
  COST_CEILING_USD,
  EVIDENCE_UNREADABLE,
  LATENCY_CEILING_MS,
  PROFILE_NAMES,
  PROFILE_TRADEOFFS,
  PROFILE_WEIGHTS,
  benchmarkCommand,
  bundleCaveats,
  candidatesFor,
  catalogIsLive,
  compareCommand,
  decisionRecord,
  fetchCompare,
  fetchCorporaOrFail,
  fetchEvidenceBundles,
  ingestCommand,
  labCommand,
  type CompareResult,
  type DecideCatalogRecord,
  type EvidenceBundle,
  type ProfileName,
} from "@/lib/decide";

const WALKTHROUGH_URL =
  "https://github.com/xmpuspus/kb-arena/blob/main/docs/own-corpus-walkthrough.md";

const METRICS = ["accuracy", "latency_ms", "cost_usd"];

const STEPS = [
  { n: 1, tab: "Corpus", heading: "A decision holds only for the documents it ran on" },
  { n: 2, tab: "Objective", heading: "Each profile ranks the same strategies differently" },
  { n: 3, tab: "Candidates", heading: "The catalog says which strategies suit this objective" },
  { n: 4, tab: "Run", heading: "A number needs a run behind it" },
  { n: 5, tab: "Compare", heading: "Two means hide which strategy wins each question" },
  { n: 6, tab: "Record", heading: "The record repeats what the run recorded" },
];

const card = { background: "var(--card)", borderColor: "var(--border)" };
const inputStyle = {
  background: "var(--card)",
  borderColor: "var(--border)",
  color: "var(--foreground)",
};

function Command({ text }: { text: string }) {
  return (
    <code
      className="block text-xs mono break-all rounded border px-3 py-2"
      style={{
        borderColor: "var(--border)",
        background: "var(--background)",
        color: "var(--foreground)",
      }}
    >
      {text}
    </code>
  );
}

function Refusal({ text }: { text: string }) {
  return (
    <p
      className="rounded border px-3 py-2 text-sm"
      style={{ borderColor: "var(--danger)", background: "var(--danger-bg)", color: "var(--danger-text)" }}
    >
      {text}
    </p>
  );
}

function num(value: number | null | undefined, places: number): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(places) : "not recorded";
}

export default function DecidePage() {
  const [step, setStep] = useState(1);
  const [corpora, setCorpora] = useState<CorpusInfo[] | null>(null);
  const [corporaError, setCorporaError] = useState<string | null>(null);
  const [corpus, setCorpus] = useState("");
  const [ownDocs, setOwnDocs] = useState(false);
  const [catalog, setCatalog] = useState<DecideCatalogRecord[]>(DEFAULT_STRATEGY_CATALOG);
  const [profile, setProfile] = useState<ProfileName>("accuracy-first");
  const [picked, setPicked] = useState<Strategy[]>([]);
  const [stratA, setStratA] = useState("");
  const [stratB, setStratB] = useState("");
  const [metric, setMetric] = useState("accuracy");
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [comparing, setComparing] = useState(false);
  const [bundles, setBundles] = useState<EvidenceBundle[]>([]);
  const [bundlesError, setBundlesError] = useState<string | null>(null);
  const [createdAt, setCreatedAt] = useState("");

  // The step lives in the URL so a reader can link one screen of the flow, and
  // so a screenshot can reach step 6 without replaying five clicks.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const wanted = Number(params.get("step"));
    if (wanted >= 1 && wanted <= STEPS.length) setStep(wanted);
    const urlCorpus = params.get("corpus");
    if (urlCorpus) setCorpus(urlCorpus);
    const urlProfile = params.get("profile");
    if (PROFILE_NAMES.includes(urlProfile as ProfileName)) setProfile(urlProfile as ProfileName);
    const urlA = params.get("a");
    const urlB = params.get("b");
    if (urlA) setStratA(urlA);
    if (urlB) setStratB(urlB);
    const urlMetric = params.get("metric");
    if (urlMetric && METRICS.includes(urlMetric)) setMetric(urlMetric);
  }, []);

  useEffect(() => {
    fetchCorporaOrFail()
      .then((list) => {
        setCorpora(list);
        setCorporaError(null);
      })
      .catch((err: Error) => {
        setCorpora([]);
        setCorporaError(err.message);
      });
    fetchStrategyCatalog().then((records) => setCatalog(records as DecideCatalogRecord[]));
  }, []);

  const usable = useMemo(
    () => (corpora ?? []).filter((c) => (c.questionCount ?? 0) > 0),
    [corpora]
  );
  const skeletons = useMemo(
    () => (corpora ?? []).filter((c) => (c.questionCount ?? 0) === 0),
    [corpora]
  );

  useEffect(() => {
    if (!corpus && usable.length > 0) setCorpus(usable[0].value);
  }, [corpus, usable]);

  useEffect(() => {
    if (!corpus) return;
    fetchEvidenceBundles(corpus)
      .then((found) => {
        setBundles(found);
        setBundlesError(null);
      })
      .catch(() => {
        setBundles([]);
        setBundlesError(EVIDENCE_UNREADABLE);
      });
  }, [corpus]);

  const live = catalogIsLive(catalog);
  const candidates = useMemo(
    () => candidatesFor(profile, catalog, live),
    [profile, catalog, live]
  );

  useEffect(() => {
    setPicked(candidates.slice(0, 2).map((c) => c.name));
  }, [candidates]);

  useEffect(() => {
    if (!stratA && picked[0]) setStratA(picked[0]);
    if (!stratB && picked[1]) setStratB(picked[1]);
  }, [picked, stratA, stratB]);

  useEffect(() => {
    if (step !== 6 || createdAt) return;
    setCreatedAt(new Date().toISOString());
  }, [step, createdAt]);

  const goto = useCallback((next: number) => {
    setStep(next);
    const params = new URLSearchParams(window.location.search);
    params.set("step", String(next));
    window.history.replaceState(null, "", `?${params.toString()}`);
  }, []);

  const readComparison = useCallback(() => {
    if (!corpus || !stratA || !stratB) return;
    setComparing(true);
    fetchCompare(corpus, stratA, stratB, metric)
      .then((result) => {
        setComparison(result);
        setComparisonError(null);
      })
      .catch((err: Error) => {
        setComparison(null);
        setComparisonError(err.message);
      })
      .finally(() => setComparing(false));
  }, [corpus, stratA, stratB, metric]);

  // A link that names both strategies asks for the comparison, so the page
  // reads it once rather than waiting for a click the linker already made.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (step >= 5 && params.get("a") && params.get("b") && corpus && !comparison && !comparisonError) {
      readComparison();
    }
  }, [step, corpus, comparison, comparisonError, readComparison]);

  const bundle = bundles[0] ?? null;
  const corpusQuestions =
    corpora === null || corporaError
      ? null
      : (corpora.find((c) => c.value === corpus)?.questionCount ?? null);

  const record = useMemo(
    () =>
      decisionRecord({
        corpus: corpus || "none chosen",
        corpusQuestions,
        profile,
        candidates: picked,
        comparison,
        comparisonError,
        bundle,
        bundleError: bundlesError,
        metric,
        createdAt: createdAt || "not yet stamped",
      }),
    [corpus, corpusQuestions, profile, picked, comparison, comparisonError, bundle, bundlesError, metric, createdAt]
  );

  function download() {
    const blob = new Blob([record], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `decision-${corpus || "no-corpus"}-${profile}.md`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function togglePick(name: Strategy) {
    setPicked((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );
  }

  const active = STEPS[step - 1];

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">
          Six steps take you from a question to a retrieval decision you can defend
        </h1>
        <p className="text-sm max-w-3xl" style={{ color: "var(--muted)" }}>
          Each step reads what this deployment records. A step that cannot read its data says so,
          and the record at the end repeats the caveats the run wrote.
        </p>
      </header>

      <ol className="flex flex-wrap gap-2" aria-label="Decision steps">
        {STEPS.map((s) => (
          <li key={s.n}>
            <button
              type="button"
              onClick={() => goto(s.n)}
              aria-current={s.n === step ? "step" : undefined}
              className="rounded-lg border px-3 py-1.5 text-sm font-medium"
              style={{
                borderColor: s.n === step ? "var(--accent)" : "var(--border)",
                background: s.n === step ? "var(--accent)" : "var(--card)",
                color: s.n === step ? "#fff" : "var(--foreground)",
              }}
            >
              {s.n}. {s.tab}
            </button>
          </li>
        ))}
      </ol>

      <section className="rounded-lg border p-5 space-y-4" style={card}>
        <h2 className="text-lg font-semibold">{active.heading}</h2>

        {step === 1 && (
          <div className="space-y-4">
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              A retrieval result transfers to your documents only when the run used them. Pick the
              built-in example to read a recorded run, or bring your own documents.
            </p>

            <div className="flex flex-wrap gap-3">
              {[
                { own: false, label: "The built-in example" },
                { own: true, label: "Your own documents" },
              ].map((choice) => (
                <button
                  key={String(choice.own)}
                  type="button"
                  onClick={() => setOwnDocs(choice.own)}
                  className="rounded-lg border px-4 py-2 text-sm font-medium"
                  style={{
                    borderColor: ownDocs === choice.own ? "var(--accent)" : "var(--border)",
                    color: ownDocs === choice.own ? "var(--accent)" : "var(--foreground)",
                    background: "var(--card)",
                  }}
                >
                  {choice.label}
                </button>
              ))}
            </div>

            {corporaError && <Refusal text={corporaError} />}

            {!ownDocs && !corporaError && (
              <div className="space-y-3">
                {corpora === null && <p className="text-sm">Reading the corpus list.</p>}
                {corpora !== null && usable.length === 0 && (
                  <p className="text-sm" style={{ color: "var(--muted)" }}>
                    This deployment holds no corpus with questions, so no step below can rank
                    anything.
                  </p>
                )}
                <div className="flex flex-wrap gap-2">
                  {usable.map((c) => (
                    <button
                      key={c.value}
                      type="button"
                      onClick={() => setCorpus(c.value)}
                      className="rounded-lg border px-3 py-2 text-sm text-left"
                      style={{
                        borderColor: corpus === c.value ? "var(--accent)" : "var(--border)",
                        background: "var(--card)",
                      }}
                    >
                      <span className="font-medium">{c.label}</span>
                      <span className="block text-xs" style={{ color: "var(--muted)" }}>
                        {c.questionCount} questions
                        {c.hasResults ? ", results on disk" : ", no results yet"}
                      </span>
                    </button>
                  ))}
                </div>
                {skeletons.length > 0 && (
                  <p className="text-xs" style={{ color: "var(--muted)" }}>
                    {skeletons.map((c) => c.value).join(", ")} hold no questions. They are skeletons
                    for a walkthrough, so they cannot support a decision.
                  </p>
                )}
              </div>
            )}

            {ownDocs && (
              <div className="space-y-3">
                <p className="text-sm">
                  Scaffold a corpus, drop your files into its raw folder, then ingest them. The
                  walkthrough runs every command against a real corpus and prints what each one
                  answered.
                </p>
                <Command text={`kb-arena init-corpus ${corpus || "my-docs"}`} />
                <Command text={ingestCommand(corpus || "my-docs")} />
                <a
                  href={WALKTHROUGH_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex text-sm font-medium underline"
                  style={{ color: "var(--accent)" }}
                >
                  Read docs/own-corpus-walkthrough.md
                </a>
                <p className="text-xs" style={{ color: "var(--muted)" }}>
                  Your own corpus carries no questions until you write them. Until it does, the
                  steps below cannot score anything on it.
                </p>
              </div>
            )}
          </div>
        )}

        {step === 2 && (
          <div className="space-y-4">
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              A profile is a set of weights the report ranks with. The weight that reads 0.0 names
              what the profile gives up.
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              {PROFILE_NAMES.map((name) => {
                const w = PROFILE_WEIGHTS[name];
                return (
                  <button
                    key={name}
                    type="button"
                    onClick={() => setProfile(name)}
                    className="rounded-lg border p-4 text-left space-y-2"
                    style={{
                      borderColor: profile === name ? "var(--accent)" : "var(--border)",
                      background: "var(--card)",
                    }}
                  >
                    <span className="block text-sm font-semibold mono">{name}</span>
                    <span className="block text-xs mono" style={{ color: "var(--muted)" }}>
                      accuracy {w.accuracy} | reliability {w.reliability} | latency {w.latency} |
                      cost {w.cost}
                    </span>
                    <span className="block text-xs leading-relaxed">{PROFILE_TRADEOFFS[name]}</span>
                  </button>
                );
              })}
            </div>
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              The report scores latency against a ceiling of {LATENCY_CEILING_MS} ms and cost
              against ${COST_CEILING_USD} per query. Both ceilings are choices you can argue with.
            </p>
          </div>
        )}

        {step === 3 && (
          <div className="space-y-4">
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              These are the catalog entries the default benchmark runs, ordered for{" "}
              <span className="mono">{profile}</span>. The catalog records no latency and no cost
              per query, so this order claims neither. It uses the one spending fact the catalog
              holds: whether a strategy calls the embedding provider per query.
            </p>
            {!live && (
              <Refusal text="The deployment catalog could not be read, so this list comes from the bundled copy. Runtime availability and architecture are unknown." />
            )}
            <ul className="space-y-3">
              {candidates.map((c) => (
                <li key={c.name} className="rounded-lg border p-4 space-y-2" style={card}>
                  <label className="flex items-start gap-2">
                    <input
                      type="checkbox"
                      checked={picked.includes(c.name)}
                      onChange={() => togglePick(c.name)}
                      className="mt-1"
                    />
                    <span>
                      <span className="text-sm font-semibold">{c.label}</span>
                      <span className="block text-xs mono" style={{ color: "var(--muted)" }}>
                        {c.name}
                      </span>
                    </span>
                  </label>
                  <p className="text-xs leading-relaxed">{STRATEGY_DESCRIPTIONS[c.name]}</p>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div>
                      <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
                        Why it is a candidate
                      </span>
                      <ul className="mt-1 space-y-1 text-xs list-disc pl-4">
                        {c.reasons.map((r) => (
                          <li key={r}>{r}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
                        What a run costs
                      </span>
                      <ul className="mt-1 space-y-1 text-xs list-disc pl-4">
                        {c.runCost.map((r) => (
                          <li key={r}>{r}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              {catalog.length - candidates.length} of {catalog.length} catalog entries stay out of
              this list. Each one needs an optional extra, or costs extra model calls per question,
              or is marked experimental.
            </p>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-4">
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              Run the candidates yourself, or read a run this deployment already recorded. Both
              paths end at the same comparison.
            </p>
            <div className="space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
                Run it, with judged answers
              </span>
              <Command text={benchmarkCommand(corpus || "my-docs", picked)} />
              <span className="text-xs font-semibold uppercase tracking-wide block pt-2" style={{ color: "var(--muted)" }}>
                Run it cheaper, retrieval metrics only
              </span>
              <Command text={labCommand(corpus || "my-docs", picked)} />
            </div>

            <div className="space-y-2 border-t pt-4" style={{ borderColor: "var(--border)" }}>
              <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
                Recorded runs for {corpus || "no corpus"}
              </span>
              {bundlesError && <Refusal text={bundlesError} />}
              {!bundlesError && bundles.length === 0 && (
                <p className="text-sm" style={{ color: "var(--muted)" }}>
                  This deployment holds no evidence bundle for {corpus || "this corpus"}, so there
                  is no recorded run to inspect. Run one of the commands above.
                </p>
              )}
              {bundles.map((b) => (
                <div key={b.run_id} className="rounded-lg border p-4 space-y-2" style={card}>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold mono">run {b.run_id}</span>
                    <span
                      className="text-xs font-medium rounded px-2 py-0.5"
                      style={{
                        background: b.citable ? "var(--success-bg)" : "var(--danger-bg)",
                        color: b.citable ? "var(--success-text)" : "var(--danger-text)",
                      }}
                    >
                      {b.citable ? "citable evidence" : "development signal"}
                    </span>
                  </div>
                  <Command text={(b.command ?? []).join(" ") || "no command recorded"} />
                  <dl className="grid gap-1 text-xs sm:grid-cols-2">
                    <div>
                      <dt className="inline font-semibold">Commit: </dt>
                      <dd className="inline mono">{b.environment?.git_sha ?? "none recorded"}</dd>
                    </div>
                    <div>
                      <dt className="inline font-semibold">Seed: </dt>
                      <dd className="inline mono">{b.seed ?? "none recorded"}</dd>
                    </div>
                    <div>
                      <dt className="inline font-semibold">Package: </dt>
                      <dd className="inline mono">{b.environment?.kb_arena ?? "not recorded"}</dd>
                    </div>
                    <div>
                      <dt className="inline font-semibold">Written: </dt>
                      <dd className="inline mono">{b.created_at ?? "not recorded"}</dd>
                    </div>
                  </dl>
                  <ul className="space-y-1 text-xs list-disc pl-4">
                    {bundleCaveats(b).map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        )}

        {step === 5 && (
          <div className="space-y-4">
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              Two means say which strategy scored higher on average. A pairing says which one wins
              each question, and how often. This reads the same route the command line reads.
            </p>
            <div className="grid gap-3 sm:grid-cols-4">
              {[
                { label: "Baseline A", value: stratA, set: setStratA },
                { label: "Candidate B", value: stratB, set: setStratB },
              ].map((side) => (
                <div key={side.label} className="space-y-1">
                  <label className="text-xs font-medium block" style={{ color: "var(--muted)" }}>
                    {side.label}
                  </label>
                  <select
                    value={side.value}
                    onChange={(e) => side.set(e.target.value)}
                    className="w-full px-3 py-1.5 rounded-lg border text-sm"
                    style={inputStyle}
                  >
                    <option value="">Pick one</option>
                    {catalog.map((r) => (
                      <option key={r.name} value={r.name}>
                        {r.label}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
              <div className="space-y-1">
                <label className="text-xs font-medium block" style={{ color: "var(--muted)" }}>
                  Metric
                </label>
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                  className="w-full px-3 py-1.5 rounded-lg border text-sm"
                  style={inputStyle}
                >
                  {METRICS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-end">
                <button
                  type="button"
                  onClick={readComparison}
                  disabled={comparing || !stratA || !stratB}
                  className="w-full rounded-lg px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
                  style={{ background: "var(--accent)" }}
                >
                  {comparing ? "Reading" : "Read the comparison"}
                </button>
              </div>
            </div>

            <Command text={compareCommand(corpus || "my-docs", stratA || "a", stratB || "b", metric)} />

            {comparisonError && <Refusal text={comparisonError} />}

            {comparison && (
              <div className="space-y-3">
                <div className="grid gap-3 sm:grid-cols-4">
                  {[
                    { label: `Mean ${comparison.a}`, value: num(comparison.mean_a, 4) },
                    { label: `Mean ${comparison.b}`, value: num(comparison.mean_b, 4) },
                    { label: "Mean delta", value: num(comparison.mean_delta, 4) },
                    {
                      label: "Wins, ties, losses",
                      value: `${comparison.wins}, ${comparison.ties}, ${comparison.losses}`,
                    },
                  ].map((box) => (
                    <div key={box.label} className="rounded-lg border p-3" style={card}>
                      <div className="text-xs" style={{ color: "var(--muted)" }}>
                        {box.label}
                      </div>
                      <div className="text-sm font-semibold mono">{box.value}</div>
                    </div>
                  ))}
                </div>
                <p className="text-xs">
                  {comparison.n_paired} paired questions. 95% CI on the mean delta [
                  {num(comparison.delta_ci_95?.[0], 4)}, {num(comparison.delta_ci_95?.[1], 4)}].
                  Wilcoxon p {comparison.wilcoxon_p === null ? "not computed" : num(comparison.wilcoxon_p, 6)}.
                </p>
                <p className="text-xs">
                  {!comparison.enough_pairs_for_inference
                    ? "The pairing sits below the floor for inference, so no significance flag fired."
                    : comparison.significant
                      ? `The comparison flags this difference as significant for ${comparison.b}.`
                      : "The comparison did not flag this difference as significant."}
                </p>
                {comparison.meta.reasons.length > 0 && (
                  <ul className="space-y-1 text-xs list-disc pl-4" style={{ color: "var(--danger-text)" }}>
                    {comparison.meta.reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                )}
                <div className="overflow-x-auto border rounded" style={{ borderColor: "var(--border)" }}>
                  <table className="min-w-full text-xs">
                    <thead style={{ background: "var(--subtle)" }}>
                      <tr>
                        <th className="px-3 py-2 text-left font-medium">Question</th>
                        <th className="px-3 py-2 text-right font-medium">{comparison.a}</th>
                        <th className="px-3 py-2 text-right font-medium">{comparison.b}</th>
                        <th className="px-3 py-2 text-right font-medium">Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.per_question.map((row) => (
                        <tr key={row.question_id} className="border-t" style={{ borderColor: "var(--border)" }}>
                          <td className="px-3 py-1.5 mono">{row.question_id}</td>
                          <td className="px-3 py-1.5 text-right mono">{num(row.a, 3)}</td>
                          <td className="px-3 py-1.5 text-right mono">{num(row.b, 3)}</td>
                          <td className="px-3 py-1.5 text-right mono">{num(row.delta, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-xs" style={{ color: "var(--muted)" }}>
                  {comparison.note}
                </p>
              </div>
            )}
          </div>
        )}

        {step === 6 && (
          <div className="space-y-4">
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              The record below names the corpus, the objective, the strategies, the numbers and the
              caveats. It copies the review verdict the run wrote, so it cannot claim more than the
              evidence bundle allows.
            </p>
            <button
              type="button"
              onClick={download}
              className="rounded-lg px-4 py-2 text-sm font-medium text-white"
              style={{ background: "var(--accent)" }}
            >
              Download the decision record
            </button>
            <pre
              className="text-xs mono whitespace-pre-wrap rounded border p-4 overflow-x-auto"
              style={{ borderColor: "var(--border)", background: "var(--background)" }}
            >
              {record}
            </pre>
          </div>
        )}
      </section>

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => goto(Math.max(1, step - 1))}
          disabled={step === 1}
          className="rounded-lg border px-4 py-2 text-sm font-medium disabled:opacity-40"
          style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
        >
          Back
        </button>
        <button
          type="button"
          onClick={() => goto(Math.min(STEPS.length, step + 1))}
          disabled={step === STEPS.length}
          className="rounded-lg px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          style={{ background: "var(--accent)" }}
        >
          Next step
        </button>
      </div>
    </div>
  );
}
