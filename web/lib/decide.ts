// The guided flow reads facts the backend already records. Nothing here
// invents a number, and nothing here recomputes a verdict the run wrote.
//
// The profile weights below mirror `PROFILES` in `kb_arena/benchmark/reporter.py`.
// They are copied because a static export cannot read Python, and
// `tests/test_decide_parity.py` fails when the two lists disagree.

import { apiFetch } from "./auth";
import { API_URL, type CorpusInfo, type StrategyCatalogRecord, type Strategy } from "./api";

export const PROFILE_NAMES = [
  "accuracy-first",
  "balanced",
  "latency-bound",
  "cost-bound",
] as const;

export type ProfileName = (typeof PROFILE_NAMES)[number];

export interface ProfileWeights {
  accuracy: number;
  reliability: number;
  latency: number;
  cost: number;
}

export const PROFILE_WEIGHTS: Record<ProfileName, ProfileWeights> = {
  "accuracy-first": { accuracy: 0.7, reliability: 0.2, latency: 0.1, cost: 0.0 },
  balanced: { accuracy: 0.5, reliability: 0.3, latency: 0.2, cost: 0.0 },
  "latency-bound": { accuracy: 0.4, reliability: 0.2, latency: 0.4, cost: 0.0 },
  "cost-bound": { accuracy: 0.4, reliability: 0.2, latency: 0.0, cost: 0.4 },
};

// The reporter scores latency and cost against these ceilings. A reader who
// argues with a ranking argues with these two numbers, so the page prints them.
export const LATENCY_CEILING_MS = 10000;
export const COST_CEILING_USD = 0.1;

// What each profile gives up. The weight that reads 0.0 is the trade-off.
export const PROFILE_TRADEOFFS: Record<ProfileName, string> = {
  "accuracy-first":
    "Cost carries weight 0.0, so this profile ranks an expensive strategy above a cheap one whenever it answers better.",
  balanced:
    "Cost carries weight 0.0 and accuracy drops to 0.5, so reliability decides more ties here than under accuracy-first.",
  "latency-bound":
    "Latency rises to 0.4 and matches accuracy, so a faster strategy outranks a more accurate one once the gap in accuracy is small.",
  "cost-bound":
    "Latency carries weight 0.0, so this profile accepts a slow strategy to save money per query.",
};

export interface EvidenceReview {
  publishable?: boolean;
  note?: string;
  questions?: number;
  reviewed_share?: number | null;
  counts?: Record<string, number>;
}

export interface EvidenceBundle {
  run_id?: string;
  bundle_version?: number;
  corpus?: string;
  command?: string[];
  results?: string[];
  seed?: number;
  created_at?: string;
  citable?: boolean;
  why_not_citable?: string;
  question_set_fingerprint?: string;
  review_question_set?: string;
  review_split?: string;
  review?: EvidenceReview;
  environment?: {
    kb_arena?: string;
    git_sha?: string | null;
    platform?: string;
    python?: { version?: string; implementation?: string };
  };
}

export interface CompareMetaSide {
  file?: string;
  run_dir?: string;
  corpus?: string;
  strategy?: string;
  run_id?: string | null;
  compatibility_key?: string | null;
  error_records?: number;
  duplicate_records?: number;
}

export interface CompareResult {
  metric: string;
  lower_is_better: boolean;
  a: string;
  b: string;
  n_paired: number;
  unpaired_a: number;
  unpaired_b: number;
  mean_a: number;
  mean_b: number;
  mean_delta: number;
  delta_ci_95: [number, number];
  ci_excludes_zero: boolean;
  effect_size_d: number | null;
  wilcoxon_p: number | null;
  significant: boolean;
  enough_pairs_for_inference: boolean;
  wins: number;
  ties: number;
  losses: number;
  per_question: { question_id: string; a: number; b: number; delta: number }[];
  note: string;
  meta: { a: CompareMetaSide; b: CompareMetaSide; comparable: boolean; reasons: string[] };
}

// A refusal to read is not an empty result set. Each of these says which one
// happened, so no step shows sample content under a real corpus name.
export const EVIDENCE_UNREADABLE =
  "The evidence bundles could not be read just now, so the review status behind any number here is unknown.";
export const COMPARE_UNAUTHORIZED =
  "The comparison needs an API token. Enter one in the top bar, then read the comparison again.";
export const COMPARE_MISSING =
  "This deployment holds no result file for one of these two strategies, so there is nothing to pair.";
export const COMPARE_UNREADABLE =
  "The comparison could not be read just now. Try again in a moment.";
export const CORPORA_UNREADABLE =
  "The corpus list could not be read, so this page names no corpus. It will not show an example under a real name.";

// `fetchCorpora` in `lib/api.ts` answers with the built-in name when the read
// fails, which reads as a corpus this deployment holds. Step 1 must tell a
// failed read from an empty deployment, so it raises instead.
export async function fetchCorporaOrFail(): Promise<CorpusInfo[]> {
  const res = await fetch(`${API_URL}/api/corpora`);
  if (!res.ok) throw new Error(CORPORA_UNREADABLE);
  const data = await readJson(res, CORPORA_UNREADABLE);
  // A body this page cannot parse is a failed read. Falling back to an empty
  // array turned it into "this deployment holds no corpus", which is a claim
  // about the deployment that nothing supports.
  // The array alone was the check, so `corpora: [null]` reached the page and
  // crashed on `c.questionCount`. Every entry needs the name the page reads and
  // a number where it reads a number.
  const shaped =
    Array.isArray(data.corpora) &&
    data.corpora.every((entry) => {
      if (entry === null || typeof entry !== "object") return false;
      const fields = entry as Record<string, unknown>;
      if (typeof fields.value !== "string") return false;
      return ["questionCount", "reviewedQuestionCount", "draftQuestionCount"].every(
        (key) => fields[key] === undefined || typeof fields[key] === "number"
      );
    });
  if (!shaped) throw new Error(CORPORA_UNREADABLE);
  return data.corpora as CorpusInfo[];
}

export interface EvidenceAnswer {
  bundles: EvidenceBundle[];
  // Runs whose bundle sits on disk and could not be parsed. A broken bundle is
  // not a missing one, so the page never reads these as an empty deployment.
  unreadable: string[];
  // The route reads a fixed number of run directories. A capped list and a
  // short list are two different answers, so the page repeats which it got.
  truncated: boolean;
  scanLimit: number;
}

// A 200 with a body this page cannot read is a failed read, not an empty
// result. Every reader below raises rather than answering with a domain value
// nothing supports.
async function readJson(res: Response, failure: string): Promise<Record<string, unknown>> {
  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new Error(failure);
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error(failure);
  }
  return data as Record<string, unknown>;
}

export async function fetchEvidenceBundles(corpus: string): Promise<EvidenceAnswer> {
  const query = corpus ? `?corpus=${encodeURIComponent(corpus)}` : "";
  const res = await fetch(`${API_URL}/api/evidence${query}`);
  if (!res.ok) throw new Error(EVIDENCE_UNREADABLE);
  const data = await readJson(res, EVIDENCE_UNREADABLE);
  // Two arrays were the whole check, so a 200 carrying `bundles: [null]` passed
  // and then crashed the step-4 table. Every field on EvidenceBundle is
  // optional, so the requirement is that each element is an object at all, and
  // that each unreadable entry is a name the page can print.
  const bundlesAreShaped =
    Array.isArray(data.bundles) &&
    data.bundles.every((bundle) => {
      if (bundle === null || typeof bundle !== "object") return false;
      const fields = bundle as Record<string, unknown>;
      // Every field is optional, so absent is fine. Present with the wrong type
      // is not: the table calls `command.join(" ")`, which a string answers with
      // a TypeError rather than a value.
      const listsAreLists = ["command", "results"].every(
        (key) => fields[key] === undefined || Array.isArray(fields[key])
      );
      const textIsText = [
        "run_id",
        "corpus",
        "created_at",
        "why_not_citable",
        "question_set_fingerprint",
        "review_question_set",
        "review_split",
      ].every((key) => fields[key] === undefined || typeof fields[key] === "string");
      // `citable` is the one field the whole flow rests on, and the string
      // "false" is truthy. So a corrupt bundle read as citable evidence.
      const flagsAreFlags = ["citable"].every(
        (key) => fields[key] === undefined || typeof fields[key] === "boolean"
      );
      const numbersAreNumbers = ["bundle_version", "seed"].every(
        (key) => fields[key] === undefined || typeof fields[key] === "number"
      );
      const nestedAreObjects = ["review", "environment"].every(
        (key) => fields[key] === undefined || (fields[key] !== null && typeof fields[key] === "object")
      );
      return (
        listsAreLists && textIsText && flagsAreFlags && numbersAreNumbers && nestedAreObjects
      );
    });
  const unreadableIsShaped =
    Array.isArray(data.unreadable) && data.unreadable.every((name) => typeof name === "string");
  if (!bundlesAreShaped || !unreadableIsShaped) {
    throw new Error(EVIDENCE_UNREADABLE);
  }
  return {
    bundles: data.bundles as EvidenceBundle[],
    unreadable: data.unreadable as string[],
    truncated: Boolean(data.truncated),
    scanLimit: typeof data.scan_limit === "number" ? data.scan_limit : 0,
  };
}

/**
 * What an empty bundle list proves, which is less than "no run exists".
 *
 * The list is empty after a capped scan and after a parse failure as well as
 * after a real absence. Only the third case supports "run one of the commands
 * above", so the other two say what actually happened.
 */
export function noBundleReason(
  truncatedLimit: number,
  unreadable: string[],
  corpus: string
): string {
  const name = corpus || "this corpus";
  if (unreadable.length > 0) {
    return `No bundle for ${name} could be read. ${unreadable.length} bundle files on disk could not be parsed, so this is not proof that no run exists.`;
  }
  if (truncatedLimit > 0) {
    // The scan stops after a fixed number of directory entries, and it takes
    // whichever ones the filesystem returned first. So the unread ones are not
    // the older ones, and the newest run on disk can be among them.
    return `No bundle for ${name} sits in the ${truncatedLimit} run directories this scan reached. The scan stops at a fixed count and reads whichever directories the filesystem lists first, so the newest run can be one it never opened. This is not proof that no run exists.`;
  }
  return `This deployment holds no evidence bundle for ${name}, so there is no recorded run to inspect. Run one of the commands above.`;
}

export async function fetchCompare(
  corpus: string,
  a: string,
  b: string,
  metric: string
): Promise<CompareResult> {
  const query = new URLSearchParams({ corpus, a, b, metric });
  const res = await apiFetch(`${API_URL}/api/compare?${query.toString()}`);
  if (res.status === 401) throw new Error(COMPARE_UNAUTHORIZED);
  if (res.status === 404) throw new Error(COMPARE_MISSING);
  if (!res.ok) throw new Error(COMPARE_UNREADABLE);
  const data = await readJson(res, COMPARE_UNREADABLE);
  // The record prints these fields as measurements. An answer missing any of
  // them would render "not recorded" beside real numbers, which reads as a
  // measured absence rather than a reply this page could not use.
  const ci = data.delta_ci_95;
  // Checked against every field `CompareResult` declares, not a subset. Three
  // review rounds each asked for one more field, which is the shape that never
  // converges. The page renders each of these as a fact, so a 200 that omits
  // one is an unreadable answer rather than a smaller one.
  const numbers = [
    "n_paired",
    "unpaired_a",
    "unpaired_b",
    "mean_a",
    "mean_b",
    "mean_delta",
    "wins",
    "ties",
    "losses",
  ];
  const booleans = [
    "lower_is_better",
    "ci_excludes_zero",
    "significant",
    "enough_pairs_for_inference",
  ];
  const strings = ["metric", "a", "b", "note"];
  const nullableNumbers = ["effect_size_d", "wilcoxon_p"];

  const rowsAreShaped =
    Array.isArray(data.per_question) &&
    data.per_question.every((row) => {
      if (row === null || typeof row !== "object") return false;
      const cells = row as Record<string, unknown>;
      return (
        typeof cells.question_id === "string" &&
        typeof cells.a === "number" &&
        typeof cells.b === "number" &&
        typeof cells.delta === "number"
      );
    });

  const meta = data.meta as Record<string, unknown> | undefined;
  const metaIsShaped =
    typeof meta === "object" &&
    meta !== null &&
    typeof meta.comparable === "boolean" &&
    Array.isArray(meta.reasons) &&
    meta.reasons.every((reason) => typeof reason === "string") &&
    typeof meta.a === "object" &&
    meta.a !== null &&
    typeof meta.b === "object" &&
    meta.b !== null;

  const shaped =
    numbers.every((key) => typeof data[key] === "number") &&
    booleans.every((key) => typeof data[key] === "boolean") &&
    strings.every((key) => typeof data[key] === "string") &&
    nullableNumbers.every((key) => data[key] === null || typeof data[key] === "number") &&
    Array.isArray(ci) &&
    ci.length === 2 &&
    ci.every((bound) => typeof bound === "number") &&
    rowsAreShaped &&
    metaIsShaped;
  if (!shaped) throw new Error(COMPARE_UNREADABLE);
  return data as unknown as CompareResult;
}

// `/strategies` serves `needs_embeddings`, the offline fallback in `lib/api.ts`
// does not. An absent value stays absent rather than reading as false.
export type DecideCatalogRecord = StrategyCatalogRecord & { needs_embeddings?: boolean };

// Every record in the offline fallback carries `status: "unknown"`. A record
// from the server carries `loaded` or `unavailable`. So this tells a live
// catalog from the hardcoded copy, and the copy states the wrong architecture.
export function catalogIsLive(catalog: DecideCatalogRecord[]): boolean {
  return catalog.some((record) => record.status !== "unknown");
}

export interface Candidate {
  name: Strategy;
  label: string;
  reasons: string[];
  runCost: string[];
}

function keyless(record: DecideCatalogRecord): boolean {
  return record.needs_embeddings === false;
}

/**
 * Candidates for one profile, drawn from the default benchmark set.
 *
 * The catalog records no latency and no cost per query, so this ordering
 * cannot claim either. It orders on the one fact the catalog does record
 * about spending: whether the strategy calls the embedding provider for
 * every query. A measured number comes from a run, not from this list.
 */
export function candidatesFor(
  profile: ProfileName,
  catalog: DecideCatalogRecord[],
  live: boolean
): Candidate[] {
  const inDefault = catalog.filter((record) => record.default_benchmark);
  const spendFirst = profile === "latency-bound" || profile === "cost-bound";
  const ordered = [...inDefault].sort((x, y) => {
    if (spendFirst && keyless(x) !== keyless(y)) return keyless(x) ? -1 : 1;
    if (x.experimental !== y.experimental) return x.experimental ? 1 : -1;
    return 0;
  });

  return ordered.map((record) => {
    const reasons = ["Runs under `--strategies all`, so the default benchmark already covers it."];
    if (live) reasons.push(`Catalog architecture: ${record.architecture}.`);
    if (spendFirst && keyless(record)) {
      reasons.push("Calls no embedding provider per query, so it adds no provider latency or cost.");
    }
    if (spendFirst && record.needs_embeddings === true) {
      reasons.push("Calls the embedding provider for every query, which this profile weighs against it.");
    }
    if (record.experimental) {
      reasons.push("Marked experimental in the catalog, so it carries no accuracy claim yet.");
    }

    const runCost: string[] = [];
    if (record.needs_embeddings === false) {
      runCost.push("No API key. A run costs nothing.");
    } else if (record.needs_embeddings === true) {
      runCost.push("Needs the configured embedding provider and its key.");
    } else {
      runCost.push("The deployment catalog could not be read, so the provider cost is unknown.");
    }
    if (record.optional_extra) {
      runCost.push(`Install first: pip install 'kb-arena[${record.optional_extra}]'.`);
    }
    if (record.status === "unavailable" && record.unavailable_reason) {
      runCost.push(`This deployment cannot build it. ${record.unavailable_reason}`);
    }
    if (record.status === "unknown") {
      runCost.push("Runtime availability is unknown because the catalog read failed.");
    }
    return { name: record.name, label: record.label, reasons, runCost };
  });
}

// The same pattern `SAFE_ID` holds in `kb_arena/benchmark/compare.py`, which
// the API applies to every id it accepts. A page that builds a command line a
// reader pastes into a terminal must refuse what the API would refuse. The
// corpus and both strategy names arrive from the URL, so `?corpus=x;rm -rf ~`
// reached a copy button before this.
export const SAFE_ID = /^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}$/;

export const UNSAFE_COMMAND =
  "One of these values carries a character the tool refuses, so this page will not build a command from it.";

export function isSafeId(value: string): boolean {
  return SAFE_ID.test(value);
}

/**
 * A command line, or the refusal, when any value would not survive a shell.
 *
 * Quoting alone is not the fix. A reader edits the line before running it, and
 * a value that the API itself refuses has no business on the clipboard.
 */
function command(parts: string[], values: string[]): string {
  return values.every(isSafeId) ? parts.join(" ") : UNSAFE_COMMAND;
}

/** A command that names strategies, or the refusal when none were picked. */
function strategyCommand(parts: string[], values: string[], names: string[]): string {
  return names.length ? command(parts, values) : NOTHING_PICKED;
}

// StrategyRunPanel refuses an empty pick. This builder expanded it to "all",
// so a reader who unchecked every box read a command that runs the nine
// defaults, each of which can call a paid provider.
export const NOTHING_PICKED = "Pick at least one strategy. An empty pick is not a run.";

function selectionOf(names: string[]): string {
  return names.join(",");
}

export function benchmarkCommand(corpus: string, names: string[]): string {
  const selection = selectionOf(names);
  return strategyCommand(
    ["kb-arena", "benchmark", "--corpus", corpus, "--strategy", selection],
    [corpus, ...names],
    names
  );
}

export function labCommand(corpus: string, names: string[]): string {
  const selection = selectionOf(names);
  return strategyCommand(
    ["kb-arena", "retriever-lab", "--corpus", corpus, "--strategies", selection],
    [corpus, ...names],
    names
  );
}

export function compareCommand(corpus: string, a: string, b: string, metric: string): string {
  return command(
    ["kb-arena", "compare", "--corpus", corpus, "--a", a, "--b", b, "--metric", metric],
    [corpus, a, b, metric]
  );
}

export function labCompareCommand(a: string, b: string, metric: string): string {
  return command(
    [
      "kb-arena",
      "compare",
      "--lab",
      "results/run_<id>/retriever_lab.json",
      "--a",
      a,
      "--b",
      b,
      "--metric",
      metric,
    ],
    [a, b, metric]
  );
}

export function initCommand(corpus: string): string {
  return command(["kb-arena", "init-corpus", corpus], [corpus]);
}

export function ingestCommand(corpus: string): string {
  return command(
    ["kb-arena", "ingest", `./datasets/${corpus}/raw/`, "--corpus", corpus],
    [corpus]
  );
}

/**
 * Why a corpus cannot support a citable decision, or null when it can.
 *
 * A question with no `review_status` counts as `unspecified`, and
 * `review_summary` refuses to publish on unspecified exactly as it refuses on
 * a draft. Warning on the draft count alone left a corpus of unspecified
 * questions looking clean, which is the case the publication gate blocks.
 */
export function reviewWarning(corpus: CorpusInfo): string | null {
  const total = corpus.questionCount ?? 0;
  const reviewed = corpus.reviewedQuestionCount ?? 0;
  const drafts = corpus.draftQuestionCount ?? 0;
  // A question file the server could not parse holds an unknown number of
  // questions with unknown statuses. Reading `reviewed >= total` as a full
  // review would count those as reviewed, which nobody measured.
  const unread = corpus.unreadableQuestionFiles ?? 0;
  if (unread > 0) {
    const files = unread === 1 ? "question file" : "question files";
    return `${unread} ${files} could not be read, so the review status of this corpus is unknown`;
  }
  if (total === 0 || reviewed >= total) return null;
  const unspecified = Math.max(0, total - reviewed - drafts);
  if (reviewed === 0) {
    if (unspecified === 0) return "Machine-drafted, so no decision here is citable";
    if (drafts === 0) {
      return `No question carries a review status, so no decision here is citable`;
    }
    return `${drafts} machine-drafted and ${unspecified} with no review status, so no decision here is citable`;
  }
  const parts: string[] = [];
  if (drafts > 0) parts.push(`${drafts} machine-drafted`);
  if (unspecified > 0) parts.push(`${unspecified} with no review status`);
  return `${parts.join(", ")} of ${total}, so a decision here is not citable`;
}

function fixed(value: number | null | undefined, places: number): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(places) : "not recorded";
}

/**
 * The caveats a bundle forces onto any number taken from its run.
 *
 * `citable` and `why_not_citable` come from the bundle. Deriving them from
 * the reviewed share here would drift from `build_bundle`, which also refuses
 * a bundle that records no command.
 */
export function bundleCaveats(bundle: EvidenceBundle | null): string[] {
  if (!bundle) {
    return [
      "No evidence bundle sits behind these numbers, so the review status of the questions is unknown.",
      "A number with no bundle records no command, no commit and no seed, so nobody can repeat the run.",
    ];
  }
  const lines: string[] = [];
  const review = bundle.review ?? {};
  if (bundle.citable) {
    lines.push("The bundle calls this run citable evidence.");
  } else {
    lines.push("The bundle calls this run a development signal, not citable evidence.");
    lines.push(`Reason the bundle records: ${bundle.why_not_citable || "none recorded"}`);
  }
  const drafts = review.counts?.["machine-assisted-draft"] ?? 0;
  const unspecified = review.counts?.["unspecified"] ?? 0;
  const total = review.questions ?? 0;
  if (drafts > 0) {
    lines.push(
      `${drafts} of ${total} questions are machine-assisted drafts. Nobody checked those answer keys.`
    );
  }
  if (unspecified > 0) {
    lines.push(`${unspecified} of ${total} questions carry no review status.`);
  }
  if (drafts === 0 && unspecified === 0 && total > 0) {
    lines.push(`All ${total} scored questions are marked human-reviewed.`);
  }
  if (review.note) lines.push(review.note);
  if (
    bundle.question_set_fingerprint &&
    bundle.review_question_set &&
    bundle.question_set_fingerprint !== bundle.review_question_set
  ) {
    lines.push(
      `The run measured question set ${bundle.question_set_fingerprint} and the review covers ${bundle.review_question_set}. They differ.`
    );
  }
  return lines;
}

export interface BundleMatch {
  bundle: EvidenceBundle | null;
  describesComparison: boolean;
}

/**
 * The bundle that produced the compared numbers, or the newest one labelled as another run.
 *
 * A corpus can hold many runs. The newest bundle on the corpus is not the run
 * behind a comparison, and quoting its review verdict over another run's
 * numbers is the record claiming more than its evidence. The committed
 * `aws-compute` bundle covers a `retriever-lab` run of bm25 alone, while the
 * comparison reads two benchmark result files, so the two never matched.
 *
 * A bundle attaches only when both compared sides name the run it records.
 */
export function bundleForComparison(
  bundles: EvidenceBundle[],
  comparison: CompareResult | null
): BundleMatch {
  const runA = comparison?.meta?.a?.run_id ?? "";
  const runB = comparison?.meta?.b?.run_id ?? "";
  if (runA && runB && runA === runB) {
    const matched = bundles.find((b) => b.run_id && b.run_id === runA);
    if (matched) return { bundle: matched, describesComparison: true };
  }
  return { bundle: bundles[0] ?? null, describesComparison: false };
}

export interface RecordInput {
  corpus: string;
  corpusQuestions: number | null;
  profile: ProfileName;
  candidates: string[];
  comparison: CompareResult | null;
  comparisonError: string | null;
  bundle: EvidenceBundle | null;
  // False when the bundle records another run on the same corpus. Its review
  // verdict then says nothing about the numbers above it.
  bundleDescribesComparison: boolean;
  bundleError: string | null;
  metric: string;
  createdAt: string;
}

/**
 * The decision record, as markdown a reader can keep.
 *
 * Every number below is copied from the comparison the API returned. The
 * inference lines stay silent when `enough_pairs_for_inference` is false,
 * because below the pair floor no flag fired and printing one would invent it.
 */
export function decisionRecord(input: RecordInput): string {
  const weights = PROFILE_WEIGHTS[input.profile];
  const c = input.comparison;
  const lines: string[] = [];

  lines.push(`# Retrieval decision record for ${input.corpus} under ${input.profile}`);
  lines.push("");
  lines.push(`Written ${input.createdAt} by the KB Arena decision flow.`);
  lines.push("");

  lines.push("## The corpus and the objective this record answers");
  lines.push("");
  lines.push(`- Corpus: ${input.corpus}`);
  lines.push(
    `- Questions the corpus holds: ${
      input.corpusQuestions === null ? "could not be read" : input.corpusQuestions
    }`
  );
  lines.push(`- Objective profile: ${input.profile}`);
  lines.push(
    `- Profile weights: accuracy ${weights.accuracy}, reliability ${weights.reliability}, latency ${weights.latency}, cost ${weights.cost}`
  );
  lines.push(`- Trade-off: ${PROFILE_TRADEOFFS[input.profile]}`);
  lines.push(
    `- The reporter scores latency against a ceiling of ${LATENCY_CEILING_MS} ms and cost against $${COST_CEILING_USD} per query.`
  );
  lines.push("");

  lines.push("## The strategies this record compared");
  lines.push("");
  lines.push(
    `- Candidates the catalog offered for this profile: ${
      input.candidates.length ? input.candidates.join(", ") : "none"
    }`
  );
  lines.push("");

  if (!c) {
    lines.push("## No comparison was read, so this record ranks nothing");
    lines.push("");
    lines.push(input.comparisonError ?? "The comparison step did not run.");
    lines.push("");
  } else {
    lines.push(`## ${c.b} against ${c.a} on ${c.n_paired} paired questions`);
    lines.push("");
    lines.push(`- Metric: ${input.metric}${c.lower_is_better ? " (lower is better)" : ""}`);
    lines.push(`- Baseline ${c.a} mean: ${fixed(c.mean_a, 4)}`);
    lines.push(`- Candidate ${c.b} mean: ${fixed(c.mean_b, 4)}`);
    lines.push(`- Mean delta, b minus a: ${fixed(c.mean_delta, 4)}`);
    lines.push(
      `- 95% bootstrap CI on the mean delta: [${fixed(c.delta_ci_95?.[0], 4)}, ${fixed(
        c.delta_ci_95?.[1],
        4
      )}]`
    );
    lines.push(`- Paired effect size d: ${c.effect_size_d === null ? "undefined, the deltas carry no spread" : fixed(c.effect_size_d, 4)}`);
    lines.push(`- Wilcoxon p: ${c.wilcoxon_p === null ? "not computed" : fixed(c.wilcoxon_p, 6)}`);
    lines.push(`- Wins, ties, losses for ${c.b}: ${c.wins}, ${c.ties}, ${c.losses}`);
    lines.push(
      `- Questions only in a: ${c.unpaired_a}. Questions only in b: ${c.unpaired_b}. Both sit outside the pairing.`
    );
    lines.push("");
    lines.push("### What the pairing supports");
    lines.push("");
    if (!c.enough_pairs_for_inference) {
      lines.push(
        `The pairing holds ${c.n_paired} questions, which is below the floor for inference. No significance flag fired, and this record claims none.`
      );
    } else if (c.significant) {
      lines.push(
        `The Wilcoxon p is below 0.05 and ${c.b} does better on average, so the comparison flags this difference as significant.`
      );
      lines.push(
        c.ci_excludes_zero
          ? "The 95% CI excludes zero."
          : "The 95% CI includes zero, so the interval does not rule out no difference."
      );
    } else {
      lines.push(
        `The comparison did not flag this difference as significant, so this record does not call ${c.b} the winner.`
      );
    }
    lines.push("");
    lines.push(c.note);
    lines.push("");
    lines.push("### What the comparison itself flagged");
    lines.push("");
    lines.push(c.meta.comparable ? "- The two runs are comparable." : "- The two runs are NOT comparable.");
    for (const reason of c.meta.reasons ?? []) lines.push(`- ${reason}`);
    lines.push("");
  }

  // A bundle from another run carries no verdict on the numbers above, so the
  // caveats fall back to the no-bundle set and the mismatch is stated.
  const attached = input.bundleDescribesComparison ? input.bundle : null;

  lines.push("## What this record may not be used for");
  lines.push("");
  if (input.bundleError) lines.push(`- ${input.bundleError}`);
  if (input.bundle && !input.bundleDescribesComparison) {
    lines.push(
      `- The bundle below records run ${
        input.bundle.run_id ?? "with no id"
      }, which did not produce the numbers above. Its review verdict does not cover them.`
    );
  }
  for (const caveat of bundleCaveats(attached)) lines.push(`- ${caveat}`);
  lines.push("");

  if (input.bundle) {
    lines.push(
      input.bundleDescribesComparison
        ? "## The run behind these numbers"
        : "## A different run on this corpus, listed for context only"
    );
    lines.push("");
    lines.push(`- Run id: ${input.bundle.run_id ?? "not recorded"}`);
    lines.push(`- Command: ${(input.bundle.command ?? []).join(" ") || "none recorded"}`);
    lines.push(`- Commit: ${input.bundle.environment?.git_sha ?? "none recorded"}`);
    lines.push(`- Package version: ${input.bundle.environment?.kb_arena ?? "not recorded"}`);
    lines.push(`- Seed: ${input.bundle.seed ?? "not recorded"}`);
    lines.push(`- Written: ${input.bundle.created_at ?? "not recorded"}`);
    lines.push("");
  }

  return lines.join("\n");
}
