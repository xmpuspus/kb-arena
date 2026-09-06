"use client";

import { useCallback, useEffect, useState } from "react";
import StateBanner from "@/components/StateBanner";
import FetchError from "@/components/FetchError";
import { fetchCorporaResult, type CorpusInfo } from "@/lib/api";
import {
  readHealth,
  readStrategies,
  type HealthReport,
  type Read,
  type StrategyStatus,
} from "@/lib/diagnostics";

/**
 * The operator page: what this server loaded, and what it refuses.
 *
 * Every line below comes from a read of `/health`, `/strategies` or
 * `/api/corpora`. A read that fails prints its failure. The page never fills a
 * gap with a built-in default, because a default here would report a
 * capability that nobody measured.
 */
const NOT_REPORTED = "This server did not report it.";

function Section({ heading, children }: { heading: string; children: React.ReactNode }) {
  return (
    <section
      className="rounded-lg border p-4 space-y-2"
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
    >
      <h2 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
        {heading}
      </h2>
      <div className="text-sm space-y-1" style={{ color: "var(--muted)" }}>{children}</div>
    </section>
  );
}

function modelKey(health: HealthReport): { heading: string; body: string } {
  const provider = health.llmProvider ?? "the configured provider";
  if (health.llmConfigured === null) {
    return {
      heading: "The model key state is unknown",
      body: NOT_REPORTED,
    };
  }
  if (!health.llmConfigured) {
    return {
      heading: `No model key is configured for ${provider}`,
      body:
        "Live questions, arena matches and graph builds answer 503. " +
        "Recorded results, the leaderboard and the benchmark pages still read.",
    };
  }
  if (health.llmAvailable === false) {
    return {
      heading: `A key is set for ${provider}, and the client did not load`,
      body: "The server holds a key and built no client from it, so live calls fail.",
    };
  }
  // A missing field is not a yes. This branch used to fall through to the
  // success answer, so a server that never reported availability read as one
  // that can call the model. That is the built-in-list mistake this page exists
  // to avoid.
  if (health.llmAvailable === null) {
    return {
      heading: `A key is set for ${provider}, and the client state is unknown`,
      body: NOT_REPORTED,
    };
  }
  return {
    heading: `A model key is configured for ${provider}`,
    body: "Live questions, arena matches and graph builds can call the model.",
  };
}

function graph(health: HealthReport): { heading: string; body: string } {
  if (health.neo4jConnected === null) {
    return { heading: "The graph database state is unknown", body: NOT_REPORTED };
  }
  if (health.neo4jConnected) {
    return {
      heading: "The graph database answers",
      body: `The server holds a driver for ${health.neo4jUri ?? "the configured address"}. ` +
        "Graph retrieval, the graph page and graph builds can run.",
    };
  }
  return {
    heading: "The graph database does not answer",
    body:
      "knowledge_graph, lightrag and hybrid cannot retrieve. " +
      (health.neo4jLastError
        ? `The server reported: ${health.neo4jLastError}`
        : "The server reported no reason."),
  };
}

function arena(health: HealthReport): { heading: string; body: string } {
  if (health.arenaAvailable === null) {
    return { heading: "The arena state is unknown", body: NOT_REPORTED };
  }
  if (health.arenaAvailable) {
    return { heading: "The arena accepts matches", body: "Votes reach the stored ELO ratings." };
  }
  return {
    heading: "The arena does not accept matches",
    body: health.arenaLastError
      ? `The server reported: ${health.arenaLastError}`
      : "The server reported no reason.",
  };
}

function corpusLine(corpus: CorpusInfo): string {
  const total = corpus.questionCount ?? 0;
  // A file the server could not parse holds an unknown number of questions with
  // unknown statuses, so neither count below describes the whole corpus.
  const unread = corpus.unreadableQuestionFiles ?? 0;
  if (unread > 0) {
    const files = unread === 1 ? "question file" : "question files";
    return `${unread} ${files} could not be read, so the counts here cover only part of this corpus.`;
  }
  if (!total) return "No question file, so no run can score it.";
  const reviewed = corpus.reviewedQuestionCount;
  const draft = corpus.draftQuestionCount;
  if (reviewed === undefined || draft === undefined) {
    return `${total} questions. This server did not report how many a reviewer checked.`;
  }
  const unlabelled = total - reviewed - draft;
  const parts = [`${total} questions`, `${reviewed} reviewed`, `${draft} machine drafts`];
  if (unlabelled > 0) parts.push(`${unlabelled} with no review status`);
  const verdict =
    reviewed === total
      ? "A reviewer checked every question, so a run on it can be cited."
      : "A run on it is a development signal, not citable evidence.";
  return `${parts.join(", ")}. ${verdict}`;
}

export default function DiagnosticsPage() {
  const [health, setHealth] = useState<Read<HealthReport> | null>(null);
  const [strategies, setStrategies] = useState<Read<StrategyStatus[]> | null>(null);
  const [corpora, setCorpora] = useState<{ corpora: CorpusInfo[]; failed: boolean } | null>(null);
  const [attempt, setAttempt] = useState(0);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setHealth(null);
    setStrategies(null);
    setCorpora(null);
    readHealth(controller.signal)
      .then((result) => active && setHealth(result))
      .catch(() => undefined);
    readStrategies(controller.signal)
      .then((result) => active && setStrategies(result))
      .catch(() => undefined);
    fetchCorporaResult().then((result) => active && setCorpora(result));
    return () => {
      active = false;
      controller.abort();
    };
  }, [attempt]);

  const loaded = strategies?.ok ? strategies.value.filter((s) => s.status === "loaded") : [];
  const missing = strategies?.ok ? strategies.value.filter((s) => s.status !== "loaded") : [];
  const withQuestions = corpora?.corpora.filter((c) => (c.questionCount ?? 0) > 0) ?? [];

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      <StateBanner />

      <header>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          This page reports only what this server answered
        </h1>
        <p className="text-sm mt-2 max-w-2xl" style={{ color: "var(--muted)" }}>
          Every line below comes from a read of /health, /strategies or /api/corpora on this
          deployment. A read that fails says so. Nothing here falls back to the built-in lists,
          because a built-in list would name a capability nobody measured.
        </p>
      </header>

      {health === null && <p className="text-sm">Reading /health...</p>}

      {health && !health.ok && (
        <FetchError
          title="The health read did not land"
          message={health.reason}
          hint="Without it this page cannot say what the deployment does, and a guess would be worse than nothing. Start the API, then try again."
          onRetry={retry}
        />
      )}

      {health?.ok && (
        <div className="space-y-3">
          <Section {...toSection(modelKey(health.value))} />
          <Section {...toSection(graph(health.value))} />
          <Section {...toSection(arena(health.value))} />
        </div>
      )}

      {strategies === null && <p className="text-sm">Reading /strategies...</p>}

      {strategies && !strategies.ok && (
        <FetchError
          title="The strategy read did not land"
          message={strategies.reason}
          hint="The built-in list names what KB Arena ships, not what this server loaded, so it stays off this page. Try again."
          onRetry={retry}
        />
      )}

      {strategies?.ok && (
        <Section
          heading={`${loaded.length} of ${strategies.value.length} strategies loaded here`}
        >
          <p style={{ color: "var(--foreground)" }}>
            Loaded: {loaded.length ? loaded.map((s) => s.name).join(", ") : "none"}.
          </p>
          {missing.length > 0 && (
            <ul className="space-y-1 pt-1">
              {missing.map((s) => (
                <li key={s.name}>
                  <span className="mono" style={{ color: "var(--foreground)" }}>
                    {s.name}
                  </span>{" "}
                  did not load. {s.unavailableReason ?? "The server reported no reason."}
                </li>
              ))}
            </ul>
          )}
        </Section>
      )}

      {corpora === null && <p className="text-sm">Reading /api/corpora...</p>}

      {corpora?.failed && (
        <FetchError
          title="The corpus read did not land"
          message="The API did not answer /api/corpora."
          hint="The built-in corpus list would name corpora this server may not hold, so it stays off this page. Try again."
          onRetry={retry}
        />
      )}

      {corpora && !corpora.failed && (
        <Section
          heading={
            withQuestions.length
              ? `${withQuestions.length} of ${corpora.corpora.length} corpora hold questions`
              : "No corpus here holds questions"
          }
        >
          {corpora.corpora.length === 0 && <p>This server holds no corpus.</p>}
          <ul className="space-y-1">
            {corpora.corpora.map((c) => (
              <li key={c.value}>
                <span className="mono" style={{ color: "var(--foreground)" }}>
                  {c.value}
                </span>{" "}
                {corpusLine(c)}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </div>
  );
}

function toSection({ heading, body }: { heading: string; body: string }) {
  return { heading, children: <p>{body}</p> };
}
