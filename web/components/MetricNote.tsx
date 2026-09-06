"use client";

import InfoTip from "@/components/InfoTip";

/**
 * What a metric measures, and what it leaves out, next to the number itself.
 *
 * A reader who sees 0.92 with no definition cannot tell a retrieval score from
 * an answer score, or a whole run from one question. Every line below is a
 * plain-English reading of `docs/methodology.md`, so the page and the method
 * document cannot drift into two different claims.
 */
export const METHODOLOGY_SOURCE = "docs/methodology.md";

export type MetricKey =
  | "accuracy"
  | "recall_at_5"
  | "ndcg_at_5"
  | "cost_per_query"
  | "cost_per_run"
  | "latency";

interface Note {
  label: string;
  /** What the number measures. */
  measures: string;
  /** What the number does not measure. */
  excludes: string;
  /** The heading in the method document this reading comes from. */
  section: string;
}

export const METRIC_NOTES: Record<MetricKey, Note> = {
  accuracy: {
    label: "accuracy",
    measures:
      "A judge model grades each generated answer against the ground-truth answer. Accuracy is the mean of those grades, so it covers retrieval and generation together.",
    excludes:
      "It does not name the chunks the strategy retrieved. A full answer score can hide a retrieval failure, so read it with the retrieval trace.",
    section: "Full answer evaluation",
  },
  recall_at_5: {
    label: "Recall@5",
    measures:
      "Recall@5 counts the chunks a reviewer marked relevant that reach the top five results. It measures retrieval alone, with no answer generation.",
    excludes:
      "It does not reward a high rank inside those five, and it needs relevance judgments. A corpus with no chunk labels reports nothing here.",
    section: "Retrieval-only",
  },
  ndcg_at_5: {
    label: "NDCG@5",
    measures:
      "NDCG@5 scores the same five results and rewards a relevant chunk that lands nearer the top. It measures retrieval alone.",
    excludes:
      "It does not measure the answer. Two numbers compare only when the runs share the corpus, question set, qrels, metric code, and top-k definition.",
    section: "Fair comparisons",
  },
  cost_per_query: {
    label: "cost per query",
    measures:
      "The mean provider spend for one question, in US dollars. It covers the model calls the run made while it answered and graded.",
    excludes:
      "It does not cover index-time work, and graph, generated-index, and hierarchical methods do more of that. A provider price change moves it.",
    section: "Full answer evaluation",
  },
  cost_per_run: {
    label: "cost per run",
    measures:
      "The mean total spend of the runs on this row, in US dollars. It is one whole run, not one question.",
    excludes:
      "It does not cover index-time work. Two rows with different question counts do not compare until you divide by the question count.",
    section: "Evidence record",
  },
  latency: {
    label: "latency",
    measures:
      "The mean wall-clock time for one question, in milliseconds. It covers retrieval and answer generation.",
    excludes:
      "It does not cover index-time work. Provider load and the machine that ran the benchmark both move it, so a repeat gives a different number.",
    section: "Full answer evaluation",
  },
};

export function metricText(metric: MetricKey): string {
  const note = METRIC_NOTES[metric];
  return `${note.measures} ${note.excludes} Source: ${METHODOLOGY_SOURCE}, ${note.section}.`;
}

interface Props {
  metric: MetricKey;
  align?: "center" | "left";
}

export default function MetricNote({ metric, align = "center" }: Props) {
  const note = METRIC_NOTES[metric];
  return <InfoTip label={note.label} text={metricText(metric)} align={align} />;
}
