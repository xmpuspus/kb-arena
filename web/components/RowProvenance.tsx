"use client";

import { REVIEW_DRAFT, REVIEW_REVIEWED, REVIEW_UNSPECIFIED, type LeaderboardRow } from "@/lib/api";

/**
 * Where a row's numbers came from, beside the numbers.
 *
 * The table showed accuracy and cost and hid the experiment key, the build,
 * and the review status of the questions. A reader who cannot see those three
 * cannot tell an incomparable row from a comparable one, and cannot tell a
 * citable result from a development signal.
 */
interface Props {
  row: LeaderboardRow;
}

export function reviewSentence(row: LeaderboardRow): {
  chip: string;
  colour: string;
  sentence: string;
} {
  const review = row.review;
  // A server older than the review summary reports real numbers and no
  // review. Reading that gap as "not citable" would be a claim about
  // questions nobody described.
  if (!review) {
    return {
      chip: "Review unknown",
      colour: "var(--muted)",
      sentence: "This server reported no review status for the questions behind this row.",
    };
  }
  if (review.questions === 0) {
    return {
      chip: "Review unknown",
      colour: "var(--muted)",
      sentence: "This row scored no question, so there is nothing to review.",
    };
  }
  const reviewed = review.counts[REVIEW_REVIEWED] ?? 0;
  const draft = review.counts[REVIEW_DRAFT] ?? 0;
  const unspecified = review.counts[REVIEW_UNSPECIFIED] ?? 0;
  if (review.publishable) {
    return {
      chip: "Citable",
      colour: "var(--success)",
      sentence: `A reviewer checked all ${review.questions} scored questions.`,
    };
  }
  const parts = [`${reviewed} of ${review.questions} questions are reviewed`];
  if (draft) parts.push(`${draft} are machine drafts`);
  if (unspecified) parts.push(`${unspecified} carry no review status`);
  return {
    chip: "Not citable",
    colour: "var(--warning)",
    sentence: `${parts.join(", ")}.`,
  };
}

export default function RowProvenance({ row }: Props) {
  const legacy = row.compatibility_key === "legacy";
  const review = reviewSentence(row);
  const experiment = legacy
    ? "This result file carries no manifest, so nothing records what it measured."
    : `Judge ${row.manifest?.judge_model ?? "unrecorded"}, split ${
        row.manifest?.question_split ?? "unrecorded"
      }, top-k ${row.manifest?.top_k ?? "unrecorded"}.`;

  return (
    <div className="space-y-1 text-xs">
      <div className="mono" title={experiment}>
        {legacy ? "legacy" : `key ${row.compatibility_key.slice(0, 6)}`}
      </div>
      <p style={{ color: "var(--muted)" }}>{experiment}</p>
      <div className="mono" style={{ color: "var(--muted)" }}>
        {row.build && row.build !== "unrecorded"
          ? `build ${row.build.length > 14 ? `${row.build.slice(0, 14)}...` : row.build}`
          : "build unrecorded"}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <span
          className="rounded px-1.5 py-0.5 font-semibold"
          style={{ background: "var(--subtle)", color: review.colour }}
        >
          {review.chip}
        </span>
        <span style={{ color: "var(--muted)" }}>{review.sentence}</span>
      </div>
      {row.mixed_with?.length > 0 && (
        <p style={{ color: "var(--muted)" }}>
          {row.mixed_with.length} other experiment{row.mixed_with.length === 1 ? "" : "s"} ran on
          this corpus and strategy. Those rows measured something else, so do not read them as one
          ranking.
        </p>
      )}
    </div>
  );
}
