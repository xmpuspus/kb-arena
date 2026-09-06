"use client";

import { REVIEW_DRAFT, REVIEW_REVIEWED, REVIEW_UNSPECIFIED, type LeaderboardRow } from "@/lib/api";

/**
 * Where a row's numbers came from, beside the numbers.
 *
 * The table showed accuracy and cost and hid the experiment key, the build,
 * and the review status of the questions. A reader who cannot see those three
 * cannot tell an incomparable row from a comparable one, and cannot tell a
 * citable result from a development signal.
 *
 * The page passes each field by name rather than the whole row, so a reader of
 * the page can see which four facts reach this component.
 */
interface Props {
  compatibilityKey: string;
  /** The page shortens the build and names the unrecorded case. */
  buildLabel: string;
  review: LeaderboardRow["review"];
  manifest: LeaderboardRow["manifest"];
  mixedWith: string[];
}

export function reviewSentence(review: LeaderboardRow["review"]): {
  chip: string;
  colour: string;
  sentence: string;
} {
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

export default function RowProvenance({
  compatibilityKey,
  buildLabel,
  review,
  manifest,
  mixedWith,
}: Props) {
  const legacy = compatibilityKey === "legacy";
  const verdict = reviewSentence(review);
  const experiment = legacy
    ? "This result file carries no manifest, so nothing records what it measured."
    : `Judge ${manifest?.judge_model ?? "unrecorded"}, split ${
        manifest?.question_split ?? "unrecorded"
      }, top-k ${manifest?.top_k ?? "unrecorded"}.`;

  return (
    <div className="space-y-1 text-xs">
      <div className="mono" title={experiment}>
        {legacy ? "legacy" : `key ${compatibilityKey.slice(0, 6)}`}
      </div>
      <p style={{ color: "var(--muted)" }}>{experiment}</p>
      <div className="mono" style={{ color: "var(--muted)" }}>
        {buildLabel}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <span
          className="rounded px-1.5 py-0.5 font-semibold"
          style={{ background: "var(--subtle)", color: verdict.colour }}
        >
          {verdict.chip}
        </span>
        <span style={{ color: "var(--muted)" }}>{verdict.sentence}</span>
      </div>
      {mixedWith.length > 0 && (
        <p style={{ color: "var(--muted)" }}>
          {mixedWith.length} other experiment{mixedWith.length === 1 ? "" : "s"} ran on this corpus
          and strategy. Those rows measured something else, so do not read them as one ranking.
        </p>
      )}
    </div>
  );
}
