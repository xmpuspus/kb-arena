"use client";

import { useServerState } from "@/components/ServerStateProvider";

/**
 * The line every route carries, so a reader always knows what the page shows.
 *
 * `sample` is the route's own claim: it names the built-in content on screen
 * right now. A route that shows recorded data leaves it out. A page that shows
 * sample rows under a real corpus name, with no label, is the defect this
 * closes.
 */
interface Props {
  sample?: string;
}

const CHIP: Record<string, { label: string; color: string }> = {
  checking: { label: "Checking", color: "var(--muted)" },
  unreachable: { label: "No server answer", color: "var(--danger)" },
  "hosted-read-only": { label: "Hosted read-only", color: "var(--warning)" },
  "live-local": { label: "Live local", color: "var(--success)" },
  "live-remote": { label: "Live server", color: "var(--success)" },
  "live-unknown": { label: "Live server", color: "var(--success)" },
  sample: { label: "Sample data", color: "var(--warning)" },
};

export default function StateBanner({ sample }: Props) {
  const { state, writesOff, keyless, refresh } = useServerState();
  const chip = CHIP[sample ? "sample" : state];

  let sentence = "";
  if (state === "checking") {
    sentence = "The page is asking the server which state it runs in.";
  } else if (state === "unreachable") {
    sentence = "The API did not answer, so this page holds no live data.";
  } else if (state === "hosted-read-only") {
    sentence =
      "An operator published this server read-only. It serves recorded results, " +
      "and live questions, arena matches and graph builds stay off.";
  } else if (writesOff && keyless) {
    sentence =
      "This server has no model key, so live questions, arena matches and " +
      "graph builds answer 503. Set a model key to turn them on.";
  } else if (writesOff) {
    sentence =
      "This server answers reads only. Live questions, arena matches and " +
      "graph builds answer 503.";
  } else if (state === "live-remote") {
    sentence = "This server accepts live runs, and it runs on another machine.";
  } else if (state === "live-unknown") {
    sentence =
      "This server accepts live runs. It did not say whether it runs on your machine.";
  } else {
    sentence = "This server runs on your machine and accepts live runs.";
  }

  return (
    <div
      role="status"
      className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border px-3 py-2 text-xs"
      style={{ borderColor: "var(--border)", background: "var(--card)", color: "var(--muted)" }}
    >
      <span
        className="rounded px-1.5 py-0.5 font-semibold"
        style={{ background: "var(--subtle)", color: chip.color }}
      >
        {chip.label}
      </span>
      {sample && <span style={{ color: "var(--foreground)" }}>{sample}</span>}
      <span>{sentence}</span>
      {state === "unreachable" && (
        <button
          type="button"
          onClick={refresh}
          className="rounded border px-2 py-0.5 font-medium transition-opacity hover:opacity-70"
          style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
        >
          Check again
        </button>
      )}
    </div>
  );
}
