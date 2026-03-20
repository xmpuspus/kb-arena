"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "";

const EXAMPLE_QUESTIONS = [
  "How does AWS Lambda handle cold starts?",
  "What is the difference between EC2 and Lambda?",
  "How do you configure auto-scaling for ECS?",
  "What are the best practices for S3 bucket security?",
];

interface MatchResult {
  match_id: string;
  question: string;
  answer_a: string;
  answer_b: string;
  latency_a_ms: number;
  latency_b_ms: number;
  sources_a: string[];
  sources_b: string[];
}

interface VoteResult {
  strategy_a: string;
  strategy_b: string;
  winner: string;
  elo: Record<string, number>;
  total_votes: number;
}

interface LeaderboardEntry {
  strategy: string;
  elo: number;
  wins: number;
  losses: number;
  ties: number;
  matches: number;
}

export default function ArenaPage() {
  const [question, setQuestion] = useState("");
  const [match, setMatch] = useState<MatchResult | null>(null);
  const [voteResult, setVoteResult] = useState<VoteResult | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [voting, setVoting] = useState(false);
  const [totalVotes, setTotalVotes] = useState(0);

  async function fetchLeaderboard() {
    try {
      const res = await fetch(`${API}/api/arena/leaderboard`);
      const data = await res.json();
      setLeaderboard(data.leaderboard || []);
      setTotalVotes(data.total_votes || 0);
    } catch {
      // ignore
    }
  }

  async function createMatch() {
    if (!question.trim()) return;
    setLoading(true);
    setMatch(null);
    setVoteResult(null);
    try {
      const res = await fetch(`${API}/api/arena/match`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error.message || "Failed to create match");
        return;
      }
      setMatch(data);
    } catch {
      alert("Failed to create match. Is the server running?");
    } finally {
      setLoading(false);
    }
  }

  async function vote(winner: "a" | "b" | "tie") {
    if (!match) return;
    setVoting(true);
    try {
      const res = await fetch(`${API}/api/arena/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ match_id: match.match_id, winner }),
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error.message || "Vote failed");
        return;
      }
      setVoteResult(data);
      fetchLeaderboard();
    } catch {
      alert("Vote failed");
    } finally {
      setVoting(false);
    }
  }

  // Fetch leaderboard on mount
  useEffect(() => {
    fetchLeaderboard();
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Strategy Arena
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Blind A/B comparison of retrieval strategies. Vote for the better answer. ELO rankings emerge.
        </p>
        {totalVotes > 0 && (
          <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
            {totalVotes} votes recorded
          </p>
        )}
      </div>

      {/* Question Input */}
      <div className="space-y-3 max-w-2xl mx-auto">
        <div className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && createMatch()}
            placeholder="Ask a question about your documentation..."
            className="flex-1 px-4 py-2.5 rounded-lg border text-sm outline-none"
            style={{
              background: "var(--card)",
              borderColor: "var(--border)",
              color: "var(--foreground)",
            }}
            disabled={loading}
          />
          <button
            onClick={createMatch}
            disabled={loading || !question.trim()}
            className="px-5 py-2.5 rounded-lg text-sm font-medium transition-opacity disabled:opacity-30"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            {loading ? "Matching..." : "Battle"}
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => setQuestion(q)}
              className="text-xs px-2.5 py-1 rounded-lg border transition-opacity hover:opacity-70 text-left"
              style={{ borderColor: "var(--border)", color: "var(--muted)" }}
            >
              {q.length > 50 ? q.slice(0, 50) + "..." : q}
            </button>
          ))}
        </div>
      </div>

      {/* Match Results */}
      {match && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Strategy A */}
            <div
              className="p-6 rounded-lg border-2 transition-colors"
              style={{
                borderColor: voteResult?.winner === "a"
                  ? "#22c55e"
                  : voteResult?.winner === "b"
                  ? "var(--border)"
                  : "var(--border)",
                background: voteResult?.winner === "a"
                  ? "color-mix(in srgb, #22c55e 8%, var(--card))"
                  : "var(--card)",
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold" style={{ color: "var(--foreground)" }}>
                  {voteResult ? voteResult.strategy_a : "Strategy A"}
                </h3>
                <span className="text-xs" style={{ color: "var(--muted)" }}>
                  {match.latency_a_ms.toFixed(0)}ms
                </span>
              </div>
              <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: "var(--foreground)" }}>
                {match.answer_a}
              </p>
              {match.sources_a.length > 0 && (
                <p className="text-xs mt-3" style={{ color: "var(--muted)" }}>
                  Sources: {match.sources_a.join(", ")}
                </p>
              )}
            </div>

            {/* Strategy B */}
            <div
              className="p-6 rounded-lg border-2 transition-colors"
              style={{
                borderColor: voteResult?.winner === "b"
                  ? "#22c55e"
                  : voteResult?.winner === "a"
                  ? "var(--border)"
                  : "var(--border)",
                background: voteResult?.winner === "b"
                  ? "color-mix(in srgb, #22c55e 8%, var(--card))"
                  : "var(--card)",
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold" style={{ color: "var(--foreground)" }}>
                  {voteResult ? voteResult.strategy_b : "Strategy B"}
                </h3>
                <span className="text-xs" style={{ color: "var(--muted)" }}>
                  {match.latency_b_ms.toFixed(0)}ms
                </span>
              </div>
              <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: "var(--foreground)" }}>
                {match.answer_b}
              </p>
              {match.sources_b.length > 0 && (
                <p className="text-xs mt-3" style={{ color: "var(--muted)" }}>
                  Sources: {match.sources_b.join(", ")}
                </p>
              )}
            </div>
          </div>

          {/* Vote Buttons */}
          {!voteResult && (
            <div className="flex justify-center gap-3">
              <button
                onClick={() => vote("a")}
                disabled={voting}
                className="px-5 py-2 rounded-lg text-sm font-medium transition-opacity disabled:opacity-50"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                A is better
              </button>
              <button
                onClick={() => vote("tie")}
                disabled={voting}
                className="px-5 py-2 rounded-lg border text-sm font-medium transition-opacity disabled:opacity-50 hover:opacity-70"
                style={{ borderColor: "var(--border)", color: "var(--muted)" }}
              >
                Tie
              </button>
              <button
                onClick={() => vote("b")}
                disabled={voting}
                className="px-5 py-2 rounded-lg text-sm font-medium transition-opacity disabled:opacity-50"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                B is better
              </button>
            </div>
          )}

          {/* Vote Result */}
          {voteResult && (
            <div className="text-center space-y-2">
              <p className="text-sm" style={{ color: "var(--muted)" }}>
                {voteResult.winner === "tie"
                  ? "Tie - ELO unchanged"
                  : `${voteResult.winner === "a" ? voteResult.strategy_a : voteResult.strategy_b} wins`}
              </p>
              <button
                onClick={() => {
                  setMatch(null);
                  setVoteResult(null);
                  setQuestion("");
                }}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                Next match
              </button>
            </div>
          )}
        </div>
      )}

      {/* Leaderboard */}
      {leaderboard.length > 0 && (
        <div className="max-w-2xl mx-auto">
          <h2 className="text-base font-semibold mb-3" style={{ color: "var(--foreground)" }}>
            ELO Leaderboard
          </h2>
          <div
            className="rounded-lg border overflow-hidden"
            style={{ borderColor: "var(--border)" }}
          >
            <table className="w-full">
              <thead>
                <tr style={{ background: "var(--card)", borderBottom: "1px solid var(--border)" }}>
                  <th className="text-left py-2.5 px-4 text-xs font-medium" style={{ color: "var(--muted)" }}>#</th>
                  <th className="text-left py-2.5 px-4 text-xs font-medium" style={{ color: "var(--muted)" }}>Strategy</th>
                  <th className="text-right py-2.5 px-4 text-xs font-medium" style={{ color: "var(--muted)" }}>ELO</th>
                  <th className="text-right py-2.5 px-4 text-xs font-medium" style={{ color: "var(--muted)" }}>W / L / T</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((entry, i) => (
                  <tr
                    key={entry.strategy}
                    style={{ borderBottom: "1px solid var(--border)" }}
                  >
                    <td className="py-2.5 px-4 text-xs" style={{ color: "var(--muted)" }}>{i + 1}</td>
                    <td className="py-2.5 px-4 text-sm font-medium" style={{ color: "var(--foreground)" }}>
                      {entry.strategy}
                    </td>
                    <td className="py-2.5 px-4 text-sm text-right font-mono" style={{ color: "var(--foreground)" }}>
                      {entry.elo}
                    </td>
                    <td className="py-2.5 px-4 text-xs text-right" style={{ color: "var(--muted)" }}>
                      {entry.wins} / {entry.losses} / {entry.ties}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
