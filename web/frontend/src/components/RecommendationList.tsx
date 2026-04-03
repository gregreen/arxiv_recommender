import { useState, useEffect, useCallback } from "react";
import { getRecommendations } from "../api/recommendations";
import type { TimeWindow, Recommendation } from "../api/types";
import PaperRow from "./PaperRow";

interface RecommendationListProps {
  selectedArxivId: string | null;
  onSelect: (arxivId: string, liked: number | null, score: number | null) => void;
  likedCache?: Record<string, number>;
}

const WINDOWS: { label: string; value: TimeWindow }[] = [
  { label: "Day", value: "day" },
  { label: "Week", value: "week" },
  { label: "Month", value: "month" },
];

export default function RecommendationList({ selectedArxivId, onSelect, likedCache = {} }: RecommendationListProps) {
  const [window, setWindow] = useState<TimeWindow>("week");
  const [results, setResults] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [onboarding, setOnboarding] = useState(false);

  const fetchRecs = useCallback(
    async (win: TimeWindow) => {
      setLoading(true);
      setError(null);
      try {
        const data = await getRecommendations(win);
        setResults(data.results);
        setOnboarding(data.onboarding ?? false);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load recommendations");
        setResults([]);
        setOnboarding(false);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    fetchRecs(window);
    const timer = setInterval(() => fetchRecs(window), 30_000);
    return () => clearInterval(timer);
  }, [window, fetchRecs]);


  return (
    <div className="flex flex-col h-full">
      {/* Tabs */}
      <div className="flex gap-1 p-3 border-b border-gray-200 shrink-0">
        {WINDOWS.map((w) => (
          <button
            key={w.value}
            onClick={() => setWindow(w.value)}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              window === w.value
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {w.label}
          </button>
        ))}
        <button
          onClick={() => fetchRecs(window)}
          disabled={loading}
          className="ml-auto text-xs text-gray-400 hover:text-gray-600 px-2"
          title="Refresh"
        >
          ↻
        </button>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto p-3">
        {loading && results.length === 0 && (
          <div className="text-sm text-gray-400 text-center mt-8">Loading…</div>
        )}
        {error && (
          <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-3 mt-4">
            {error}
          </div>
        )}
        {onboarding && (
          <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-3 mb-3">
            Not enough data to generate recommendations yet. Mark papers as relevant or add papers to your Library.
          </div>
        )}
        {!loading && !error && !onboarding && results.length === 0 && (
          <div className="text-sm text-gray-400 text-center mt-8">No recommendations yet.</div>
        )}
        {results.map((rec) => (
          <PaperRow
            key={rec.arxiv_id}
            rec={{ ...rec, liked: likedCache[rec.arxiv_id] ?? rec.liked }}
            selected={rec.arxiv_id === selectedArxivId}
            onClick={() => onSelect(rec.arxiv_id, rec.liked, rec.score)}
          />
        ))}
      </div>
    </div>
  );
}
