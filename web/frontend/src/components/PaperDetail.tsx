import { useEffect, useState } from "react";
import { getPaper } from "../api/papers";
import { updatePaper } from "../api/user";
import type { Paper } from "../api/types";
import { scoreBar } from "./scoreColor";
import MathText from "./MathText";

interface PaperDetailProps {
  arxivId: string | null;
  initialLiked?: number | null;
  score?: number | null;
  onLikedChange?: (arxivId: string, liked: 1 | -1 | 0) => void;
}

export default function PaperDetail({ arxivId, initialLiked, score, onLikedChange }: PaperDetailProps) {
  const [paper, setPaper] = useState<Paper | null>(null);
  const [liked, setLiked] = useState<number | null>(initialLiked ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!arxivId) { setPaper(null); return; }
    setLoading(true);
    setError(null);
    setLiked(initialLiked ?? null);
    getPaper(arxivId)
      .then(setPaper)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load paper");
      })
      .finally(() => setLoading(false));
  }, [arxivId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync liked state when the parent updates it (e.g. toggled from the library list).
  useEffect(() => {
    setLiked(initialLiked ?? null);
  }, [initialLiked]);

  async function handleRate(newLiked: 1 | -1 | 0) {
    if (!arxivId) return;
    setSaving(true);
    try {
      await updatePaper(arxivId, newLiked);
      const val = newLiked === liked ? 0 : newLiked;
      setLiked(val);
      onLikedChange?.(arxivId, val as 1 | -1 | 0);
    } catch {
      // silently ignore rating errors
    } finally {
      setSaving(false);
    }
  }

  if (!arxivId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        Select a paper to see details
      </div>
    );
  }

  if (loading) {
    return <div className="p-6 text-gray-500 text-sm">Loading…</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500 text-sm">{error}</div>;
  }

  if (!paper) return null;

  const authors = paper.authors ?? [];
  const authorStr = authors.length > 5
    ? authors.slice(0, 5).join(", ") + " et al."
    : authors.join(", ");

  const arxivUrl = `https://arxiv.org/abs/${paper.arxiv_id}`;

  return (
    <div className="p-6 overflow-y-auto h-full">
      <div className="flex items-start justify-between gap-4 mb-2">
        <h2 className="text-[23px] font-semibold text-gray-900 leading-snug">{paper.title}</h2>
        {score != null ? (() => {
          const { hue } = scoreBar(score);
          return (
            <span
              className="text-xs font-mono whitespace-nowrap mt-1 shrink-0 px-2 py-0.5 rounded-md cursor-default"
              title="Paper relevance score: 0 = most relevant, -∞ = not relevant"
              style={{
                color: `hsl(${hue}, 70%, 35%)`,
                backgroundColor: `hsla(${hue}, 75%, 50%, 0.1)`,
                border: `1px solid hsla(${hue}, 70%, 45%, 0.5)`,
              }}
            >
              {score.toFixed(3)}
            </span>
          );
        })() : (
          <span
            className="text-xs font-mono whitespace-nowrap mt-1 shrink-0 px-2 py-0.5 rounded-md cursor-default"
            title="No relevance score (not enough liked papers yet)"
            style={{
              color: "#9ca3af",
              backgroundColor: "#f3f4f6",
              border: "1px solid #d1d5db",
            }}
          >
            &#x2015;
          </span>
        )}
      </div>
      <div className="text-base text-gray-500 mb-1">{authorStr}</div>
      <div className="text-base text-gray-400 mb-4">{paper.published_date}</div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={() => handleRate(liked === 1 ? 0 : 1)}
          disabled={saving}
          className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            liked === 1
              ? "bg-green-600 text-white"
              : "bg-gray-100 text-gray-700 hover:bg-green-100"
          }`}
        >
          👍 Relevant
        </button>
        <button
          onClick={() => handleRate(liked === -1 ? 0 : -1)}
          disabled={saving}
          className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            liked === -1
              ? "bg-red-600 text-white"
              : "bg-gray-100 text-gray-700 hover:bg-red-100"
          }`}
        >
          👎 Not Relevant
        </button>
        <a
          href={arxivUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="px-3 py-1.5 rounded text-sm bg-blue-50 text-blue-700 hover:bg-blue-100 transition-colors"
        >
          arXiv:{paper.arxiv_id} ↗
        </a>
      </div>

      {paper.abstract && (
        <div>
          <h3 className="text-lg font-semibold text-gray-600 uppercase tracking-wide mb-1">
            Abstract
            <span className="normal-case font-normal text-gray-400"> (original)</span>
          </h3>
          <p className="text-sm text-gray-700 leading-relaxed">
            <MathText text={paper.abstract} />
          </p>
        </div>
      )}

      {paper.summary && (() => {
        const HEADINGS = ["Keywords", "Scientific Questions", "Data", "Methods", "Results", "Conclusions", "Key takeaway"];
        const re = new RegExp(`^(${HEADINGS.join("|")}):`, "m");
        const parts = paper.summary.split(new RegExp(`(?=^(?:${HEADINGS.join("|")}):)`, "m"));
        return (
          <div className="mt-4">
            <h3 className="text-lg font-semibold text-gray-600 uppercase tracking-wide mb-1">
              Summary
              <span className="normal-case font-normal text-gray-400"> (automatically generated)</span>
            </h3>
            <div className="text-sm text-gray-700 leading-relaxed space-y-[1.12em]">
              {parts.map((part, i) => {
                const m = re.exec(part);
                if (!m) return <p key={i}><MathText text={part.trim()} /></p>;
                const heading = m[1];
                const body = part.slice(m[0].length).trim();
                return (
                  <p key={i}>
                    <span className="font-semibold text-gray-800">{heading}: </span>
                    <MathText text={body} />
                  </p>
                );
              })}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
