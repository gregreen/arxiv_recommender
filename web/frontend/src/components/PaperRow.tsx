import { memo } from "react";
import type { Recommendation } from "../api/types";
import { scoreBar } from "./scoreColor";
import { formatTimestamp } from "../utils";
import MathText from "./MathText";

interface PaperRowProps {
  rec: Recommendation;
  selected: boolean;
  onClick: () => void;
}

function PaperRow({ rec, selected, onClick }: PaperRowProps) {
  const scored = rec.score != null;
  const { pct, color } = scored ? scoreBar(rec.score!) : { pct: 0, color: "#d1d5db" };

  let likedClass = "";
  if (rec.liked === 1) likedClass = "bg-green-50 border-green-200";
  else if (rec.liked === -1) likedClass = "bg-red-50 border-red-200";
  else likedClass = "bg-white border-gray-200";
  if (selected) likedClass = "bg-blue-50 border-blue-400";

  return (
    <div
      onClick={onClick}
      className={`cursor-pointer border rounded p-3 mb-1.5 hover:border-blue-300 transition-colors ${likedClass}`}
    >
      <div className="text-sm font-medium text-gray-800 leading-snug line-clamp-2"><MathText text={rec.title} /></div>
      <div className="flex items-center gap-2 mt-1.5">
        <div className="flex-1 bg-gray-200 rounded-full h-1.5">
          <div
            className="h-1.5 rounded-full transition-all"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
        <span className="text-xs text-gray-400 whitespace-nowrap">
          {formatTimestamp(rec.published_date)}
        </span>
      </div>
    </div>
  );
}

export default memo(PaperRow);
