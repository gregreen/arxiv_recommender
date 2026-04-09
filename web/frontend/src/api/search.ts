import { apiFetch } from "./client";
import type { SearchResponse, TimeWindow } from "./types";

export async function searchPapers(
  query: string,
  window: TimeWindow
): Promise<SearchResponse> {
  return apiFetch("/api/search", {
    method: "POST",
    body: JSON.stringify({ query, window }),
  });
}
