import { apiFetch } from "./client";
import type { SearchResponse } from "./types";

export async function searchPapers(query: string): Promise<SearchResponse> {
  return apiFetch("/api/search", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}
