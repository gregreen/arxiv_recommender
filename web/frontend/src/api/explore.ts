import type { ExploreResponse, TimeWindow } from "./types";
import { apiFetch } from "./client";

export async function getExplore(window: TimeWindow): Promise<ExploreResponse> {
  return apiFetch<ExploreResponse>(`/api/explore?window=${window}`);
}
