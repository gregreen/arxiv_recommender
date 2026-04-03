import { apiFetch } from "./client";
import type { RecommendationsResponse, TimeWindow } from "./types";

export async function getRecommendations(
  window: TimeWindow
): Promise<RecommendationsResponse> {
  return apiFetch(`/api/recommendations?window=${window}`);
}
