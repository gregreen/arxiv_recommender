import { apiFetch } from "./client";
import type { Paper } from "./types";

export async function getPaper(arxiv_id: string): Promise<Paper> {
  return apiFetch(`/api/papers/${encodeURIComponent(arxiv_id)}`);
}
