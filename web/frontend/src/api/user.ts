import { apiFetch } from "./client";
import type { ImportResult, UserPaper } from "./types";

export async function getMyPapers(): Promise<UserPaper[]> {
  return apiFetch("/api/users/me/papers");
}

export async function addPaper(
  arxiv_id: string,
  liked: number = 1
): Promise<{ arxiv_id: string; liked: number }> {
  return apiFetch("/api/users/me/papers", {
    method: "POST",
    body: JSON.stringify({ arxiv_id, liked }),
  });
}

export async function updatePaper(
  arxiv_id: string,
  liked: number
): Promise<{ arxiv_id: string; liked: number }> {
  return apiFetch(`/api/users/me/papers/${encodeURIComponent(arxiv_id)}`, {
    method: "PATCH",
    body: JSON.stringify({ liked }),
  });
}

export async function deletePaper(arxiv_id: string): Promise<void> {
  return apiFetch(`/api/users/me/papers/${encodeURIComponent(arxiv_id)}`, { method: "DELETE" });
}

export async function importAds(text: string): Promise<ImportResult> {
  return apiFetch("/api/users/me/papers/import/ads", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}
