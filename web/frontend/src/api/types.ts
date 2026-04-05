export interface Author {
  name: string;
}

export interface Paper {
  arxiv_id: string;
  title: string;
  abstract: string | null;
  authors: string[];
  published_date: string | null;
  categories: string[];
  summary: string | null;
  url: string;
}

export interface Recommendation {
  arxiv_id: string;
  title: string;
  authors: string[];
  published_date: string | null;
  score: number | null;
  rank: number | null;
  liked: number | null;
  generated_at: string | null;
}

export interface RecommendationsResponse {
  window: string;
  count: number;
  generated_at: string | null;
  onboarding: boolean;
  message: string | null;
  results: Recommendation[];
}

export interface UserPaper {
  arxiv_id: string;
  title: string | null;
  authors: string[];
  published_date: string | null;
  liked: number;
  added_at: string;
}

export interface ImportResult {
  imported: number;
  skipped: number;
  rate_limited: number;
}

export type TimeWindow = "day" | "week" | "month";
