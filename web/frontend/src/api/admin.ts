import { apiFetch } from "./client";

export interface AdminUser {
  id: number;
  email: string;
  is_active: boolean;
  is_admin: boolean;
  email_verified: boolean;
  created_at: string;
  paper_count: number;
  model_trained_at: string | null;
  import_count: number;
}

export interface AdminTask {
  id: number;
  type: "fetch_meta" | "embed" | "recommend";
  payload: string;
  status: "pending" | "running" | "done" | "failed";
  priority: number | null;
  attempts: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export interface AdminPaper {
  arxiv_id: string;
  title: string | null;
  authors: string | null;
  published_date: string | null;
  categories: string | null;
  embedded_at: string;
}

export interface Paginated<T> {
  total: number;
  offset: number;
  limit: number;
  items: T[];
}

export function getAdminUsers(): Promise<AdminUser[]> {
  return apiFetch("/api/admin/users");
}

export function patchAdminUser(userId: number, is_active: boolean): Promise<{ user_id: number; is_active: boolean }> {
  return apiFetch(`/api/admin/users/${userId}`, {
    method: "PATCH",
    body: JSON.stringify({ is_active }),
  });
}

export function getAdminTasks(params?: {
  type?: string;
  status?: string;
  q?: string;
  limit?: number;
  offset?: number;
}): Promise<Paginated<AdminTask>> {
  const q = new URLSearchParams();
  if (params?.type)   q.set("type",   params.type);
  if (params?.status) q.set("status", params.status);
  if (params?.q)      q.set("q",      params.q);
  if (params?.limit  !== undefined) q.set("limit",  String(params.limit));
  if (params?.offset !== undefined) q.set("offset", String(params.offset));
  const qs = q.toString() ? `?${q}` : "";
  return apiFetch(`/api/admin/tasks${qs}`);
}

export function resetAdminTask(taskId: number): Promise<AdminTask> {
  return apiFetch(`/api/admin/tasks/${taskId}/reset`, { method: "POST" });
}

export function deleteAdminTask(taskId: number): Promise<void> {
  return apiFetch(`/api/admin/tasks/${taskId}`, { method: "DELETE" });
}

export function resetUserImportLog(userId: number): Promise<void> {
  return apiFetch(`/api/admin/users/${userId}/import-log`, { method: "DELETE" });
}

export function getAdminPapers(params?: {
  q?: string;
  limit?: number;
  offset?: number;
}): Promise<Paginated<AdminPaper>> {
  const q = new URLSearchParams();
  if (params?.q)      q.set("q",      params.q);
  if (params?.limit  !== undefined) q.set("limit",  String(params.limit));
  if (params?.offset !== undefined) q.set("offset", String(params.offset));
  const qs = q.toString() ? `?${q}` : "";
  return apiFetch(`/api/admin/papers${qs}`);
}

export interface AdminGroup {
  id: number;
  name: string;
  created_at: string;
  member_count: number;
  last_joined_at: string | null;
  admin_emails: string[];
  pending_invite_count: number;
}

export interface AdminGroupMember {
  user_id: number;
  email: string;
  is_admin: boolean;
  joined_at: string;
}

export interface AdminGroupInvite {
  id: number;
  token: string;
  created_by_email: string;
  created_at: string;
  expires_at: string;
}

export interface AdminGroupDetail {
  id: number;
  name: string;
  created_at: string;
  members: AdminGroupMember[];
  pending_invites: AdminGroupInvite[];
}

export function getAdminGroups(): Promise<AdminGroup[]> {
  return apiFetch("/api/admin/groups");
}

export function getAdminGroup(groupId: number): Promise<AdminGroupDetail> {
  return apiFetch(`/api/admin/groups/${groupId}`);
}

export function deleteAdminGroup(groupId: number): Promise<void> {
  return apiFetch(`/api/admin/groups/${groupId}`, { method: "DELETE" });
}
