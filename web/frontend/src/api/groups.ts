import { apiFetch } from "./client";

export interface Group {
  id: number;
  name: string;
  created_at: string;
  is_admin?: boolean;
}

export interface GroupMember {
  user_id: number;
  email: string;
  is_admin: boolean;
  joined_at: string;
}

export interface GroupDetail extends Group {
  members: GroupMember[];
}

export interface GroupInvite {
  id: number;
  token: string;
  created_at: string;
  expires_at: string;
  remaining_uses: number;
}

export interface GroupRecommendationsResponse {
  group_id: number;
  group_name: string;
  window: string;
  method: string;
  count: number;
  group_member_count: number;
  active_member_count: number;
  results: GroupRecommendation[];
}

export interface GroupRecommendation {
  arxiv_id: string;
  title: string;
  authors: string[];
  published_date: string | null;
  score: number;
  rank: number;
  liked: number | null;
  generated_at: string | null;
}

export async function createGroup(name: string): Promise<GroupDetail> {
  return apiFetch<GroupDetail>("/api/groups", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export async function getMyGroups(): Promise<Group[]> {
  return apiFetch<Group[]>("/api/groups");
}

export async function getGroup(groupId: number): Promise<GroupDetail> {
  return apiFetch<GroupDetail>(`/api/groups/${groupId}`);
}

export async function getGroupRecommendations(
  groupId: number,
  window: string,
  method: string = "softmax_sum",
): Promise<GroupRecommendationsResponse> {
  return apiFetch<GroupRecommendationsResponse>(
    `/api/groups/${groupId}/recommendations?window=${encodeURIComponent(window)}&method=${encodeURIComponent(method)}`,
  );
}

export async function createInvite(groupId: number, maxUses: number = 1): Promise<GroupInvite> {
  return apiFetch<GroupInvite>(`/api/groups/${groupId}/invites`, {
    method: "POST",
    body: JSON.stringify({ max_uses: maxUses }),
  });
}

export async function listInvites(groupId: number): Promise<GroupInvite[]> {
  return apiFetch<GroupInvite[]>(`/api/groups/${groupId}/invites`);
}

export async function revokeInvite(groupId: number, inviteId: number): Promise<void> {
  await apiFetch<void>(`/api/groups/${groupId}/invites/${inviteId}`, {
    method: "DELETE",
  });
}

export async function removeMember(groupId: number, userId: number): Promise<void> {
  await apiFetch<void>(`/api/groups/${groupId}/members/${userId}`, {
    method: "DELETE",
  });
}

export async function makeAdmin(groupId: number, userId: number): Promise<GroupDetail> {
  return apiFetch<GroupDetail>(`/api/groups/${groupId}/members/${userId}`, {
    method: "PATCH",
  });
}

export async function deleteGroup(groupId: number): Promise<void> {
  await apiFetch<void>(`/api/groups/${groupId}`, {
    method: "DELETE",
  });
}

export interface JoinInfo {
  group_id: number;
  group_name: string;
}

export async function getJoinInfo(token: string): Promise<JoinInfo> {
  return apiFetch<JoinInfo>(`/api/groups/join-info?token=${encodeURIComponent(token)}`);
}

export async function joinGroup(token: string): Promise<GroupDetail> {
  return apiFetch<GroupDetail>(`/api/groups/join?token=${encodeURIComponent(token)}`, {
    method: "POST",
  });
}
