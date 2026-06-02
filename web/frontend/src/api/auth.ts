import { apiFetch } from "./client";

export async function login(email: string, password: string) {
  return apiFetch<{ user_id: number; email: string }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function register(email: string, password: string) {
  return apiFetch<{ message: string }>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function logout() {
  return apiFetch<{ message: string }>("/api/auth/logout", { method: "POST" });
}

export async function verifyEmail(token: string) {
  return apiFetch<{ message: string }>(`/api/auth/verify-email?token=${encodeURIComponent(token)}`);
}

export async function resendVerification(email: string) {
  return apiFetch<{ message: string }>("/api/auth/resend-verification", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export async function requestPasswordReset(email: string) {
  return apiFetch<{ message: string }>("/api/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export async function resetPassword(token: string, password: string) {
  return apiFetch<{ message: string }>("/api/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ token, password }),
  });
}

export async function getEmailEnabled() {
  return apiFetch<{ email_enabled: boolean; contact_email: string }>("/api/auth/email-enabled");
}

export async function deleteAccount(password: string): Promise<void> {
  return apiFetch<void>("/api/users/me", {
    method: "DELETE",
    body: JSON.stringify({ password }),
  });
}

export async function exportData(password: string): Promise<void> {
  const response = await fetch("/api/users/me/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ password }),
  });
  if (!response.ok) {
    let detail = "An error occurred. Please try again.";
    try {
      const data = await response.json();
      if (data?.detail) detail = data.detail;
    } catch {
      // ignore parse errors
    }
    const { ApiError } = await import("./client");
    throw new ApiError(response.status, detail);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "arxiv-recommender-data.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function changePassword(currentPassword: string, newPassword: string) {
  return apiFetch<{ message: string }>("/api/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}
