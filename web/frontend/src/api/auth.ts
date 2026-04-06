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
