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
