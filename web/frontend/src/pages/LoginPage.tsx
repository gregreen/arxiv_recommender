import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { login, resendVerification } from "../api/auth";
import { useAuth } from "../AuthContext";
import { apiFetch, ApiError } from "../api/client";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pendingVerification, setPendingVerification] = useState(false);
  const [resendStatus, setResendStatus] = useState<string | null>(null);
  const { setUser } = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setPendingVerification(false);
    setLoading(true);
    try {
      await login(email, password);
      const data = await apiFetch<{ user_id: number; email: string; is_admin: boolean }>("/api/auth/me");
      setUser({ userId: data.user_id, email: data.email, isAdmin: data.is_admin });
      navigate("/");
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 403 && err.message === "verify_email_pending") {
        setPendingVerification(true);
      } else {
        setError(err instanceof Error ? err.message : "Login failed");
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    setResendStatus(null);
    try {
      await resendVerification(email);
      setResendStatus("Verification email sent. Please check your inbox.");
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 429) {
        setResendStatus(err.message);
      } else {
        setResendStatus("Could not send verification email. Please try again later.");
      }
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="bg-white shadow rounded-lg p-8 w-full max-w-sm">
        <h1 className="text-2xl font-bold mb-6 text-gray-800">Sign In</h1>
        {pendingVerification ? (
          <div className="mb-4 text-sm text-yellow-800 bg-yellow-50 border border-yellow-200 rounded p-3 space-y-2">
            <p>Your email address hasn't been verified yet. Please check your inbox for a verification link.</p>
            <button
              onClick={handleResend}
              className="text-blue-600 hover:underline text-sm font-medium"
            >
              Resend verification email
            </button>
            {resendStatus && <p className="text-gray-600">{resendStatus}</p>}
          </div>
        ) : error ? (
          <div className="mb-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
            {error}
          </div>
        ) : null}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoFocus
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded px-4 py-2 text-sm transition-colors"
          >
            {loading ? "Signing in…" : "Sign In"}
          </button>
        </form>
        <p className="mt-4 text-sm text-gray-500 text-center">
          Don't have an account?{" "}
          <Link to="/register" className="text-blue-600 hover:underline">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
